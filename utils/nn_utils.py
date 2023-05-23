#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from data.pdb_utils import VOCAB
from evaluation.rmsd import kabsch_torch


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def graph_to_batch(tensor, batch_id, padding_value=0, mask_is_pad=True):
    '''
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    '''
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    batch = torch.ones((bs, max_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) * padding_value
    # generate pad mask: 1 for pad and 0 for data
    pad_mask = torch.zeros((bs, max_n + 1), dtype=torch.long, device=tensor.device)
    pad_mask[(torch.arange(bs, device=tensor.device), lengths)] = 1
    pad_mask = (torch.cumsum(pad_mask, dim=-1)[:, :-1]).bool()
    data_mask = torch.logical_not(pad_mask)
    # fill data
    batch[data_mask] = tensor
    mask = pad_mask if mask_is_pad else data_mask
    return batch, mask


def _knn_edges(X, AP, src_dst, atom_pos_pad_idx, k_neighbors, batch_info, given_dist=None):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param AP: [N, n_channel], atom position with pad type need to be ignored
    :param src_dst: [Ef, 2], full possible edges represented in (src, dst)
    :param given_dist: [Ef], given distance of edges
    '''
    offsets, batch_id, max_n, gni2lni = batch_info

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = X.shape[0]
    if given_dist is None:
        dist = X[src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = AP[src_dst] == atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * BIGINT  # [Ef, n_channel, n_channel]
        del pos_pad  # release memory
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
    else:
        dist = given_dist
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(src_dst[0], gni2lni[src_dst[1]])] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k_neighbors, dim=-1, largest=False)  # [N, topk]

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)

    dst = dst + offsets[batch_id[src]]  # mapping from local to global node index

    edges = torch.stack([src, dst])  # message passed from dst to src
    return edges  # [2, E]


class EdgeConstructor:
    def __init__(self, boa_idx, boh_idx, bol_idx, atom_pos_pad_idx, ag_seg_id) -> None:
        self.boa_idx, self.boh_idx, self.bol_idx = boa_idx, boh_idx, bol_idx
        self.atom_pos_pad_idx = atom_pos_pad_idx
        self.ag_seg_id = ag_seg_id

        # buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.row = None
        self.col = None
        self.row_global = None
        self.col_global = None
        self.row_seg = None
        self.col_seg = None
        self.offsets = None
        self.max_n = None
        self.gni2lni = None
        self.not_global_edges = None

    def get_batch_edges(self, batch_id):
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        return (row, col), (offsets, max_n, gni2lni)

    def _prepare(self, S, batch_id, segment_ids) -> None:
        (row, col), (offsets, max_n, gni2lni) = self.get_batch_edges(batch_id)

        # not global edges
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        
        # segment ids
        row_seg, col_seg = segment_ids[row], segment_ids[col]

        # add to buffer
        self.row, self.col = row, col
        self.offsets, self.max_n, self.gni2lni = offsets, max_n, gni2lni
        self.row_global, self.col_global = row_global, col_global
        self.not_global_edges = not_global_edges
        self.row_seg, self.col_seg = row_seg, col_seg

    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: same seg, not global
        select_edges = torch.logical_and(self.row_seg == self.col_seg, self.not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges

    def _construct_outer_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(self.row_seg != self.col_seg, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        outer_edges = _knn_edges(
            X, atom_pos, torch.stack([inter_all_row, inter_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return outer_edges

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(self.row_global, self.col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_seq_edges(self):
        row, col = self.row, self.col
        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph
            self.not_global_edges,  # not global edges (also ensure the edges are in the same segment)
            self.row_seg != self.ag_seg_id  # not epitope
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return seq_adj

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, k_neighbors, atom_pos, segment_ids):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # prepare inputs
        self._prepare(S, batch_id, segment_ids)

        ctx_edges, inter_edges = [], []

        # edges within chains
        inner_edges = self._construct_inner_edges(X, batch_id, k_neighbors, atom_pos)
        # edges between global nodes and normal/global nodes
        global_normal, global_global = self._construct_global_edges()
        # edges on the 1D sequence
        seq_edges = self._construct_seq_edges()

        # construct context edges
        ctx_edges = torch.cat([inner_edges, global_normal, global_global, seq_edges], dim=1)  # [2, E]

        # construct interaction edges
        inter_edges = self._construct_outer_edges(X, batch_id, k_neighbors, atom_pos)

        self._reset_buffer()
        return ctx_edges, inter_edges


class GMEdgeConstructor(EdgeConstructor):
    '''
    Edge constructor for graph matching (kNN internel edges and all bipartite edges)
    '''
    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: both in ag or ab, not global
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = torch.logical_and(row_is_ag == col_is_ag, self.not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = sequential_and(
            self.row_global, self.col_global, # self-loop has been deleted
            row_is_ag == col_is_ag)  # only inter-ag or inter-ab globals
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_outer_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible inter edges: one in ag and one in ab, not global
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = torch.logical_and(row_is_ag != col_is_ag, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        return torch.stack([inter_all_row, inter_all_col])  # [2, E]


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings

# embedding of amino acids. (default: concat residue embedding and atom embedding to one vector)
class AminoAcidEmbedding(nn.Module):
    '''
    [residue embedding + position embedding, mean(atom embeddings + atom position embeddings)]
    '''
    def __init__(self, num_res_type, num_atom_type, num_atom_pos, res_embed_size, atom_embed_size,
                 atom_pad_id=VOCAB.get_atom_pad_idx(), relative_position=True, max_position=192):  # max position (with IMGT numbering)
        super().__init__()
        self.residue_embedding = nn.Embedding(num_res_type, res_embed_size)
        if relative_position:
            self.res_pos_embedding = SinusoidalPositionEmbedding(res_embed_size)  # relative positional encoding
        else:
            self.res_pos_embedding = nn.Embedding(max_position, res_embed_size)  # absolute position encoding
        self.atom_embedding = nn.Embedding(num_atom_type, atom_embed_size)
        self.atom_pos_embedding = nn.Embedding(num_atom_pos, atom_embed_size)
        self.atom_pad_id = atom_pad_id
        self.eps = 1e-10  # for mean of atom embedding (some residues have no atom at all)
    
    def forward(self, S, RP, A, AP):
        '''
        :param S: [N], residue types
        :param RP: [N], residue positions
        :param A: [N, n_channel], atom types
        :param AP: [N, n_channel], atom positions
        '''
        res_embed = self.residue_embedding(S) + self.res_pos_embedding(RP)  # [N, res_embed_size]
        atom_embed = self.atom_embedding(A) + self.atom_pos_embedding(AP)   # [N, n_channel, atom_embed_size]
        atom_not_pad = (AP != self.atom_pad_id)  # [N, n_channel]
        denom = torch.sum(atom_not_pad, dim=-1, keepdim=True) + self.eps
        atom_embed = torch.sum(atom_embed * atom_not_pad.unsqueeze(-1), dim=1) / denom  # [N, atom_embed_size]
        return torch.cat([res_embed, atom_embed], dim=-1)  # [N, res_embed_size + atom_embed_size]


class AminoAcidFeature(nn.Module):
    def __init__(self, embed_size, relative_position=True, edge_constructor=EdgeConstructor, backbone_only=False) -> None:
        super().__init__()

        self.backbone_only = backbone_only

        # number of classes
        self.num_aa_type = len(VOCAB)
        self.num_atom_type = VOCAB.get_num_atom_type()
        self.num_atom_pos = VOCAB.get_num_atom_pos()

        # atom-level special tokens
        self.atom_mask_idx = VOCAB.get_atom_mask_idx()
        self.atom_pad_idx = VOCAB.get_atom_pad_idx()
        self.atom_pos_mask_idx = VOCAB.get_atom_pos_mask_idx()
        self.atom_pos_pad_idx = VOCAB.get_atom_pos_pad_idx()
        
        # embedding
        self.aa_embedding = AminoAcidEmbedding(
            self.num_aa_type, self.num_atom_type, self.num_atom_pos,
            embed_size, embed_size, self.atom_pad_idx, relative_position)

        # global nodes and mask nodes
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)
        self.mask_idx = VOCAB.get_mask_idx()

        # segment ids
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 1, 2, 3

        # atoms encoding
        residue_atom_type, residue_atom_pos = [], []
        backbone = [VOCAB.atom_to_idx(atom[0]) for atom in VOCAB.backbone_atoms]
        n_channel = VOCAB.MAX_ATOM_NUMBER if not backbone_only else 4
        special_mask = VOCAB.get_special_mask()
        for i in range(len(VOCAB)):
            if i == self.boa_idx or i == self.boh_idx or i == self.bol_idx or i == self.mask_idx:
                # global nodes
                residue_atom_type.append([self.atom_mask_idx for _ in range(n_channel)])
                residue_atom_pos.append([self.atom_pos_mask_idx for _ in range(n_channel)])
            elif special_mask[i] == 1:
                # other special token (pad)
                residue_atom_type.append([self.atom_pad_idx for _ in range(n_channel)])
                residue_atom_pos.append([self.atom_pos_pad_idx for _ in range(n_channel)])
            else:
                # normal amino acids
                sidechain_atoms = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                atom_type = backbone
                atom_pos = [VOCAB.atom_pos_to_idx(VOCAB.atom_pos_bb) for _ in backbone]
                if not backbone_only:
                    sidechain_atoms = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                    atom_type = atom_type + [VOCAB.atom_to_idx(atom[0]) for atom in sidechain_atoms]
                    atom_pos = atom_pos + [VOCAB.atom_pos_to_idx(atom[1]) for atom in sidechain_atoms]
                num_pad = n_channel - len(atom_type)
                residue_atom_type.append(atom_type + [self.atom_pad_idx for _ in range(num_pad)])
                residue_atom_pos.append(atom_pos + [self.atom_pos_pad_idx for _ in range(num_pad)])
        
        # mapping from residue to atom types and positions
        self.residue_atom_type = nn.parameter.Parameter(
            torch.tensor(residue_atom_type, dtype=torch.long),
            requires_grad=False)
        self.residue_atom_pos = nn.parameter.Parameter(
            torch.tensor(residue_atom_pos, dtype=torch.long),
            requires_grad=False)

        # sidechain geometry
        if not backbone_only:
            sc_bonds, sc_bonds_mask = [], []
            sc_chi_atoms, sc_chi_atoms_mask = [], []
            for i in range(len(VOCAB)):
                if special_mask[i] == 1:
                    sc_bonds.append([])
                    sc_chi_atoms.append([])
                else:
                    symbol = VOCAB.idx_to_symbol(i)
                    atom_type = VOCAB.backbone_atoms + VOCAB.get_sidechain_info(symbol)
                    atom2channel = { atom: i for i, atom in enumerate(atom_type) }
                    chi_atoms, bond_atoms = VOCAB.get_sidechain_geometry(symbol)
                    sc_chi_atoms.append(
                        [[atom2channel[atom] for atom in atoms] for atoms in chi_atoms]
                    )
                    bonds = []
                    for src_atom in bond_atoms:
                        for dst_atom in bond_atoms[src_atom]:
                            bonds.append((atom2channel[src_atom], atom2channel[dst_atom]))
                    sc_bonds.append(bonds)
            max_num_chis = max([len(chis) for chis in sc_chi_atoms])
            max_num_bonds = max([len(bonds) for bonds in sc_bonds])
            for i in range(len(VOCAB)):
                num_chis, num_bonds = len(sc_chi_atoms[i]), len(sc_bonds[i])
                num_pad_chis, num_pad_bonds = max_num_chis - num_chis, max_num_bonds - num_bonds
                sc_chi_atoms_mask.append(
                    [1 for _ in range(num_chis)] + [0 for _ in range(num_pad_chis)]
                )
                sc_bonds_mask.append(
                    [1 for _ in range(num_bonds)] + [0 for _ in range(num_pad_bonds)]
                )
                sc_chi_atoms[i].extend([[-1, -1, -1, -1] for _ in range(num_pad_chis)])
                sc_bonds[i].extend([(-1, -1) for _ in range(num_pad_bonds)])

            # mapping residues to their sidechain chi angle atoms and bonds
            self.sidechain_chi_angle_atoms = nn.parameter.Parameter(
                torch.tensor(sc_chi_atoms, dtype=torch.long),
                requires_grad=False)
            self.sidechain_chi_mask = nn.parameter.Parameter(
                torch.tensor(sc_chi_atoms_mask, dtype=torch.bool),
                requires_grad=False
            )
            self.sidechain_bonds = nn.parameter.Parameter(
                torch.tensor(sc_bonds, dtype=torch.long),
                requires_grad=False
            )
            self.sidechain_bonds_mask = nn.parameter.Parameter(
                torch.tensor(sc_bonds_mask, dtype=torch.bool),
                requires_grad=False
            )

        # edge constructor
        self.edge_constructor = edge_constructor(self.boa_idx, self.boh_idx, self.bol_idx, self.atom_pos_pad_idx, self.ag_seg_id)

    def _is_global(self, S):
        return sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)  # [N]

    def _construct_residue_pos(self, S):
        # construct residue position. global node is 1, the first residue is 2, ... (0 for padding)
        glbl_node_mask = self._is_global(S)
        glbl_node_idx = torch.nonzero(glbl_node_mask).flatten()  # [batch_size * 3] (boa, boh, bol)
        shift = F.pad(glbl_node_idx[:-1] - glbl_node_idx[1:] + 1, (1, 0), value=1) # [batch_size * 3]
        residue_pos = torch.ones_like(S)
        residue_pos[glbl_node_mask] = shift
        residue_pos = torch.cumsum(residue_pos, dim=0)
        return residue_pos

    def _construct_segment_ids(self, S):
        # construct segment ids. 1/2/3 for antigen/heavy chain/light chain
        glbl_node_mask = self._is_global(S)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == self.boa_idx), (glbl_nodes == self.boh_idx), (glbl_nodes == self.bol_idx)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = self.ag_seg_id, self.hc_seg_id, self.lc_seg_id
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)
        return segment_ids

    def _construct_atom_type(self, S):
        # construct atom types
        return self.residue_atom_type[S]
    
    def _construct_atom_pos(self, S):
        # construct atom positions
        return self.residue_atom_pos[S]

    @torch.no_grad()
    def get_sidechain_chi_angles_atoms(self, S):
        chi_angles_atoms = self.sidechain_chi_angle_atoms[S]  # [N, max_num_chis, 4]
        chi_mask = self.sidechain_chi_mask[S]  # [N, max_num_chis]
        return chi_angles_atoms, chi_mask

    @torch.no_grad()
    def get_sidechain_bonds(self, S):
        bonds = self.sidechain_bonds[S]  # [N, max_num_bond, 2]
        bond_mask = self.sidechain_bonds_mask[S]
        return bonds, bond_mask

    def update_globel_coordinates(self, X, S, atom_pos=None):
        X = X.clone()

        if atom_pos is None:  # [N, n_channel]
            atom_pos = self._construct_atom_pos(S)

        glbl_node_mask = self._is_global(S)
        chain_id = glbl_node_mask.long()
        chain_id = torch.cumsum(chain_id, dim=0)  # [N]
        chain_id[glbl_node_mask] = 0    # set global nodes to 0
        chain_id = chain_id.unsqueeze(-1).repeat(1, atom_pos.shape[-1])  # [N, n_channel]
        
        not_global = torch.logical_not(glbl_node_mask)
        not_pad = (atom_pos != self.atom_pos_pad_idx)[not_global]
        flatten_coord = X[not_global][not_pad]  # [N_atom, 3]
        flatten_chain_id = chain_id[not_global][not_pad]

        global_x = scatter_mean(
            src=flatten_coord, index=flatten_chain_id,
            dim=0, dim_size=glbl_node_mask.sum() + 1)  # because index start from 1
        X[glbl_node_mask] = global_x[1:].unsqueeze(1)

        return X

    def embedding(self, S, residue_pos=None, atom_type=None, atom_pos=None):
        '''
        :param S: [N], residue types
        '''
        if residue_pos is None:  # Residue positions in the chain
            residue_pos = self._construct_residue_pos(S)  # [N]

        if atom_type is None:  # Atom types in each residue
            atom_type = self.residue_atom_type[S]  # [N, n_channel]

        if atom_pos is None:   # Atom position in each residue
            atom_pos = self.residue_atom_pos[S]     # [N, n_channel]

        H = self.aa_embedding(S, residue_pos, atom_type, atom_pos)
        return H, (residue_pos, atom_type, atom_pos)

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, k_neighbors, atom_pos=None, segment_ids=None):

        # prepare inputs
        if atom_pos is None:  # Atom position in each residue (pad need to be ignored)
            atom_pos = self.residue_atom_pos[S]
        
        if segment_ids is None:
            segment_ids = self._construct_segment_ids(S)

        ctx_edges, inter_edges = self.edge_constructor.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos, segment_ids)

        return ctx_edges, inter_edges

    def forward(self, X, S, batch_id, k_neighbors):
        H, (_, _, atom_pos) = self.embedding(S)
        ctx_edges, inter_edges = self.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos=atom_pos)
        return H, (ctx_edges, inter_edges)


class SeparatedAminoAcidFeature(AminoAcidFeature):
    '''
    Separate embeddings of atoms and residues
    '''
    def __init__(self, embed_size, atom_embed_size, relative_position=True, edge_constructor=EdgeConstructor, fix_atom_weights=False, backbone_only=False) -> None:
        super().__init__(embed_size, relative_position=relative_position, edge_constructor=edge_constructor, backbone_only=backbone_only)
        atom_weights_mask = self.residue_atom_type == self.atom_pad_idx
        self.register_buffer('atom_weights_mask', atom_weights_mask)
        self.fix_atom_weights = fix_atom_weights
        if fix_atom_weights:
            atom_weights = torch.ones_like(self.residue_atom_type, dtype=torch.float)
        else:
            atom_weights = torch.randn_like(self.residue_atom_type, dtype=torch.float)
        atom_weights[atom_weights_mask] = 0
        self.atom_weight = nn.parameter.Parameter(atom_weights, requires_grad=not fix_atom_weights)
        self.zero_atom_weight = nn.parameter.Parameter(torch.zeros_like(atom_weights), requires_grad=False)
        
        # override
        self.aa_embedding = AminoAcidEmbedding(
            self.num_aa_type, self.num_atom_type, self.num_atom_pos,
            embed_size, atom_embed_size, self.atom_pad_idx, relative_position)
    
    def get_atom_weights(self, residue_types):
        weights = torch.where(
            self.atom_weights_mask,
            self.zero_atom_weight,
            self.atom_weight
        )  # [num_aa_classes, max_atom_number(n_channel)]
        if not self.fix_atom_weights:
            weights = F.normalize(weights, dim=-1)
        return weights[residue_types]

    def forward(self, X, S, batch_id, k_neighbors, residue_pos=None, smooth_prob=None, smooth_mask=None):
        if residue_pos is None:
            residue_pos = self._construct_residue_pos(S)  # [N]
        atom_type = self.residue_atom_type[S]  # [N, n_channel]
        atom_pos = self.residue_atom_pos[S]     # [N, n_channel]

        # residue embedding
        pos_embedding = self.aa_embedding.res_pos_embedding(residue_pos)
        H = self.aa_embedding.residue_embedding(S)
        if smooth_prob is not None:
            res_embeddings = self.aa_embedding.residue_embedding(
                torch.arange(smooth_prob.shape[-1], device=S.device, dtype=S.dtype)
            )  # [num_aa_type, embed_size]
            H[smooth_mask] = smooth_prob.mm(res_embeddings)
        H = H + pos_embedding

        # atom embedding
        atom_embedding = self.aa_embedding.atom_embedding(atom_type) +\
                         self.aa_embedding.atom_pos_embedding(atom_pos)
        atom_weights = self.get_atom_weights(S)
        
        ctx_edges, inter_edges = self.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos=atom_pos)
        return H, (ctx_edges, inter_edges), (atom_embedding, atom_weights)


class ProteinFeature:
    def __init__(self, backbone_only=False):
        self.backbone_only = backbone_only

    def _cal_sidechain_bond_lengths(self, S, X, aa_feature: AminoAcidFeature):
        bonds, bonds_mask = aa_feature.get_sidechain_bonds(S)
        n = torch.nonzero(bonds_mask)[:, 0]  # [Nbonds]
        src, dst = bonds[bonds_mask].T
        src_X, dst_X = X[(n, src)], X[(n, dst)]  # [Nbonds, 3]
        bond_lengths = torch.norm(dst_X - src_X, dim=-1)
        return bond_lengths

    def _cal_sidechain_chis(self, S, X, aa_feature: AminoAcidFeature):
        chi_atoms, chi_mask = aa_feature.get_sidechain_chi_angles_atoms(S)
        n = torch.nonzero(chi_mask)[:, 0]  # [Nchis]
        a0, a1, a2, a3 = chi_atoms[chi_mask].T  # [Nchis]
        x0, x1, x2, x3 = X[(n, a0)], X[(n, a1)], X[(n, a2)], X[(n, a3)]  # [Nchis, 3]
        u_0, u_1, u_2 = (x1 - x0), (x2 - x1), (x3 - x2)  # [Nchis, 3]
        # normals of the two planes
        n_1 = F.normalize(torch.cross(u_0, u_1), dim=-1)  # [Nchis, 3]
        n_2 = F.normalize(torch.cross(u_1, u_2), dim=-1)  # [Nchis, 3]
        cosChi = (n_1 * n_2).sum(-1)  # [Nchis]
        eps = 1e-7
        cosChi = torch.clamp(cosChi, -1 + eps, 1 - eps)
        return cosChi

    def _cal_backbone_bond_lengths(self, X, seg_id):
        # loss of backbone (...N-CA-C(O)-N...) bond length
        # N-CA, CA-C, C=O
        bl1 = torch.norm(X[:, 1:4] - X[:, :3], dim=-1)  # [N, 3], (N-CA), (CA-C), (C=O)
        # C-N
        bl2 = torch.norm(X[1:, 0] - X[:-1, 2], dim=-1)  # [N-1]
        same_chain_mask = seg_id[1:] == seg_id[:-1]
        bl2 = bl2[same_chain_mask]
        bl = torch.cat([bl1.flatten(), bl2], dim=0)
        return bl

    def _cal_angles(self, X, seg_id):
        ori_X = X
        X = X[:, :3].reshape(-1, 3)  # [N * 3, 3], N, CA, C
        U = F.normalize(X[1:] - X[:-1], dim=-1)  # [N * 3 - 1, 3]

        # 1. dihedral angles
        u_2, u_1, u_0 = U[:-2], U[1:-1], U[2:]   # [N * 3 - 3, 3]
        # backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        # angle between normals
        eps = 1e-7
        cosD = (n_2 * n_1).sum(-1)  # [(N-1) * 3]
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        # D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        seg_id_atom = seg_id.repeat(1, 3).flatten()  # [N * 3]
        same_chain_mask = sequential_and(
            seg_id_atom[:-3] == seg_id_atom[1:-2],
            seg_id_atom[1:-2] == seg_id_atom[2:-1],
            seg_id_atom[2:-1] == seg_id_atom[3:]
        )  # [N * 3 - 3]
        # D = D[same_chain_mask]
        cosD = cosD[same_chain_mask]

        # 2. bond angles (C_{n-1}-N, N-CA), (N-CA, CA-C), (CA-C, C=O), (CA-C, C-N_{n+1}), (O=C, C-Nn)
        u_0, u_1 = U[:-1], U[1:]  # [N*3 - 2, 3]
        cosA1 = ((-u_0) * u_1).sum(-1)  # [N*3 - 2], (C_{n-1}-N, N-CA), (N-CA, CA-C), (CA-C, C-N_{n+1})
        same_chain_mask = sequential_and(
            seg_id_atom[:-2] == seg_id_atom[1:-1],
            seg_id_atom[1:-1] == seg_id_atom[2:]
        )
        cosA1 = cosA1[same_chain_mask]  # [N*3 - 2 * num_chain]
        u_co = F.normalize(ori_X[:, 3] - ori_X[:, 2], dim=-1)  # [N, 3], C=O
        u_cca = -U[1::3]  # [N, 3], C-CA
        u_cn = U[2::3] # [N-1, 3], C-N_{n+1}
        cosA2 = (u_co * u_cca).sum(-1)  # [N], (C=O, C-CA)
        cosA3 = (u_co[:-1] * u_cn).sum(-1)  # [N-1], (C=O, C-N_{n+1})
        same_chain_mask = (seg_id[:-1] == seg_id[1:]) # [N-1]
        cosA3 = cosA3[same_chain_mask]
        cosA = torch.cat([cosA1, cosA2, cosA3], dim=-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)

        return cosD, cosA

    def coord_loss(self, pred_X, true_X, batch_id, atom_mask, reference=None):
        pred_bb, true_bb = pred_X[:, :4], true_X[:, :4]
        bb_mask = atom_mask[:, :4]
        true_X = true_X.clone()
        ops = []

        align_obj = pred_bb if reference is None else reference[:, :4]

        for i in range(torch.max(batch_id) + 1):
            is_cur_graph = batch_id == i
            cur_bb_mask = bb_mask[is_cur_graph]
            _, R, t = kabsch_torch(
                true_bb[is_cur_graph][cur_bb_mask],
                align_obj[is_cur_graph][cur_bb_mask],
                requires_grad=True)
            true_X[is_cur_graph] = torch.matmul(true_X[is_cur_graph], R.T) + t
            ops.append((R.detach(), t.detach()))

        xloss = F.smooth_l1_loss(
            pred_X[atom_mask], true_X[atom_mask],
            reduction='sum') / atom_mask.sum()  # atom-level loss
        bb_rmsd = torch.sqrt(((pred_X[:, :4] - true_X[:, :4]) ** 2).sum(-1).mean(-1))  # [N]
        return xloss, bb_rmsd, ops

    def structure_loss(self, pred_X, true_X, S, cmask, batch_id, xloss_mask, aa_feature, full_profile=False, reference=None):
        atom_pos = aa_feature._construct_atom_pos(S)[cmask]
        seg_id = aa_feature._construct_segment_ids(S)[cmask]
        atom_mask = atom_pos != aa_feature.atom_pos_pad_idx
        atom_mask = torch.logical_and(atom_mask, xloss_mask[cmask])

        pred_X, true_X, batch_id = pred_X[cmask], true_X[cmask], batch_id[cmask]

        # loss of absolute coordinates
        xloss, bb_rmsd, ops = self.coord_loss(pred_X, true_X, batch_id, atom_mask, reference)

        # loss of backbone (...N-CA-C(O)-N...) bond length
        true_bl = self._cal_backbone_bond_lengths(true_X, seg_id)
        pred_bl = self._cal_backbone_bond_lengths(pred_X, seg_id)
        bond_loss = F.smooth_l1_loss(pred_bl, true_bl)

        # loss of backbone dihedral angles
        if full_profile:
            true_cosD, true_cosA = self._cal_angles(true_X, seg_id)
            pred_cosD, pred_cosA = self._cal_angles(pred_X, seg_id)
            angle_loss = F.smooth_l1_loss(pred_cosD, true_cosD)
            bond_angle_loss = F.smooth_l1_loss(pred_cosA, true_cosA)

        S = S[cmask]
        if self.backbone_only:
            sc_bond_loss, sc_chi_loss = 0, 0
        else:
            # loss of sidechain bonds
            true_sc_bl = self._cal_sidechain_bond_lengths(S, true_X, aa_feature)
            pred_sc_bl = self._cal_sidechain_bond_lengths(S, pred_X, aa_feature)
            sc_bond_loss = F.smooth_l1_loss(pred_sc_bl, true_sc_bl)

            # loss of sidechain chis
            if full_profile:
                true_sc_chi = self._cal_sidechain_chis(S, true_X, aa_feature)
                pred_sc_chi = self._cal_sidechain_chis(S, pred_X, aa_feature)
                sc_chi_loss = F.smooth_l1_loss(pred_sc_chi, true_sc_chi)

        # exerting constraints on bond lengths only is sufficient
        violation_loss = bond_loss + sc_bond_loss
        loss = xloss + violation_loss

        if full_profile:
            details = (xloss, bond_loss, bond_angle_loss, angle_loss, sc_bond_loss, sc_chi_loss)
        else:
            details = (xloss, bond_loss, sc_bond_loss)

        return loss, details, bb_rmsd, ops


class SeperatedCoordNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor(0)
        self.std = torch.tensor(10)
        self.mean = nn.parameter.Parameter(self.mean, requires_grad=False)
        self.std = nn.parameter.Parameter(self.std, requires_grad=False)
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)

    def normalize(self, X):
        X = (X - self.mean) / self.std
        return X

    def unnormalize(self, X):
        X = X * self.std + self.mean
        return X

    def centering(self, X, S, batch_id, aa_feature: AminoAcidFeature):
        # centering antigen and antibody separatedly
        segment_ids = aa_feature._construct_segment_ids(S)
        not_bol = S != aa_feature.bol_idx
        tmp_S = S[not_bol]
        tmp_X = aa_feature.update_globel_coordinates(X[not_bol], tmp_S)
        self.ag_centers = tmp_X[tmp_S == aa_feature.boa_idx][:, 0]
        self.ab_centers = tmp_X[tmp_S == aa_feature.boh_idx][:, 0]

        is_ag = segment_ids == aa_feature.ag_seg_id
        is_ab = torch.logical_not(is_ag)

        # compose centers
        centers = torch.zeros(X.shape[0], X.shape[-1], dtype=X.dtype, device=X.device)
        centers[is_ag] = self.ag_centers[batch_id[is_ag]]
        centers[is_ab] = self.ab_centers[batch_id[is_ab]]
        X = X - centers.unsqueeze(1)
        self.is_ag, self.is_ab = is_ag, is_ab
        return X

    def uncentering(self, X, batch_id, _type=1):
        if _type == 0:
            # type 0: [N, 3]
            X = X.unsqueeze(1) # then it is type 1
        
        if _type == 0 or _type == 1:
            # type 1: [N, n_channel, 3]
            centers = torch.zeros(X.shape[0], X.shape[-1], dtype=X.dtype, device=X.device)
            centers[self.is_ag] = self.ag_centers[batch_id[self.is_ag]]
            centers[self.is_ab] = self.ab_centers[batch_id[self.is_ab]]
            X = X + centers.unsqueeze(1)
        elif _type == 2:
            # type 2: [2, bs, K, 3], X[0] for antigen, X[1] for antibody
            centers = torch.stack([self.ag_centers, self.ab_centers], dim=0)  # [2, bs, 3]
            X = X + centers.unsqueeze(-2)
        elif _type == 3:
            # type 3: [2, Ef, 3], X[0] for antigen, X[1] for antibody
            centers = torch.stack([self.ag_centers[batch_id], self.ab_centers[batch_id]], dim=0)
            X = X + centers
        elif _type == 4:
            # type 4: [N, n_channel, 3], but all uncentering to the center of antigen
            centers = self.ag_centers[batch_id]
            X = X + centers.unsqueeze(1)
        else:
            raise NotImplementedError(f'uncentering for type {_type} not implemented')

        if _type == 0:
            X = X.squeeze(1)
        return X

    def clear_cache(self):
        self.ag_centers, self.ab_centers, self.is_ag, self.is_ab = None, None, None, None