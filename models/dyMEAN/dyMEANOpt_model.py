#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std

from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import EdgeConstructor

from ..modules.am_egnn import AMEGNN

'''
Masked 1D & 3D language model
Add noise to ground truth 3D coordination
Add mask to 1D sequence
'''
class dyMEANOptModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1, struct_only=False,
                 fix_atom_weights=False, cdr_type='H3', relative_position=False) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        self.cdr_type = cdr_type  # only to indicate the usage of the model

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            relative_position=relative_position,
            edge_constructor=EdgeConstructor,
            fix_atom_weights=fix_atom_weights)
        self.protein_feature = ProteinFeature()
        
        self.memory_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, embed_size)
        )
        self.struct_only = struct_only
        if not struct_only:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )
        else:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.gnn = AMEGNN(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False)
        
        # training related cache
        self.start_seq_training = False
        self.batch_constants = {}

    def init_mask(self, X, S, cmask, smask, init_noise):
        if not self.struct_only:
            S[smask] = self.mask_id
        coords = X[cmask]
        noise = torch.randn_like(coords) if init_noise is None else init_noise
        X = X.clone()
        X[cmask] = coords + noise
        return X, S

    def message_passing(self, X, S, residue_pos, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None):
        # embeddings
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)
        inter_edges = self._get_binding_edges(X, S, inter_edges)
        edges = torch.cat([ctx_edges, inter_edges], dim=1)

        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        # update coordination of the global node
        X = self.aa_feature.update_globel_coordinates(X, S)

        H, pred_X = self.gnn(H_0, X, edges,
                             channel_attr=atom_embeddings,
                             channel_weights=atom_weights)


        pred_logits = None if self.struct_only else self.ffn_residue(H)

        return pred_logits, pred_X, H # [N, num_classes], [N, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def _prepare_batch_constants(self, S, lengths):
        # generate batch id
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1

        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids

        # interface relatd
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        self.batch_constants['is_ag'] = is_ag
    
    @torch.no_grad()
    def _get_binding_edges(self, X, S, inter_edges):
        atom_pos = self.aa_feature._construct_atom_pos(S)
        src_dst = inter_edges.T
        dist = X[src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * 1e10  # [Ef, n_channel, n_channel]
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
        is_binding = dist <= self.bind_dist_cutoff
        return src_dst[is_binding].T

    def _clean_batch_constants(self):
        self.batch_constants = {}

    def _forward(self, X, S, cmask, smask, residue_pos, init_noise=None):
        batch_id = self.batch_constants['batch_id']

        # mask sequence and add noise to ground truth coordinates
        X, S = self.init_mask(X, S, cmask, smask, init_noise)

        # update center
        X = self.aa_feature.update_globel_coordinates(X, S)

        # sequence and structure loss
        r_pred_S_logits, pred_S_dist = [], None
        memory_H = None
        # message passing
        for t in range(self.round):
            pred_S_logits, pred_X, H = self.message_passing(X, S, residue_pos, batch_id, t, memory_H, pred_S_dist, smask)
            r_pred_S_logits.append((pred_S_logits, smask))
            memory_H = H
            # 1. update X
            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_globel_coordinates(X, S)

            if not self.struct_only:
                # 2. update S
                S = S.clone()
                if t == self.round - 1:
                    S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_S_logits[smask], dim=-1)

        if self.struct_only:
            # predicted rmsd
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()  # [N_ab]
        else:
            prmsd = None

        return H, S, r_pred_S_logits, pred_X, prmsd

    def forward(self, X, S, cmask, smask, residue_pos, lengths, xloss_mask, context_ratio=0, seq_alpha=1):
        '''
        :param bind_ag: [N_bind], node idx of binding residues in antigen
        :param bind_ab: [N_bind], node idx of binding residues in antibody
        :param bind_ag_X: [N_bind, 3], coordinations of the midpoint of binding pairs relative to ag
        :param bind_ab_X: [N_bind, 3], coordinations of the midpoint of binding pairs relative to ab
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training
        :param seq_alpha: float, weight of SNLL, linearly increase from 0 to 1 at warmup phase
        '''
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, lengths)
        batch_id = self.batch_constants['batch_id']

        # provide some ground truth for annealing sequence training
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)

        # get results
        H, pred_S, r_pred_S_logits, pred_X, prmsd = self._forward(X, S, cmask, smask, residue_pos)

        # sequence negtive log likelihood
        snll, total = 0, 0
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total

        # coordination loss
        struct_loss, struct_loss_details, bb_rmsd, _ = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss# + prmsd_i_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss
        loss = seq_alpha * snll + struct_loss + (0 if pdev_loss is None else pdev_loss)

        self._clean_batch_constants()

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]

        return loss, (snll, aar), (struct_loss, *struct_loss_details), (pdev_loss, prmsd_loss)

    def sample(self, X, S, cmask, smask, residue_pos, lengths, return_hidden=False, init_noise=None):
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        s_batch_id = batch_id[smask]

        # generate
        H, pred_S, r_pred_S_logits, pred_X, _ = self._forward(X, S, cmask, smask, residue_pos, init_noise)

        # PPL
        if not self.struct_only:
            S_logits = r_pred_S_logits[-1][0][smask]
            S_dists = torch.softmax(S_logits, dim=-1)
            pred_S[smask] = torch.multinomial(S_dists, num_samples=1).squeeze()
            S_probs = S_dists[torch.arange(s_batch_id.shape[0], device=S_dists.device), pred_S[smask]]
            nlls = -torch.log(S_probs)
            ppl = scatter_mean(nlls, s_batch_id)  # [batch_size]
        else:
            ppl = torch.zeros(batch_size, device=pred_S.device)

        # 1. set generated part
        gen_X[cmask] = pred_X[cmask]
        if not self.struct_only:
            gen_S[smask] = pred_S[smask]
        
        self._clean_batch_constants()

        if return_hidden:
            return gen_X, gen_S, ppl, H
        return gen_X, gen_S, ppl

    def optimize_sample(self, X, S, cmask, smask, residue_pos, lengths, predictor, opt_steps=10, init_noise=None, mask_only=False):
        self._prepare_batch_constants(S, lengths)
        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        opt_mask = smask if mask_only else cmask
        # noise_batch_id = batch_id[smask].unsqueeze(1).repeat(1, X.shape[1] * X.shape[2]).flatten()
        # noise_batch_id = batch_id[cmask].unsqueeze(1).repeat(1, X.shape[1] * X.shape[2]).flatten()
        noise_batch_id = batch_id[opt_mask].unsqueeze(1).repeat(1, X.shape[1] * X.shape[2]).flatten()

        final_X, final_S = X.clone(), S.clone()
        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10

        all_noise = torch.randn_like(X, requires_grad=False)
        # init_noise = torch.randn_like(X[smask], requires_grad=True)
        # init_noise = torch.randn_like(X[cmask], requires_grad=True)
        init_noise = torch.randn_like(X[opt_mask], requires_grad=True)
        optimizer = torch.optim.Adam([init_noise], lr=1.0)
        optimizer.zero_grad()
        
        for i in range(opt_steps):
            all_noise = all_noise.detach()
            X, S, cmask, smask, residue_pos, lengths = X.clone(), S.clone(), cmask.clone(), smask.clone(), residue_pos.clone(), lengths.clone()
            # all_noise[smask] = init_noise
            # all_noise[cmask] = init_noise
            all_noise[opt_mask] = init_noise
            gen_X, gen_S, _, H = self.sample(X, S, cmask, smask, residue_pos, lengths, return_hidden=True, init_noise=all_noise[cmask])
            h = scatter_mean(H, batch_id, dim=0)
            pmetric = predictor.inference(h)

            # use KL to regularize noise
            mean = scatter_mean(init_noise.flatten(), noise_batch_id)  # [bs]
            std = scatter_std(init_noise.flatten(), noise_batch_id)
            # std, mean = torch.std_mean(init_noise.flatten())
            kl = -0.5 * (1 + 2 * torch.log(std) - std ** 2 - mean ** 2)

            (pmetric + kl).sum().backward()
            pmetric = pmetric.detach()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                update = pmetric < best_metric
                cupdate = cmask & update[batch_id]
                supdate = smask & update[batch_id]
                # update pmetric best history
                best_metric[update] = pmetric[update]

                final_X[cupdate] = gen_X[cupdate].detach()
                if not self.struct_only:
                    final_S[supdate] = gen_S[supdate].detach()
            
        return final_X, final_S, best_metric


if __name__ == '__main__':
    torch.random.manual_seed(0)
    # equivariance test
    embed_size, hidden_size = 64, 128
    n_channel, d = 14, 3
    n_aa_type = 20
    scale = 10
    dtype = torch.float
    device = torch.device('cuda:0')
    model = dyMEANOptModel(embed_size, hidden_size, n_channel,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   k_neighbors=9, bind_dist_cutoff=6.6, n_layers=3)
    model.to(device)
    model.eval()

    ag_len, h_len, l_len = 48, 120, 110
    center_x = torch.randn(3, 1, n_channel, d, device=device, dtype=dtype) * scale
    ag_X = torch.randn(ag_len, n_channel, d, device=device, dtype=dtype) * scale
    h_X = torch.randn(h_len, n_channel, d, device=device, dtype=dtype) * scale
    l_X = torch.randn(l_len, n_channel, d, device=device, dtype=dtype) * scale

    X = torch.cat([center_x[0], ag_X, center_x[1], h_X, center_x[2], l_X], dim=0)
    S = torch.cat([torch.tensor([model.aa_feature.boa_idx], device=device),
                   torch.randint(low=0, high=20, size=(ag_len,), device=device),
                   torch.tensor([model.aa_feature.boh_idx], device=device),
                   torch.randint(low=0, high=20, size=(h_len,), device=device),
                   torch.tensor([model.aa_feature.bol_idx], device=device),
                   torch.randint(low=0, high=20, size=(l_len,), device=device)], dim=0)
    cmask = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + [1 for _ in range(h_len)] + [0] + [1 for _ in range(l_len)], device=device).bool()
    smask = torch.zeros_like(cmask)
    smask[ag_len+10:ag_len+20] = 1
    residue_pos = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + list(range(1, h_len + 1)) + [0] + list(range(1, l_len + 1)), device=device)
    lengths = torch.tensor([ag_len + h_len + l_len + 3], device=device)

    torch.random.manual_seed(1)
    gen_X, _, _ = model.sample(X, S, cmask, smask, residue_pos, lengths)
    tmpx1 = model.tmpx

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q, t = U.mm(V), torch.randn(3, device=device)

    X = torch.matmul(X, Q) + t

    # this is f(Qx+t)
    model.op = (Q, t)
    torch.random.manual_seed(1)
    gen_op_X, _, _ = model.sample(X, S, cmask, smask, residue_pos, lengths)
    tmpx2 = model.tmpx
    gt_tmpx = torch.matmul(tmpx1, Q) + t
    error = torch.abs(gt_tmpx[:, :4] - tmpx2[:, :4]).sum(-1).flatten().mean()
    print(error.item())
    assert error < 1e-3
    print('independent equivariance check passed')


    gt_op_X = torch.matmul(gen_X, Q) + t

    error = torch.abs(gt_op_X[cmask][:, :4] - gen_op_X[cmask][:, :4]).sum(-1).flatten().mean()
    print(error.item())
    assert error < 1e-3
    print('independent equivariance check passed')


