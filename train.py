#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
setup_seed(SEED)

########### Import your packages below ##########
from data.dataset import E2EDataset, VOCAB
from trainer import TrainConfig


def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')
    parser.add_argument('--cdr', type=str, default=None, nargs='+', help='cdr to generate, L1/2/3, H1/2/3,(can be list, e.g., L3 H3) None for all including framework')
    parser.add_argument('--paratope', type=str, default='H3', nargs='+', help='cdrs to use as paratope')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='exponential decay from lr to final_lr')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--patience', type=int, default=1000, help='patience before early stopping (set with a large number to turn off early stopping)')
    parser.add_argument('--save_topk', type=int, default=10, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['dyMEAN', 'dyMEANOpt'],
                        help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--iter_round', type=int, default=3, help='Number of iterations for generation')

    # dyMEANOpt related
    parser.add_argument('--seq_warmup', type=int, default=0, help='Number of epochs before starting training sequence')

    # task setting
    parser.add_argument('--struct_only', action='store_true', help='Predict complex structure given the sequence')
    parser.add_argument('--bind_dist_cutoff', type=float, default=6.6, help='distance cutoff to decide the binding interface')

    # ablation
    parser.add_argument('--no_pred_edge_dist', action='store_true', help='Turn off edge distance prediction at the interface')
    parser.add_argument('--backbone_only', action='store_true', help='Model backbone only')
    parser.add_argument('--fix_channel_weights', action='store_true', help='Fix channel weights, may also for special use (e.g. antigen with modified AAs)')
    parser.add_argument('--no_memory', action='store_true', help='No memory passing')

    return parser.parse_args()


def main(args):
    ########### load your train / valid set ###########
    if (len(args.gpus) > 1 and int(os.environ['LOCAL_RANK']) == 0) or len(args.gpus) == 1:
        print_log(args)
        print_log(f'CDR type: {args.cdr}')
        print_log(f'Paratope: {args.paratope}')
        print_log('structure only' if args.struct_only else 'sequence & structure codesign')

    train_set = E2EDataset(args.train_set, cdr=args.cdr, paratope=args.paratope)
    valid_set = E2EDataset(args.valid_set, cdr=args.cdr, paratope=args.paratope)


    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(**vars(args))

    if args.model_type == 'dyMEAN':
        from trainer import dyMEANTrainer as Trainer
        from models import dyMEANModel
        model = dyMEANModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   iter_round=args.iter_round,
                   backbone_only=args.backbone_only,
                   fix_channel_weights=args.fix_channel_weights,
                   pred_edge_dist=not args.no_pred_edge_dist,
                   keep_memory=not args.no_memory,
                   cdr_type=args.cdr, paratope=args.paratope)
    elif args.model_type == 'dyMEANOpt':
        from trainer import dyMEANOptTrainer as Trainer
        from models import dyMEANOptModel
        model = dyMEANOptModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   fix_atom_weights=args.fix_channel_weights, cdr_type=args.cdr)
    else:
        raise NotImplemented(f'model {args.model_type} not implemented')

    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)

    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None
    config.local_rank = args.local_rank

    if args.local_rank == 0 or args.local_rank == -1:
        print_log(f'step per epoch: {step_per_epoch}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    
    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = parse()
    main(args)
