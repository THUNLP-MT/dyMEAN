#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

from models.dyMEAN.predictor import Predictor
from trainer.abs_trainer import Trainer, TrainConfig


class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_file) -> None:
        super().__init__()
        with open(json_file, 'r') as fin:
            self.data = [json.loads(l) for l in fin.readlines()]
        self.hidden_size = len(self.data[0]['hidden'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'ddg': torch.tensor(item['ddg'], dtype=torch.float),
            'h': torch.tensor(item['hidden'], dtype=torch.float)
        }


class PredictorTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, config):
        super().__init__(model, train_loader, valid_loader, config)

    def get_scheduler(self, optimizer):
        return None

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def share_step(self, batch, batch_idx, val):
        loss = self.model(**batch)
        suffix = 'Valid' if val else 'Train'
        self.log(f'Loss/{suffix}', loss, batch_idx, val)
        return loss


def _eval(ckpt, test_loader, device):
    model = torch.load(ckpt, map_location='cpu')
    model.to(device)
    model.eval()
    pred_ddgs, gt_ddgs = [], []
    with torch.no_grad():
        for batch in test_loader:
            ddgs = model.inference(batch['h'].to(device))
            pred_ddgs.extend(ddgs.tolist())
            gt_ddgs.extend(batch['ddg'].tolist())
    corr = np.corrcoef(pred_ddgs, gt_ddgs)[0][1]
    print(f'correlation on test set: {corr}')


def train(args):
    train_set = Dataset(args.train_set)
    valid_set = Dataset(args.valid_set)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    valid_loader = DataLoader(valid_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    model = Predictor(train_set.hidden_size, 256, 1, args.n_layers)
    config = TrainConfig(args.save_dir, args.lr, args.max_epoch,
                         patience=args.patience,
                         grad_clip=args.grad_clip,
                         save_topk=args.save_topk)
    trainer = PredictorTrainer(model, train_loader, valid_loader, config)
    trainer.train([args.gpu], -1)
    
    best_ckpt = trainer.topk_ckpt_map[0][-1]
    print(f'best checkpoint: {best_ckpt}')
    test_set = Dataset(args.test_set)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    _eval(best_ckpt, test_loader, torch.device(f'cuda:{args.gpu}'))


def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')
    parser.add_argument('--test_set', type=str, required=True, help='path to test set')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--patience', type=int, default=10, help='patience before early stopping')
    parser.add_argument('--save_topk', type=int, default=10, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)

    # device
    parser.add_argument('--gpu', type=int, required=True, help='gpu to use, -1 for cpu')
    
    # model
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')

    return parser.parse_args()


if __name__ == '__main__':
    train(parse())