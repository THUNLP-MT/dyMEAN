#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import cos, pi, log, exp
from random import random
import torch
from .abs_trainer import Trainer


class dyMEANOptTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        self.seq_warmup = config.seq_warmup
        super().__init__(model, train_loader, valid_loader, config)

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def get_scheduler(self, optimizer):
        log_alpha = self.log_alpha
        lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'scheduler': scheduler,
            'frequency': 'batch'
        }

    def train_step(self, batch, batch_idx):
        # batch['seq_alpha'] = min((self.epoch + 1) / (self.seq_warmup + 1), 1) # linear
        batch['seq_alpha'] = 1.0 - 1.0 * self.epoch / self.config.max_epoch
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        batch['seq_alpha'] = 1
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def get_context_ratio(self):
        ratio = random() * 0.9
        return ratio

    def share_step(self, batch, batch_idx, val=False):
        del batch['paratope_mask']
        del batch['template']
        batch['context_ratio'] = self.get_context_ratio()
        loss, seq_detail, structure_detail, pdev_detail = self.model(**batch)
        snll, aar = seq_detail
        struct_loss, xloss, bond_loss, sc_bond_loss = structure_detail
        pdev_loss, prmsd_loss = pdev_detail

        log_type = 'Validation' if val else 'Train'

        self.log(f'Overall/Loss/{log_type}', loss, batch_idx, val)

        self.log(f'Seq/SNLL/{log_type}', snll, batch_idx, val)
        self.log(f'Seq/AAR/{log_type}', aar, batch_idx, val)

        self.log(f'Struct/StructLoss/{log_type}', struct_loss, batch_idx, val)
        self.log(f'Struct/XLoss/{log_type}', xloss, batch_idx, val)
        self.log(f'Struct/BondLoss/{log_type}', bond_loss, batch_idx, val)
        self.log(f'Struct/SidechainBondLoss/{log_type}', sc_bond_loss, batch_idx, val)

        if pdev_loss is not None:
            self.log(f'PDev/PDevLoss/{log_type}', pdev_loss, batch_idx, val)
            self.log(f'PDev/PRMSDLoss/{log_type}', prmsd_loss, batch_idx, val)

        if not val:
            lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
            lr = lr[0]
            self.log('lr', lr, batch_idx, val)
            self.log('context_ratio', batch['context_ratio'], batch_idx, val)
            self.log('seq_alpha', batch['seq_alpha'], batch_idx, val)
        return loss
