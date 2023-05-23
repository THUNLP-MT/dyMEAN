#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=3, act_func=nn.ReLU) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        layers = []
        for n in range(n_layers):
            if n == 0:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(act_func())
            elif n == n_layers - 1:
                layers.append(nn.Linear(hidden_size, output_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(act_func())
        self.model = nn.Sequential(*layers)

    def inference(self, h):
        return self.model(h).squeeze() # if output dimension is 1

    def forward(self, h, ddg):
        pred_ddg = self.inference(h)
        loss = F.smooth_l1_loss(pred_ddg, ddg.squeeze())
        return loss