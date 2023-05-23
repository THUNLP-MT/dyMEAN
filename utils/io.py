#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd


def read_csv(fpath, sep=','):
    heads, entries = [], []
    df = pd.read_csv(fpath, sep=sep)
    heads = list(df.columns)
    for rid in range(len(df)):
        entry = []
        for h in heads:
            entry.append(str(df[h][rid]))
        entries.append(entry)
    return heads, entries