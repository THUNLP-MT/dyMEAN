#!/usr/bin/python
# -*- coding:utf-8 -*-
def try_catch_oom(exec_func, *args, **kwargs):
    try:
        res = exec_func(*args, **kwargs)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            return None
        else:
            raise e
    return res
