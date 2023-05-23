#!/usr/bin/python
# -*- coding:utf-8 -*-
import requests

from .logger import print_log


def url_get(url, tries=3):
    for i in range(tries):
        if i > 0:
            print_log(f'Trying for the {i + 1} times', level='WARN')
        try:
            res = requests.get(url)
        except ConnectionError:
            continue
        if res.status_code == 200:
            return res
    print_log(f'Get {url} failed', level='WARN')
    return None