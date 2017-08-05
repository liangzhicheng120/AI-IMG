#!/bin/bash
# -*-coding=utf-8-*-
import os
import re
import sys

'''
@FileName: filter.py
@Author：liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description：选出指定列
@File URL: https://github.com/liangzhicheng120/contest
'''
import sys


class Select(object):
    def __init__(self):
        self._base = sys.path[1]
        self._source = self._base + '\\p-labels.txt'  # 输入文件
        self._result = self._base + '\\file\\compare\\p.txt'  # 输出文件
        self._split = '\t'  # 每列的分隔符
        self._row = 2

    def run(self):
        with open(self._source, 'r') as s, open(self._result, 'w') as r:
            for line in s:
                content = line.strip().split(self._split)[self._row - 1] + '\n'
                r.write(content)


if __name__ == '__main__':
    select = Select()
    select.run()
