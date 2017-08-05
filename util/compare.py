#!/bin/bash
# -*-coding:utf-8-*-
import re
import linecache
import sys

'''
@FileName: compare.py
@Author：liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description：两份文件按行对比
@File URL: https://github.com/liangzhicheng120/contest
'''


class Compare(object):
    def __init__(self):
        self._base = '{0}\\file\\compare\\'.format(sys.path[1])  # 根路径
        self._pfile = self._base + 'p.txt'  # 预测文本
        self._tfile = self._base + 't.txt'  # 真实文本
        self._result = self._base + 'compare.txt'  # 输出文本
        self._num = 1000  # 对比的行数
        pass

    def run(self):
        _false = 0
        with open(self._result, 'w') as r:
            for i in range(0, self._num):
                p = linecache.getline(self._pfile, i).strip()
                t = linecache.getline(self._tfile, i).strip()
                if p != t:
                    _false += 1
                    r.write('{0}\t{1}\t{2}\n'.format(p, t, p == t))
        print('acc:{0}'.format(1 - (_false / self._num)))



if __name__ == '__main__':
    compare = Compare()
    compare.run()
