#!/bin/bash
# -*-coding=utf-8-*-
import os
import re
import sys
import linecache
import sys
import os

'''
@FileName: filter.py
@Author：liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description：配置过滤规则,过滤不符合1-2-3|(1-2)-3|2-(9-9)的字符
@File URL: https://github.com/liangzhicheng120/contest
'''


class Filter(object):
    def __init__(self):
        self._base = '{0}\\file\\filter\\'.format(sys.path[1])  # 根路径
        self._source = self._base + 'p-label.txt'  # 需要过滤源文件
        self._result = self._base + 'p-label-result.txt'  # 过滤后输出文件
        pass

    def main(self):
        _false = 0
        _true = 0
        with open(self._source, 'r') as s, open(self._result, 'w') as r:
            for i, line in enumerate(s):
                tf = self.rule(line)
                if tf:
                    _true += 1
                else:
                    _false += 1
                r.write('{0}\t{1}\t{2}\n'.format(i, line.strip(), tf))
        print('True:{0}\tFalse:{1}'.format(_true, _false))

    def rule(self, source):
        '''
        配置过滤规则
        :param source: 字符串
        :return: True or False
        '''
        source = source.strip()
        _len = len(source)
        if _len == 7:
            _flag1 = re.match(r'^(\()(\d)(\+|\-|\*)(\d)(\))(\+|\-|\*)(\d)$', source)  # (1-2)+3
            _flag2 = re.match(r'^(\d)(\+|\-|\*)(\()(\d)(\+|\-|\*)(\d)(\))$', source)  # 1-(2-3)
            return True if _flag1 or _flag2 else False
        elif _len == 5:
            _flag = re.match(r'^(\d)(\+|\-|\*)(\d)(\+|\-|\*)(\d)$', source)  # 1-2-3
            return True if _flag else False
        else:
            return False


if __name__ == '__main__':
    filter = Filter()
    filter.main()
