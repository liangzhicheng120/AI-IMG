#!/bin/bash
# -*-coding:utf-8-*-
import re
import os
import sys
import shutil
import linecache

'''
@FileName: remove.py
@Author: liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description: 移动图片到指定位置
@File URL: https://github.com/liangzhicheng120/contest
'''


class Remove(object):
    def __init__(self):
        self._base = '{0}\\file\\remove\\'.format(sys.path[1])  # 根路径
        self._index = self._base + 'false.txt'  # 需要移动的图片序号
        self._source = 'D:\\other\\image_contest_level_1_validate\\image_contest_level_1_validate\\{0}.png'  # 图片存放位置
        self._target = 'D:\\other\\'  # 移动目标位置
        pass

    def run(self):
        for i in linecache.getlines(self._index):
            self.copy(i)

    def copy(self, i):
        '''
        复制图片
        :param i: 图片序号
        :return:
        '''
        shutil.copy(self._source.format(i.strip()), self._target)


if __name__ == '__main__':
    remove = Remove()
    remove.run()
