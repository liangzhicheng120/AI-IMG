#!/bin/bash
# -*-coding:utf-8 -*-
import re
import os
import sys
import linecache
import codecs
import numpy as np
from demo.base import *

'''
# str_2_matrix 字符串转二进制矩阵
# matrix_2_str 矩阵转字符串
# test 测试
'''


class Util(Base):
    def str_2_matrix(self, str):
        '''
        字符串转二进制矩阵,每行代表一个字符
        :param str: 字符串
        :return:
        '''
        # result = map(lambda x: list(self.str_2_bin(x)), list(str))  # 不使用补零
        result = map(lambda x: list(self.add_zero(self.str_2_bin(x))), list(str.strip()))  # 使用补零
        result = map(lambda x: list(map(eval, x)), result)
        result = np.array(list(result))
        return result

    def matrix_2_str(self, matrix):
        '''
        矩阵转字符串
        :param matrix:矩阵
        :return:
        '''
        result = matrix.tolist()
        result = map(lambda x: ''.join(list(map(str, x))), result)
        result = map(self.bin_2_str, result)
        result = list(result)
        result = ''.join(result)
        return result

    def test(self, str):
        matrix = self.str_2_matrix(str)
        str = self.matrix_2_str(matrix)
        print('======================\n{0}\n{1}'.format(matrix, str))


if __name__ == '__main__':
    util = Util()
    util.test('我是中国人')  # 简体字
    util.test('123456')  # 数字
    util.test('!@#$%^&*()')  # 英文特殊字符
    util.test('！@#￥%……&*（）')  # 中文特殊字符
    util.test('犇猋骉麤毳淼掱焱垚赑')  # 生僻字
    util.test('にほんご、にっぽんご')  # 日语
    util.test('???')  # 韩语  暂不支持
