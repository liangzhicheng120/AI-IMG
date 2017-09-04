#!/bin/bash
# -*-coding:gbk -*-
import re
import os
import sys
import linecache
import codecs
import numpy as np


class StringUtil(object):
    def __init__(self):
        pass

    def bin_2_str(self, str):
        '''
        二进制转字符串
        :param str: 二进制数字型字符串
        :return:
        '''
        return chr(self.bin_2_dec(str))

    def bin_2_dec(self, str):
        '''
        二进制转十进制
        :param str:二进制数字型字符串
        :return:
        '''
        return int(str, 2);

    def str_2_bin(self, str):
        '''
        字符串转二进制
        :param str: 任意字符串
        :return: 二进制编码
        '''
        return '{0:b}'.format((ord(str)))

    def read_file(self, fileName):
        '''
        读取文件,统一转换成unicode编码
        :param fileName:文件名
        :return:
        '''
        f = codecs.open(fileName, encoding='utf-8')
        try:
            f.read(1)
        except:
            f = codecs.open(fileName, encoding='gbk')
        return f

    def add_zero(self, str, size=3):
        '''
        补零
        英文1字节,汉字3字节,生僻字4-6字节,默认3字节
        :param str: 字符串
        :param num: 数字
        :return:
        '''
        return str.zfill(size * 8)

    def str_2_matrix(self, str):
        '''
        字符串转二进制矩阵
        :param str: 字符串
        :return:
        '''
        result = map(self.str_2_bin, list(str))
        result = map(int, result)
        result = list(result)
        result = np.array(result)
        return result

    def matrix_2_str(self, matrix):
        '''
        矩阵转字符串
        :param matrix:
        :return:
        '''
        result = matrix.tolist()
        result = map(lambda x: self.bin_2_str(str(x)), result)
        result = list(result)
        return result

    def test(self, str):
        matrix = self.str_2_matrix(str)
        str = self.matrix_2_str(matrix)
        print('======================')
        print(matrix)
        print(str)


if __name__ == '__main__':
    stringUtil = StringUtil()
    stringUtil.test('我是中国人')  # 简体字
    stringUtil.test('123456')  # 数字
    stringUtil.test('!@#$%^&*()')  # 英文特殊字符
    stringUtil.test('！@#￥%……&*（）')  # 中文特殊字符
    stringUtil.test('犇猋骉麤毳淼掱焱垚赑')  # 生僻字
    stringUtil.test('にほんご、にっぽんご')  # 日语
    stringUtil.test('???')  # 韩语  暂不支持
