#!/bin/bash
# -*-coding:gbk -*-
import re
import os
import sys
import linecache
import codecs
import numpy as np

'''
    # bin_2_str ������ת�ַ���
    # bin_2_dec ������תʮ����
    # str_2_bin �ַ���ת������
    # read_file ��ȡ�ļ�,ͳһת����unicode����
    # add_zero ����
    # str_2_matrix �ַ���ת�����ƾ���
    # matrix_2_str ����ת�ַ���
    # test ����
'''


class StringUtil(object):
    def __init__(self):
        pass

    def bin_2_str(self, str):
        '''
        ������ת�ַ���
        :param str: �������������ַ���
        :return:
        '''
        return chr(self.bin_2_dec(str))

    def bin_2_dec(self, str):
        '''
        ������תʮ����
        :param str:�������������ַ���
        :return:
        '''
        return int(str, 2);

    def str_2_bin(self, str):
        '''
        �ַ���ת������
        :param str: �����ַ���
        :return: �����Ʊ���
        '''
        return '{0:b}'.format((ord(str)))

    def read_file(self, fileName):
        '''
        ��ȡ�ļ�,ͳһת����unicode����
        :param fileName:�ļ���
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
        ����
        Ӣ��1�ֽ�,����3�ֽ�,��Ƨ��4-6�ֽ�,Ĭ��3�ֽ�
        :param str: �ַ���
        :param num: ����
        :return:
        '''
        return str.zfill(size * 8)

    def str_2_matrix(self, str):
        '''
        �ַ���ת�����ƾ���,ÿ�д���һ���ַ�
        :param str: �ַ���
        :return:
        '''
        # result = map(lambda x: list(self.str_2_bin(x)), list(str))  # ��ʹ�ò���
        result = map(lambda x: list(self.add_zero(self.str_2_bin(x))), list(str.strip()))  # ʹ�ò���
        result = map(lambda x: list(map(eval, x)), result)
        result = np.array(list(result))
        return result

    def matrix_2_str(self, matrix):
        '''
        ����ת�ַ���
        :param matrix:
        :return:
        '''
        result = matrix.tolist()
        result = map(lambda x: ''.join(list(map(str, x))), result)
        result = map(self.bin_2_str, result)
        result = list(result)
        return result

    def test(self, str):
        matrix = self.str_2_matrix(str)
        str = self.matrix_2_str(matrix)
        print('======================')
        print(matrix)
        print(''.join(str))


if __name__ == '__main__':
    stringUtil = StringUtil()
    stringUtil.test('�����й���   ')  # ������
    stringUtil.test('123456')  # ����
    stringUtil.test('!@#$%^&*()')  # Ӣ�������ַ�
    stringUtil.test('��@#��%����&*����')  # ���������ַ�
    stringUtil.test('�ĪY�T������͈��P')  # ��Ƨ��
    stringUtil.test('�ˤۤ󤴡��ˤäݤ�')  # ����
    stringUtil.test('???')  # ����  �ݲ�֧��
