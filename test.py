#!/bin/bash
# -*-coding:gbk -*-
import re
import os
import sys
import linecache
import codecs
import numpy as np


def bin_2_str(str):
    '''
    ������ת�ַ���
    :param str: �������������ַ���
    :return:
    '''
    return chr(bin_2_dec(str))


def bin_2_dec(str):
    '''
    ������תʮ����
    :param str:�������������ַ���
    :return:
    '''
    return int(str, 2);


def str_2_bin(str):
    '''
    �ַ���ת������
    :param str: �����ַ���
    :return: �����Ʊ���
    '''
    return '{0:b}'.format((ord(str)))


def read_file(fileName):
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


def add_zero(str, size=3):
    '''
    ����
    Ӣ��1�ֽ�,����3�ֽ�,��Ƨ��4-6�ֽ�,Ĭ��3�ֽ�
    :param str: �ַ���
    :param num: ����
    :return:
    '''
    return str.zfill(size * 8)


def str_2_matrix(str):
    '''
    �ַ���ת�����ƾ���
    :param str: �ַ���
    :return:
    '''
    result = map(str_2_bin, list(str))
    result = map(int, result)
    result = list(result)
    result = np.array(result)
    return result


def matrix_2_str(matrix):
    '''
    ����ת�ַ���
    :param matrix:
    :return:
    '''
    result = matrix.tolist()
    result = map(lambda x: bin_2_str(str(x)), result)
    result = list(result)
    return result


def test(str):
    print('======================')
    matrix = str_2_matrix(str)
    print(matrix)
    str = matrix_2_str(matrix)
    print(str)


if __name__ == '__main__':
    test('�����й���')  # ������
    test('123456')  # ����
    test('!@#$%^&*()')  # Ӣ�������ַ�
    test('��@#��%����&*����')  # ���������ַ�
    test('�ĪY�T������͈��P')  # ��Ƨ��
    test('�ˤۤ󤴡��ˤäݤ�')  # ����
    test('???')  # ����  �ݲ�֧��
    pass
