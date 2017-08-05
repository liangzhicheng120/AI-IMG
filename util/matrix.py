#!/bin/bash
# -*-coding=utf-8-*-
import re
import os
from PIL import Image
import numpy as np

'''
@FileName: matrix.py
@Author：liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description：图片转矩阵,输出mat文件用于存放矩阵信息
@File URL: https://github.com/liangzhicheng120/contest
'''


class Matrix(object):
    def __init__(self):
        self._num = 200000  # 图片数量
        self._source = 'D:/other/image_contest_level_1_validate/image_contest_level_1_validate/{0}.png'  # 图片存放位置
        self._result = 'D:/other/validate_temp/{0}.mat'  # 矩阵输出位置
        self._image_height = int(60 / 2)  # 图片高度
        self._image_width = int(180 / 2)  # 图片宽度
        pass

    def run(self):
        for i in range(0, self._num):
            image_handle = Image.open(self._source.format(i))
            image_handle = image_handle.convert('L')
            image_matrix = self.binaryzation(image_handle)
            np.savetxt(self._result.format(i), image_matrix, delimiter=',')
        pass

    def binaryzation(self, image_gray):
        '''
        图片二值化
        :param image_gray:图片
        :return: 矩阵
        '''
        image_gray = image_gray.resize((self._image_width, self._image_height), Image.ANTIALIAS)
        image_array = np.array(image_gray)
        image_array = (image_array / 255.0) * (image_array / 255.0)
        image_array = 1 - image_array
        image_shape = [self._image_height, self._image_width]
        image_array = 1 + np.floor(image_array - image_array.mean())
        return image_array


if __name__ == '__main__':
    matrix = Matrix()
    matrix.run()
