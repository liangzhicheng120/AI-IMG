#!/bin/bash
# -*-coding=utf-8-*-
'''
@FileName: img_cnn_train.py
@Author：liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description：图片识别训练
@File URL: https://github.com/liangzhicheng120/contest
'''
import linecache
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import math
import csv
import pandas as pd
import random
import sys
from util.time import *


class ImgCnnTrain(object):
    def __init__(self):
        self._image_num = int(99000)  # 训练图片数量
        self._batch_size = 64  # 每个批次的图片数量
        self._image_height = int(60 / 2)  # 图片高度
        self._image_width = int(180 / 2)  # 图片宽度
        self._max_captcha = 7  # 图片含字符个数
        self._acc = 0.997  # 输出模型的精度
        self._acc_step = 0.001  # 达到输出模型精度后继续增加精度
        self._mapping_table, self._char_set_len = self.maping('0,1,2,3,4,5,6,7,8,9,+,-,*,(,), ')  # 字符映射表,字符总数
        self._converse_table = {v: k for k, v in (self._mapping_table).items()}
        self._save_model_path = sys.path[1] + '\\model\\{0}\\'  # 训练模型输出路径
        self._image_path = 'D:\\other\\image_contest_level_1\\{0}.png'  # 图片存放路径
        self._labels = self.read_file('D:\\other\\image_contest_level_1\\labels.txt')  # 标签文件
        self._image_matrix = 'D:\\other\\temp\\{0}.mat'  # 图片矩阵存放位置
        self._X = tf.placeholder(tf.float32, [None, self._image_height * self._image_width])
        self._Y = tf.placeholder(tf.float32, [None, self._max_captcha * self._char_set_len])
        self._keep_prob = tf.placeholder(tf.float32)

    def read_file(self, path):
        '''
        读取标签文件
        :param path:
        :return:
        '''
        return list(map(lambda x: x.split(' ')[0], linecache.getlines(path)))

    def str_2_matrix(self, str):
        '''
        字符转矩阵
        :param str:
        :return:
        '''
        matrix = np.zeros([self._max_captcha, self._char_set_len])
        for i in range(len(str)):
            if len(str) == 5:
                matrix[5][self._mapping_table[' '] - 1] = 1.0
                matrix[6][self._mapping_table[' '] - 1] = 1.0
            matrix[i][self._mapping_table[str[i]] - 1] = 1.0
        return matrix

    def maping(self, str):
        '''
        字符映射
        :param str:
        :return:
        '''
        _list = str.split(',')
        return dict(zip(_list, range(1, len(_list) + 1))), len(_list)

    def get_next_batch(self, startPoint=0, endPoint=100):
        '''
        获取训练批次
        :param startPoint:
        :param endPoint:
        :return:
        '''
        batch_x = np.zeros([abs(startPoint - endPoint), self._image_height * self._image_width])
        batch_y = np.zeros([abs(startPoint - endPoint), self._max_captcha * self._char_set_len])
        for i in range(startPoint, endPoint):
            batch_x[i - startPoint, :] = self.read_mat(i).flatten()
            batch_y[i - startPoint, :] = self.str_2_matrix(self._labels[i]).flatten()
        return batch_x, batch_y

    def read_mat(self, name):
        '''
        读取矩阵
        :param name:
        :return:
        '''
        try:
            matrix = np.loadtxt(open(self._image_matrix.format(name), 'rb'), delimiter=',', skiprows=0)
        except:
            raise Exception('请使用matrix.py生成对应的mat文件')
        return matrix

    # 定义CNN
    def crack_captcha_cnn(self, w_alpha=0.05, b_alpha=0.2):
        '''
        定义网络结构
        :param w_alpha:
        :param b_alpha:
        :return:
        '''
        # 将占位符 转换为 按照图片给的新样式
        x = tf.reshape(self._X, shape=[-1, self._image_height, self._image_width, 1])

        # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
        # w_c2_alpha = np.sqrt(2.0/(3*3*32))
        # w_c3_alpha = np.sqrt(2.0/(3*3*64))
        # w_d1_alpha = np.sqrt(2.0/(8*32*64))
        # out_alpha = np.sqrt(2.0/1024)

        # 3 conv layer
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))  # 从正太分布输出随机值
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self._keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self._keep_prob)
        #
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self._keep_prob)

        # TODO 可配置
        conv = conv3
        # Fully connected layer
        w_a, w_b, w_c = map(int, str(conv.get_shape()).replace(')', '').split(',')[1:])
        w_d = tf.Variable(w_alpha * tf.random_normal([w_a * w_b * w_c, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv, [-1, w_d.get_shape().as_list()[0]])

        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self._keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024, self._max_captcha * self._char_set_len]))
        b_out = tf.Variable(b_alpha * tf.random_normal([self._max_captcha * self._char_set_len]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out

    def random_batch(self, range=1000):
        '''
        随机选取1000(默认)张图片做测试
        :param range:
        :return:
        '''
        try:
            _num = random.randint(0, self._image_num - range)
        except:
            raise Exception('batch范围超出训练图片数量')
        return _num, _num + range

    def predict(self, sess, predict, accuracy, start, end):
        '''
        预测
        :param sess:
        :param predict:
        :param accuracy:
        :param start:
        :param end:
        :return:
        '''
        _count = 0.
        batch_x_test, batch_y_test = self.get_next_batch(start, end)
        predict = sess.run(predict, feed_dict={self._X: batch_x_test, self._Y: batch_y_test, self._keep_prob: 0.75})
        one_acc = sess.run(accuracy, feed_dict={self._X: batch_x_test, self._Y: batch_y_test, self._keep_prob: 1.})
        predict = np.array(predict)
        predict = np.reshape(predict, [-1, self._max_captcha, self._char_set_len])
        batch_y = np.reshape(batch_y_test, [-1, self._max_captcha, self._char_set_len])
        for i in range(len(predict)):
            _index_pre = np.argmax(predict[i], axis=1)
            _index_bat = np.argmax(batch_y[i], axis=1)
            predict[i] = 0
            for j in range(len(_index_pre)):
                predict[i][j][_index_pre[j]] = 1
            _count += 1 * np.all(_index_pre == _index_bat)
        all_acc = _count / len(predict)
        return predict, all_acc, one_acc

    def mkdir(self, acc):
        '''
        创建路径,增加精度
        :param acc:
        :return:
        '''
        path = self._save_model_path.format(acc)
        os.makedirs(path)
        self._acc += self._acc_step
        return path + '\\crack_capcha.model'

    def train_crack_captcha_cnn(self):
        '''
        训练
        :return:
        '''
        _num = 0
        _step = 0
        _output = self.crack_captcha_cnn()
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_output, labels=self._Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
        predict = tf.reshape(_output, [-1, self._max_captcha, self._char_set_len])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self._Y, [-1, self._max_captcha, self._char_set_len]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while True:
                batch_x, batch_y = self.get_next_batch(_num, _num + self._batch_size)
                _, loss_ = sess.run([optimizer, loss], feed_dict={self._X: batch_x, self._Y: batch_y, self._keep_prob: 0.75})
                if _step % 10 == 0:
                    _start, _end = self.random_batch()
                    _, all_acc, one_acc = self.predict(sess, predict, accuracy, _start, _end)
                    if all_acc >= self._acc:
                        saver.save(sess, save_path=self.mkdir(acc=self._acc), global_step=_step)
                    print('step: {0}, one_acc: {1}, all_acc: {2}'.format(_step, one_acc, all_acc))
                _step += 1
                _num = 0 if _num >= self._image_num else _num + self._batch_size

    @fn_timer
    def run(self):
        self.train_crack_captcha_cnn()


if __name__ == '__main__':
    imgCnnTrain = ImgCnnTrain()
    imgCnnTrain.run()
