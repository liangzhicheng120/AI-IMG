#!/bin/bash
# -*-coding=utf-8-*-
import tensorflow as tf
import numpy as np
from util.matrix import *
from PIL import Image
from util.matrix import *
from util.time import *
from util.filter import *
import linecache

'''
@FileName: img_cnn_train.py
@Author：liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description：单张图片识别
@File URL: https://github.com/liangzhicheng120/contest
'''


class ImgCnnPredict(object):
    def __init__(self):
        self._image_num = int(99000)  # 训练图片数量
        self._batch_size = 32  # 每个批次的图片数量
        self._image_height = int(60 / 2)  # 图片高度
        self._image_width = int(180 / 2)  # 图片宽度
        self._max_captcha = 7  # 图片含字符个数
        self._mapping_table, self._char_set_len = self.maping('0,1,2,3,4,5,6,7,8,9,+,-,*,(,), ')  # 字符映射表,字符总数
        self._converse_table = {v: k for k, v in (self._mapping_table).items()}
        self._labels = self.read_file('D:\\other\\image_contest_level_1\\labels.txt')  # 标签文件
        # self._image_matrix = 'D:\\other\\validate_temp\\{0}.mat'  # 图片矩阵存放位置
        self._image_matrix = 'D:\\other\\temp\\{0}.mat'  # 图片矩阵存放位置
        self._train_model = 'D:\\other\\model\\{0}\\crack_capcha_success.model-{1}'.format(995, 49390)
        self._predict_file = sys.path[1] + '\\p-labels.txt'
        self._X = tf.placeholder(tf.float32, [None, self._image_height * self._image_width])
        self._Y = tf.placeholder(tf.float32, [None, self._max_captcha * self._char_set_len])
        self._keep_prob = tf.placeholder(tf.float32)

    def read_file(self, path):
        '''
        读标签文件
        :param path:
        :return:
        '''
        return list(map(lambda x: x.split(' ')[0], linecache.getlines(path)))

    def str_2_matrix(self, str):
        '''
        字符串转矩阵
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
        字符串映射成矩阵
        :param str:
        :return:
        '''
        _list = str.split(',')
        return dict(zip(_list, range(1, len(_list) + 1))), len(_list)

    def get_next_batch(self, img):
        batch_x = np.zeros([1, self._image_height * self._image_width])
        batch_y = np.zeros([1, self._max_captcha * self._char_set_len])
        batch_x[0, :] = self.get_matrix(img).flatten()
        batch_y[0, :] = self.str_2_matrix('1-2-3').flatten()
        return batch_x, batch_y

    def get_matrix(self, img):
        '''
        读取mat文件
        :param name:
        :return:
        '''
        matrix = Matrix()
        image_handle = Image.open(img).convert('L')
        image_matrix = matrix.binaryzation(image_handle)
        return image_matrix

        # 定义CNN

    def crack_captcha_cnn(self, w_alpha=0.05, b_alpha=0.2):
        '''
        构建网络模型
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

    def matrix_2_str(self, matrix):
        '''
        矩阵转字符串
        :param array:
        :return:
        '''
        _str = []
        for i in range(len(matrix)):
            _char = ''
            _max_index = np.argmax(matrix[i], axis=1)
            for j in range(len(_max_index)):
                _char += self._converse_table[_max_index[j] + 1]
            _str.append(_char)
        return ''.join(_str)

    def predict(self, sess, predict, img):
        '''
        预测
        :param sess:
        :param predict:
        :param accuracy:
        :param start:
        :param end:
        :return:
        '''
        batch_x_test, batch_y_test = self.get_next_batch(img)
        predict = sess.run(predict, feed_dict={self._X: batch_x_test, self._Y: batch_y_test, self._keep_prob: 0.75})
        predict = np.array(predict)
        predict = np.reshape(predict, [-1, self._max_captcha, self._char_set_len])
        batch_y = np.reshape(batch_y_test, [-1, self._max_captcha, self._char_set_len])
        for i in range(len(predict)):
            _index_pre = np.argmax(predict[i], axis=1)
            _index_bat = np.argmax(batch_y[i], axis=1)
            predict[i] = 0
            for j in range(len(_index_pre)):
                predict[i][j][_index_pre[j]] = 1
        predict = self.matrix_2_str(predict)
        return predict

    def vote(self, sess, predict, img, votes):
        max_vote = ''
        if votes == 0:
            max_vote = self.predict(sess, predict, img)
        else:
            vote_map = {}
            for i in range(0, votes):
                predict = self.predict(sess, predict, img)
                vote_map[predict] = vote_map.get(predict, 0) + 1
            max_vote = sorted(vote_map, key=lambda x: vote_map[x])[-1].strip()
        return max_vote

    def train_crack_captcha_cnn(self, img, votes):
        '''
        训练
        :param 0:
        :param 1:
        :param votes:
        :return:
        '''
        _output = self.crack_captcha_cnn()
        predict = tf.reshape(_output, [-1, self._max_captcha, self._char_set_len])
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self._train_model)
            s = self.vote(sess, predict, img, votes)
            print(s)

    @fn_timer
    def run(self, img, votes):
        self.train_crack_captcha_cnn(img, votes)


if __name__ == '__main__':
    predict = ImgCnnPredict()
    predict.run(img='{0}\\img\\0.png'.format(sys.path[1]), votes=0)
