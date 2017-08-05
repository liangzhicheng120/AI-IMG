#!/bin/bash
# -*-coding=utf-8-*-
import os
import sys
import re
import time
from functools import wraps

'''
@FileName: util.py
@Author：liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description：工具类
@File URL: https://github.com/liangzhicheng120/contest
'''


def fn_timer(function):
    '''
    计算程序运行时间
    :param function:
    :return:
    '''

    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.clock()
        result = function(*args, **kwargs)
        t1 = time.clock()
        print("Total time running : %s s" % (str(t1 - t0)))
        return result

    return function_timer
