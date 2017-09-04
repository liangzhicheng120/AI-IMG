# -*-coding:utf-8-*-
import string
import numpy as np


bit = 16


def cstr2vec(aString):
    '''
    输入单个中文字符串
    '''
    if type(aString) == type(u'\u0000'):
        strBinarySys = bin(ord(aString))
    else:
        strBinarySys = bin(ord(aString.decode("utf8")))
    strBinarySys = strBinarySys[2:len(strBinarySys)]
    strBinarySys = (len(strBinarySys) == 15) * "0" + strBinarySys
    vec = list(strBinarySys)
    vec = np.array(map(eval, vec))
    return vec


def centence2array(string):
    '''
    输入一句话
    '''
    if type(string) == type(u'\u0000'):
        string_UTF8 = string
    else:
        string_UTF8 = string.decode("utf8")

    array = np.zeros([len(string_UTF8), bit])
    for i, word in enumerate(string_UTF8):
        array[i, :] = cstr2vec(word)

    return array


def array2centence(array):
    string_UTF8 = u""
    for row in array:
        vec = map(int, list(row))
        string_UTF8 = string_UTF8 + str(hex(eval('0b' + "".join(map(str, vec))))).replace('0x', '\\u')
    # print(eval('u"'+string_UTF8+'"'))
    return eval('u"' + string_UTF8 + '"')


if __name__ == "__main__":
    centence = "哈哈呵呵".encode('utf-8')
    print(centence, centence)
    # array = centence2array(centence)
    # print(array)
    # centence1 = array2centence(array)
    # print(centence1)
