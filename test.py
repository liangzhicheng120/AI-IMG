# /bin/bash
# -*- coding:gbk -*-

class 动物(object):
    def 吃(self):
        print("%s 吃 " % self.名字)

    def 喝(self):
        print("%s 喝 " % self.名字)

    def 拉(self):
        print("%s 拉 " % self.名字)

    def 撒(self):
        print("%s 撒 " % self.名字)


class 猫(动物):
    def __init__(self, 名字):
        self.名字 = 名字
        self.种类 = '猫'

    def 叫(self):
        print('喵喵叫')

# 在类后面括号中写入另外一个类名，表示当前类继承另外一个类
class 狗(动物):
    def __init__(self, 名字):
        self.名字 = 名字
        self.种类 = '狗'

    def 叫(self):
        print('汪汪叫')

# 在类后面括号中写入另外一个类名，表示当前类继承另外一个类
if __name__ == '__main__':
    小黑猫 = 猫('小白家的小黑猫')
    小黑猫.吃()

    小白猫 = 猫('小黑的小白猫')
    小白猫.喝()

    小白狗 = 狗('胖子家的小白狗')
    小白狗.吃()
