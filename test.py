# /bin/bash
# -*- coding:gbk -*-

class ����(object):
    def ��(self):
        print("%s �� " % self.����)

    def ��(self):
        print("%s �� " % self.����)

    def ��(self):
        print("%s �� " % self.����)

    def ��(self):
        print("%s �� " % self.����)


class è(����):
    def __init__(self, ����):
        self.���� = ����
        self.���� = 'è'

    def ��(self):
        print('������')

# �������������д������һ����������ʾ��ǰ��̳�����һ����
class ��(����):
    def __init__(self, ����):
        self.���� = ����
        self.���� = '��'

    def ��(self):
        print('������')

# �������������д������һ����������ʾ��ǰ��̳�����һ����
if __name__ == '__main__':
    С��è = è('С�׼ҵ�С��è')
    С��è.��()

    С��è = è('С�ڵ�С��è')
    С��è.��()

    С�׹� = ��('���Ӽҵ�С�׹�')
    С�׹�.��()
