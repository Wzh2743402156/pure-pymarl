from collections import namedtuple

def convert(dictionary):
    """
    将一个字典转换为命名元组（namedtuple）

    参数：
        dictionary (dict): 待转换的字典

    返回：
        namedtuple: 一个命名元组，其字段名称对应字典的键，字段值对应字典的值

    说明：
        1. namedtuple('GenericDict', dictionary.keys()) 根据字典的键生成一个命名元组类型，
           其中 'GenericDict' 为生成的类型名（可以任意命名）。
        2. (**dictionary) 将字典中的键值对解包为命名元组的初始化参数，从而构造出该命名元组的实例。
    """
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)
