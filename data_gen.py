from bert4keras.tokenizers import Tokenizer
import codecs
import sys
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from math import ceil

dict_path = "chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/vocab.txt"

def get_token_dict():
    """
    # 将词表中的字编号转换为字典
    :return: 返回自编码字典
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示   UNK是unknown的意思
        return R


def read_file_from_csv(filename):
    """
    :param filename: the format must be csv
    :return:
    """
    format = filename.split('.')[-1]
    if format== "csv":
        data = pd.read_csv(filename, encoding='utf-8')
    elif format == "tsv":
        data = pd.read_table(filename,encoding='utf-8')
    else:
        Warning("More format haven't develop, also you can overload it")
        sys.exit()

    return data


class DataReader():
    """
    Get dataset from files

    Examples:
        train, dev, test = DataReader("data/train.csv","data/dev.csv","data/test.csv").read()
    """

    def __init__(self, train_file, dev_file=None, test_file=None):
        """
        Init dataset information.

        Inputs:
            train_file: train file's location & full name
            dev_file: dev file's location & full name
            test_file: test file's location & full name

        Examples:
            DataReader("data/train.txt","data/dev.txt","data/test.txt")
        """
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file


    def formate(self, _file):
        """
        Formate raw data

        Inputs:
            raw_data: a set with raw data

        Returns:
            dataset: a set with formated data

        Examples:
            raw = ["1 Abc def\\n", "0 xyz"]
            dataset = formate(raw)
            assert(dataset == [(1, "abc def"]), (0, "xyz")])
        """
        if _file == None:
            return None
        data = read_file_from_csv(_file)
        return data


    def read(self):
        """
        Get dataset and formate.

        Returns:
            train: train dataset
            dev: dev dataset
            test: test dataset

        Examples:
            train, dev, test = read()
        """
        train = self.formate(self.train_file)
        dev = self.formate(self.dev_file)
        test = self.formate(self.test_file)
        return train, dev, test


def gen_(dataset,tokenizer,batchsize):
    '''
    :param dataset: dict
    :param tokenizer: Bert's default tokenizer
    :return:
    '''
    labels = dataset['label']
    contexts = dataset['text']
    assert len(labels) == len(contexts)
    bacthS = ceil(len(labels)/batchsize)
    while(1):
        for i in range(bacthS):
            tokens = []
            segms = []
            if(i<bacthS-1):
                texts = contexts[i*batchsize:(i+1)*batchsize]
                label_ = labels[i*batchsize:(i+1)*batchsize]
            else:
                texts = contexts[i*batchsize:]
                label_ = labels[i*batchsize:]
            for text in texts:
                token_ids, segment_ids = tokenizer.encode(text)
                tokens.append(token_ids)
                segms.append(segment_ids)
            len_max = max([len(token) for token in tokens])
            tokens = np.array([token + [0,] * (len_max-len(token)) for token in tokens])
            segms = np.array([segm + [0,] * (len_max - len(segm)) for segm in segms])
            label_ = to_categorical(np.array(label_),2)
            yield ({'input_1':np.array(tokens),'input_2':np.array(segms)},{'output':label_})