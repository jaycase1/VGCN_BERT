from sklearn.utils import shuffle
import numpy as np
from math import log
from scipy import sparse as sp
import pickle as pkl
dump_dir='data/dump_data'


def get_dataset(train,valid,test,valid_split=0.1,test_split=0.05):
    """
    :param train: DataFrame
    :param valid: DataFrame or None
    :param test:  DataFrame or None
    :return: trainset, validset ,testset
    """
    if(valid==None):
        train_size = len(train)
        train_ = train[:int((1-valid_split-test_split)*train_size)]
        valid_ = train[len(train_):int(train_size*(1-test_split))]
        test_ = train[len(valid_)+len(train_):]
    else:
        train_ = train
        valid_ = valid
        test_ =  test

    trainset , validset, testset = {},{},{}
    headers = train.columns.values

    for data, dataset in [(train_,trainset),(valid_,validset),(test_,testset)]:
        label = data[headers[0]]
        text =  data[headers[1]]
        dataset['label'] = label.tolist()
        dataset['text'] = text.tolist()
    return trainset, validset, testset

def showed_word_dict(tokenizer,corpus):
    dict_ = tokenizer._token_dict
    dictShow = {}
    wordPair = {}
    for text in corpus:
        words = tokenizer._tokenize(text)
        for index, word in enumerate(words):
            if(word not in dictShow.keys()):
                dictShow[word] = 1
            else:
                dictShow[word] += 1
            if(index<len(words)-1):
                for other in words[index+1:]:
                    if(word==other):
                        continue

                    pair = "{0},{1}".format(dict_[word],dict_[other])
                    if(pair not in wordPair.keys()):
                        wordPair[pair] = 1
                    else:
                        wordPair[pair] += 1
    return dictShow,wordPair



def create_pmi_matrix(tokenizer,corpus):
    dict_ = tokenizer._token_dict
    id2token = {v:k for k,v in dict_.items()}
    dictShow,wordPair = showed_word_dict(tokenizer,corpus)
    #print(64,dictShow['[PAD]'])
    wordDict = {}
    for word in dictShow.keys():
        wordDict[word] = len(wordDict)
    wordLen = len(dictShow)
    num_window = len(corpus)
    pmi_matrix = np.eye(wordLen)

    for key in wordPair.keys():
        count = wordPair[key]
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        freq_i = dictShow[id2token[i]]
        freq_j = dictShow[id2token[j]]

        revers_pair = "{0},{1}".format(j,i)
        if revers_pair in wordPair.keys():
            count += wordPair[revers_pair]
        npmi =  log(1.0 * freq_i * freq_j/(num_window * num_window))/log(1.0 * count / num_window) -1
        if npmi > 0:
            pmi_matrix[int(wordDict[id2token[i]])][int(wordDict[id2token[j]])] = npmi
            pmi_matrix[int(wordDict[id2token[j]])][int(wordDict[id2token[i]])] = npmi

    return wordDict,pmi_matrix


def text_2_graph(text,tokenizer,pmiMatrix,wordDict):
    words = tokenizer._tokenize(text)
    graph = np.eye(len(words))
    for i ,word in enumerate(words):
        for j in range(i+1,len(words)):
            graph[i][j] = pmiMatrix[wordDict[words[i]]][wordDict[words[j]]]
            graph[j][i] = pmiMatrix[wordDict[words[j]]][wordDict[words[j]]]
    return sp.csr_matrix(graph)

def texts_2_graphs(texts,tokenizer,pmiMatrix,wordDict):
    graphs = []
    for text in texts:
        graph = text_2_graph(text,tokenizer,pmiMatrix,wordDict)
        graphs.append(graph)
    return graphs

def write_to_pkl(dataset,graphs,name,tokenizer,dict_):
    texts = dataset['text']
    texts = [tokenizer._tokenize(text) for text in texts]
    texts = [[dict_[word] for word in words] for words in texts]
    labels = dataset['label']
    with open(dump_dir+name+".text","wb") as f:
        pkl.dump(texts,f)
    with open(dump_dir + name + ".label","wb") as f:
        pkl.dump(labels,f)
    with open(dump_dir + name + ".graph","wb") as f:
        pkl.dump(graphs,f)

def load_pkl(name):
    with open(name,'rb') as f:
        return pkl.load(f,encoding='utf-8')


def normalize_adj(graph):
    """Symmetrically normalize adjacency matrix.
        the graph have already add self-loop
        the output is the graph's Laplacian Operator
    """
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(graph.sum(1)) #D-degree matrix
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt).todense()
    return graph.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)




