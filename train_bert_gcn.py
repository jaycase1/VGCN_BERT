from _build_transformer_model import build_transformer_model
from data_utils import load_pkl,normalize_adj
import numpy as np
from math import ceil
from keras.utils import to_categorical
from scipy import sparse as sp
from gcn_bert_model import GCNBERTModel
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

epochs = 10
batch_size = 10
config_path = "chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_model.ckpt"
dict_path = "chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/vocab.txt"

train_graph = load_pkl("data/dump_data/waimai_train.graph")
test_graph = load_pkl("data/dump_data/waimai_test.graph")
valid_graph = load_pkl("data/dump_data/waimai_valid.graph")

train_ids = load_pkl("data/dump_data/waimai_train.text")
valid_ids = load_pkl("data/dump_data/waimai_valid.text")
test_ids = load_pkl("data/dump_data/waimai_test.text")

train_labels = load_pkl("data/dump_data/waimai_train.label")
valid_labels = load_pkl("data/dump_data/waimai_valid.label")
test_labels = load_pkl("data/dump_data/waimai_test.label")


def graph_padding(graph,maxLen):
    graph = graph.tocoo()
    rows = graph.row
    cols = graph.col
    data = graph.data
    return sp.coo_matrix((data,(rows,cols)),shape=(maxLen,maxLen)).todense()



def graphs_padding(graphs,maxLen):
    '''
    :param graphs: the list of sparse matrix
    :param maxLen: should padding to length
    :return:
    '''
    graphs_ = []
    for graph in graphs:
        graphs_.append(normalize_adj(graph_padding(graph,maxLen)))
    return np.array(graphs_)







def gen_ids(ids,labels,graphs,batch_size):
    assert len(ids) == len(labels)
    assert len(ids) == len(graphs)
    len_ = len(ids)
    batch_ = ceil(len_/float(batch_size))
    while(1):
        for i in range(batch_):
            if(i<batch_-1):
                ids_ = ids[i*batch_size:(i+1)*batch_size]
                labels_ = labels[i*batch_size:(i+1)*batch_size]
                graphs_ = graphs[i*batch_size:(i+1)*batch_size]
            else:
                ids_ = ids[i*batch_size:]
                labels_ = labels[i*batch_size:]
                graphs_ = graphs[i*batch_size:]
            maxLen = max([len(id) for id in ids_])
            ids_ = np.array([id + [0,] * (maxLen - len(id)) for id in ids_])
            seg_ = np.zeros(shape=(len(ids_),maxLen))
            graphs_ = graphs_padding(graphs_,maxLen)
            labels_ = to_categorical(labels_,2)
            yield ({"input_1":ids_,"input_2":seg_,"input_3":graphs_},{"output":labels_})


train_gen = gen_ids(train_ids,train_labels,train_graph,batch_size)
valid_gen = gen_ids(valid_ids,valid_labels,valid_graph,batch_size)
test_gen = gen_ids(test_ids,test_labels,test_graph,batch_size)

model,transformer = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path)


gcn_bert = GCNBERTModel(model=model,transform=transformer)
ids = Input(shape=(None,),name="input_1")
segs = Input(shape=(None,),name="input_2")
graph = Input(shape=(None,None),name="input_3")

out = gcn_bert([ids,segs,graph])
y = LSTM(256)(out)
y = Dense(2,activation="softmax",name="output")(y)
model = Model([ids,segs,graph],y)

model.compile(loss="binary_crossentropy",
              optimizer=Adam(1e-5),
              metrics=["binary_accuracy"])


model.fit_generator(train_gen,
                    steps_per_epoch=510,
                    validation_data=valid_gen,
                    validation_steps=20,
                    epochs=epochs)



