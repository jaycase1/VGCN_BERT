import pickle as pkl
from data_utils import load_pkl
from bert4keras.models import build_transformer_model
from math import ceil
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input,LSTM,Dense
from keras.models import Model
from keras.optimizers import Adam
from onlstm import ONLSTM


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


def gen_ids(ids,labels,batch_size):
    assert len(ids) == len(labels)
    len_ = len(ids)
    batch_ = ceil(len_/float(batch_size))
    while(1):
        for i in range(batch_):
            if(i<batch_-1):
                ids_ = ids[i*batch_size:(i+1)*batch_size]
                labels_ = labels[i*batch_size:(i+1)*batch_size]
            else:
                ids_ = ids[i*batch_size:]
                labels_ = labels[i*batch_size:]
            maxLen = max([len(id) for id in ids_])







            ids_ = np.array([id + [0,] * (maxLen - len(id)) for id in ids_])
            seg_ = np.zeros(shape=(len(ids_),maxLen))
            labels_ = to_categorical(labels_,2)
            yield ({"input_1":ids_,"input_2":seg_},{"output":labels_})



train_gen = gen_ids(train_ids,train_labels,batch_size)
valid_gen = gen_ids(valid_ids,valid_labels,batch_size)
test_gen = gen_ids(test_ids,test_labels,batch_size)

model_bert = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path)

for l in model_bert.layers:
        l.trainable = True

tokens = Input(shape=(None,),name='input_1')
segments = Input(shape=(None,),name='input_2')
output = model_bert([tokens,segments])
output = ONLSTM(768,16,return_sequences=True)(output)
output = ONLSTM(768,8)(output)
y = Dense(32)(output)
y = Dense(2,activation="softmax",name='output')(y)
model = Model([tokens,segments],y)

model.compile(loss="binary_crossentropy",
              optimizer=Adam(1e-5),
              metrics=["binary_accuracy"])


model.fit_generator(train_gen,
                    steps_per_epoch=510,
                    validation_data=valid_gen,
                    validation_steps=20,
                    epochs=epochs)