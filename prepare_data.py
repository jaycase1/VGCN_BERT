import os
from data_gen import DataReader,OurTokenizer,get_token_dict
from data_utils import get_dataset,showed_word_dict,create_pmi_matrix,texts_2_graphs,write_to_pkl
import numpy as np
import pickle as pkl

dump_dir='data/dump_data'
if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)

train, valid ,test = DataReader("data/waimai_clean.csv").read()
trainset, validset, testset = get_dataset(train,valid,test)
print('train_szie:%d, valid_size:%d, test_size:%d'%(len(trainset['label']),len(validset['label']),len(testset['label'])))
corpus = trainset['text'] + validset['text'] + testset['text']
y = np.array(trainset['label'] + validset['label'] + testset['label'])
label2idx = {'0':0,'1':1}
idx2label = {0:'0',1:'1'}
corpus_size = len(corpus)
y_prob = np.eye(corpus_size, len(label2idx))[y]
tokenizer = OurTokenizer(get_token_dict())
dict_ = tokenizer._token_dict
id2token = {v:k for k,v in dict_.items()}


wordDict, pmi_matrix = create_pmi_matrix(tokenizer,corpus)
train_graphs = texts_2_graphs(trainset['text'],tokenizer,pmi_matrix,wordDict)
valid_graphs = texts_2_graphs(validset['text'],tokenizer,pmi_matrix,wordDict)
test_graphs =  texts_2_graphs(testset['text'],tokenizer,pmi_matrix,wordDict)

write_to_pkl(trainset,train_graphs,"/waimai_train",tokenizer,dict_)
write_to_pkl(validset,valid_graphs,"/waimai_valid",tokenizer,dict_)
write_to_pkl(testset,test_graphs,"/waimai_test",tokenizer,dict_)

with open(dump_dir+"/waimai_pmi.graph","wb") as f:
    pkl.dump(pmi_matrix,f)

with open(dump_dir+"/waimai_word.dict","wb") as f:
    pkl.dump(wordDict,f)

with open(dump_dir+"/waimai_alldict","wb") as f:
    pkl.dump(dict_,f)


