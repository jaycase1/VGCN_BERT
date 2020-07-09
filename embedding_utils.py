from keras import backend as K
from keras.layers import Layer, Dense,ReLU,Dropout ,Embedding
import tensorflow as tf
from bert4keras.backend import recompute_grad

class VocabGraphConvolution(Layer):
    """Vocabulary GCN module.

    Params:
        `voc_dim`: The size of vocabulary graph
        `num_adj`: The number of the adjacency matrix of Vocabulary graph
        `hid_dim`: The hidden dimension after XAW
        `out_dim`: The output dimension after Relu(XAW)W
        `dropout_rate`: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.

    Inputs:
        `vocab_adj_list`: The list of the adjacency matrix
        `X_dv`: the feature of mini batch document, can be TF-IDF (batch, vocab), or word embedding (batch, word_embedding_dim, vocab)

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """
    def __init__(self,vocab_adj_list,voc_dim, hid_dim, out_dim, dropout_rate=0.2):
        super(VocabGraphConvolution, self).__init__()
        self.vocab_adj_list = vocab_adj_list
        self.voc_dim=voc_dim # 5847
        self.num_adj=len(vocab_adj_list) # 1
        self.hid_dim=hid_dim # 128
        self.out_dim=out_dim # 16
        self.fc_hc=Dense(units=out_dim)
        self.act_func = ReLU
        self.dropout = Dropout(dropout_rate)


    def build(self, input_shape):
        self.params = []
        for i in range(self.num_adj):
            self.params.append(Dense(name='W%d_vh'%i,units=self.hid_dim,trainable=True))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_dim,)

    @recompute_grad
    def call(self, inputs, add_linear_mapping_term=False):
        X_dv = inputs
        vocab_adj_list = self.vocab_adj_list
        for i in range(self.num_adj):
            H_vh = self.params[i](vocab_adj_list[i])
            H_vh=self.dropout(H_vh)
            H_dh = tf.matmul(X_dv,H_vh)
            if add_linear_mapping_term:
                H_linear = self.params[i](X_dv)
                H_linear=self.dropout(H_linear)
                H_dh+=H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out=self.fc_hc(fused_H)
        # 经过图卷积之后的输出Out size 为 16 * 768 * 16
        return out


class gcnBertEmbedding(Layer):
    def __init__(self,layers,dropout=0.2,output_dim=768):
        self.layers = layers
        assert len(layers) == 8
        self.input_token = layers[0]
        self.input_seg = layers[1]
        self.embedding_token = layers[2]
        self.embedding_seg = layers[3]
        self.add_token_seg = layers[4]
        self.embedding_pos = layers[5]
        self.norm = layers[6]
        self._dropout = layers[7]
        self.dropout = dropout
        self.output_dim = output_dim
        self.dense_ = Dense(768)
        super(gcnBertEmbedding,self).__init__()


    def compute_output_shape(self, input_shape):
        _input_shape = input_shape[0] + (self.output_dim,)
        return _input_shape

    @recompute_grad
    def call(self, inputs, **kwargs):
        input_ids = inputs[0]
        segment_ids = inputs[1]
        graph = inputs[2]

        token_ = self.input_token(input_ids)
        segment_ = self.input_seg(segment_ids)
        token_embedding = self.embedding_token(token_)
        segment_embedding = self.embedding_seg(segment_)
        add_token_seg = self.add_token_seg([token_embedding,segment_embedding])
        embedding_pos = self.embedding_pos(add_token_seg)
        embedding_pos_ = tf.matmul(graph,embedding_pos)
        embedding_pos_ = self.dense_(embedding_pos_)
        embedding_pos += embedding_pos_
        norm = self.norm(embedding_pos)
        output = self._dropout(norm)
        return output