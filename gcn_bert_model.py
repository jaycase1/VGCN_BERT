from embedding_utils import gcnBertEmbedding
from keras.layers import Layer
from bert4keras.backend import recompute_grad
from keras.backend import concatenate

class GCNBERTModel(Layer):
    def __init__(self,model,transform,num_hidden_layers=12):
        '''
        :param model: Bert_cased 模型
        '''
        super(GCNBERTModel,self).__init__()
        print("model, ",type(model))
        self._model = model
        self._transform = transform
        embedding_layers = model.layers[0:8]
        self.num_hidden_layers = num_hidden_layers
        self.embedding_model = gcnBertEmbedding(embedding_layers)
        self.num_hidden_layers = num_hidden_layers

    @recompute_grad
    def call(self, inputs):
        output = self.embedding_model(inputs)
        for i in range(self.num_hidden_layers):
            output = self._transform.apply_main_layers(output,i)
        output = self._transform.apply_final_layers(output)
        return output

    def compute_output_shape(self, input_shape):
        return self.embedding_model.output_shape