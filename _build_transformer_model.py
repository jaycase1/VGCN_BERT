from bert4keras.layers import *
from bert4keras.snippets import delete_arguments
from keras.models import Model
from bert4keras.models import BERT,ALBERT,ALBERT_Unshared,NEZHA,ELECTRA,GPT2_ML,T5,extend_with_language_model,extend_with_unified_language_model
import json
'''
overload bert4keras.models.build_transformer_model
only changed in line 58 in this .py

'''


def build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    return_keras_model=True,
    **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings')
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')

    model, application = model.lower(), application.lower()

    models = {
        'bert': BERT,
        'albert': ALBERT,
        'albert_unshared': ALBERT_Unshared,
        'nezha': NEZHA,
        'electra': ELECTRA,
        'gpt2_ml': GPT2_ML,
        't5': T5,
    }
    MODEL = models[model]

    if model != 't5':
        if application == 'lm':
            MODEL = extend_with_language_model(MODEL)
        elif application == 'unilm':
            MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return transformer.model,transformer
    else:
        return transformer