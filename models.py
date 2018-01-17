import torch
import torch.nn.functional as F
from torch import nn


# TODO: keras model. To transform into pytorch.
# print ('constructing network...')
# ep = Sequential()
# eg = Sequential()
#
# ep.add(Embedding(nb_patient, p_emb_size, input_length=1, embeddings_regularizer=l2(1e-5)))
# eg.add(Embedding(nb_genes, g_emb_size, input_length=1, embeddings_regularizer=l2(1e-5)))
#
# model = Sequential()
# model.add(Merge([ep, eg], mode='concat'))
# model.add(Flatten())
# model.add(Dense(150, activation='tanh', kernel_regularizer=l2(1e-5)))
# model.add(Dense(100, activation='tanh', kernel_regularizer=l2(1e-5)))
# model.add(Dense(75, activation='tanh', kernel_regularizer=l2(1e-5)))
# model.add(Dense(50, activation='tanh', kernel_regularizer=l2(1e-5)))
# model.add(Dense(25, activation='tanh', kernel_regularizer=l2(1e-5)))
# model.add(Dense(10, activation='tanh', kernel_regularizer=l2(1e-5)))
# model.add(Dense(1))


class FactorizedMLP(nn.Module):

    def __init__(self):
        super(FactorizedMLP, self).__init__()

    def forward(self, x):

        # TODO: do.

        return x


def get_model(opt):

    # All of the different models.

    if opt.model == 'factor':
        model = FactorizedMLP()
    else:
        raise NotImplementedError()

    return model