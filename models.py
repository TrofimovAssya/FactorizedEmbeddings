import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable

class FactorizedMLP(nn.Module):

    def __init__(self, layers_size, inputs_size, emb_size=2):
        super(FactorizedMLP, self).__init__()

        self.layers_size = layers_size
        self.emb_size = emb_size
        self.inputs_size = inputs_size


        # The embedding
        assert len(inputs_size) == 2

        self.emb_1 = nn.Embedding(inputs_size[0], emb_size)
        self.emb_2 = nn.Embedding(inputs_size[1], emb_size)

        # The list of layers.
        layers = []
        dim = [emb_size * 2] + layers_size # Adding the emb size.
        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], 1)

    def get_embeddings(self, x):

        gene, patient = x[:, 0], x[:, 1]
        # Embedding.
        gene = self.emb_1(gene.long())
        patient = self.emb_2(patient.long())

        return gene, patient

    def forward(self, x):

        # Get the embeddings
        emb_1, emb_2 = self.get_embeddings(x)

        # Forward pass.
        mlp_input = torch.cat([emb_1, emb_2], 1)

        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)

        mlp_output = self.last_layer(mlp_input)

        return mlp_output

    def generate_datapoint(self, e, gpu):
        #getting a datapoint embedding coordinate
        emb_1 = self.emb_1.weight.cpu().data.numpy()
        emb_2 = (np.ones(emb_1.shape[0]*2).reshape((emb_1.shape[0],2)))*e
        emb_1 = torch.FloatTensor(emb_1)
        emb_2 = torch.FloatTensor(emb_2)
        emb_1 = Variable(emb_1, requires_grad=False).float()
        emb_2 = Variable(emb_2, requires_grad=False).float()
        #if gpu:
        emb_1 = emb_1.cuda(gpu)
        emb_2 = emb_2.cuda(gpu)
        mlp_input = torch.cat([emb_1, emb_2],1)
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)
        return mlp_output

class TripleFactorizedMLP(nn.Module):

    def __init__(self, layers_size, inputs_size, emb_size=2):
        super(TripleFactorizedMLP, self).__init__()

        self.layers_size = layers_size
        self.emb_size = emb_size
        self.inputs_size = inputs_size


        # The embedding
        assert len(inputs_size) == 3

        self.emb_1 = nn.Embedding(inputs_size[0], emb_size)
        self.emb_2 = nn.Embedding(inputs_size[1], emb_size)
        self.emb_3 = nn.Embedding(inputs_size[2], emb_size)

        # The list of layers.
        layers = []
        dim = [emb_size * 3] + layers_size # Adding the emb size.
        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], 1)

    def get_embeddings(self, x):

        gene, patient, domain = x[:, 0], x[:, 1], x[:, 2]
        # Embedding.
        gene = self.emb_1(gene.long())
        patient = self.emb_2(patient.long())
        domain = self.emb_3(domain.long())

        return gene, patient, domain

    def forward(self, x):

        # Get the embeddings
        emb_1, emb_2, emb_3 = self.get_embeddings(x)

        # Forward pass.
        mlp_input = torch.cat([emb_1, emb_2, emb_3], 1)

        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)

        mlp_output = self.last_layer(mlp_input)

        return mlp_output

    def generate_datapoint(self, e, d, gpu):
        #getting a datapoint embedding coordinate
        emb_1 = self.emb_1.weight.cpu().data.numpy()
        emb_2 = (np.ones(emb_1.shape[0]*2).reshape((emb_1.shape[0],2)))*e
        emb_3 = (np.ones(emb_1.shape[0]*2).reshape((emb_1.shape[0],2)))*d

        emb_1 = torch.FloatTensor(emb_1)
        emb_2 = torch.FloatTensor(emb_2)
        emb_3 = torch.FloatTensor(emb_2)

        emb_1 = Variable(emb_1, requires_grad=False).float()
        emb_2 = Variable(emb_2, requires_grad=False).float()
        emb_3 = Variable(emb_3, requires_grad=False).float()
        #if gpu:
        emb_1 = emb_1.cuda(gpu)
        emb_2 = emb_2.cuda(gpu)
        emb_3 = emb_3.cuda(gpu)

        mlp_input = torch.cat([emb_1, emb_2, emb_3],1)
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)
        return mlp_output

def get_model(opt, inputs_size, model_state=None):

    if opt.model == 'factor':
        model_class = FactorizedMLP
        model = model_class(layers_size=opt.layers_size,emb_size=opt.emb_size,inputs_size=inputs_size)

    elif opt.model == 'triple':
        model_class = TripleFactorizedMLP
        model = model_class(layers_size=opt.layers_size,emb_size=opt.emb_size,inputs_size=inputs_size)
    else:
        raise NotImplementedError()

    if model_state is not None:
        model.load_state_dict(model_state)

    return model
