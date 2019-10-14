import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
from sklearn.decomposition import PCA

class FactorizedMLP(nn.Module):

    def __init__(self, layers_size, inputs_size, emb_size=2, data_dir = 'data/', set_gene_emb = '.', warm_pca = '.'):
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

        ### Warm start for gene embeddings
        if not set_gene_emb == '.':
            new_embs = np.load(set_gene_emb)
            self.emb_1.weight.data = torch.FloatTensor(new_embs)

        ### PCA start for sample embeddings
        if not warm_pca == '.':
            self.start_with_PCA(datadir = data_dir, datafile = warm_pca)

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

    def start_with_PCA(self, datadir = 'data/',datafile = '.'):
        data = np.load(''.join([datadir, datafile]))
        data = np.log10(data+1)
        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(data)
        self.emb_2.weight.data = torch.FloatTensor(X_pca)


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

    def freeze_all(self):

        for layer in self.mlp_layers:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        self.emb_1.weight.requires_grad = False


    def unfreeze_all(self):

        for layer in self.mlp_layers:
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True
        self.emb_1.weight.requires_grad = True

class ChoyEmbedding(nn.Module):

    def __init__(self, emb_size, inputs_size):
        super(ChoyEmbedding, self).__init__()

        self.emb_size = emb_size
        self.inputs_size = inputs_size


        # The embedding
        assert len(inputs_size) == 2

        self.emb_1 = nn.Embedding(inputs_size[0], emb_size)
        self.emb_2 = nn.Embedding(inputs_size[1], emb_size)
        self.emb_1_bias = nn.Embedding(inputs_size[0],1)
        self.emb_2_bias = nn.Embedding(inputs_size[1],1)

        
    def get_embeddings(self, x):

        gene, patient = x[:, 0], x[:, 1]
        # Embedding.
        gene = self.emb_1(gene.long())
        patient = self.emb_2(patient.long())

        return gene, patient

    def get_bias_embeddings(self, x):

        gene, patient = x[:,0], x[:,1]
        gene = self.emb_1_bias(gene.long())
        patient = self.emb_2_bias(patient.long())

        return gene, patient

    def forward(self, x):

        # Get the embeddings
        #import pdb; pdb.set_trace()
        emb_1, emb_2 = self.get_embeddings(x)
        #import pdb; pdb.set_trace()
        bias_1, bias_2 = self.get_bias_embeddings(x)
        b = int(emb_1.shape[0])
        m = int(50)
        t1 = emb_1.view(b,1,m)
        t2 = emb_2.view(b,m,1)

        # Forward pass.
        mlp_output = torch.bmm(t1, t2).squeeze()
        mlp_output += bias_1.squeeze()
        mlp_output += bias_2.squeeze()
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
        
        mlp_output = torch.mm(emb_1, emb_2)
        return mlp_output


class MultipleFactorizedMLP(nn.Module):

    def __init__(self, layers_size, inputs_size, emb_size=2):
        super(MultipleFactorizedMLP, self).__init__()

        self.layers_size = layers_size
        self.emb_size = emb_size
        self.inputs_size = inputs_size


        # The embedding
        assert len(inputs_size) == 3

        self.emb_1 = nn.Embedding(inputs_size[1], emb_size)
        self.emb_2 = nn.Embedding(inputs_size[0], emb_size)
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
        model = model_class(layers_size=opt.layers_size,emb_size=opt.emb_size,inputs_size=inputs_size, data_dir = opt.data_dir, set_gene_emb = opt.set_gene_emb, warm_pca = opt.warm_pca)

    elif opt.model == 'triple':
        model_class = TripleFactorizedMLP
        model = model_class(layers_size=opt.layers_size,emb_size=opt.emb_size,inputs_size=inputs_size)

    elif opt.model == 'multiple':
        model_class = MultipleFactorizedMLP
        model = model_class(layers_size=opt.layers_size,emb_size=opt.emb_size,inputs_size=inputs_size)

    elif opt.model == 'choybenchmark':
        model_class = ChoyEmbedding
        model = model_class(emb_size=50,inputs_size=inputs_size)

    else:
        raise NotImplementedError()

    if model_state is not None:
        model.load_state_dict(model_state)

    return model

