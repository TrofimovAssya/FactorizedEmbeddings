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

    def __init__(self, layers_size, nb_gene, nb_tissue, emb_size=2):
        super(FactorizedMLP, self).__init__()

        self.layers_size = layers_size
        self.emb_size = emb_size
        self.nb_gene = nb_gene
        self.nb_tissue = nb_tissue

        # The embedding
        # TODO: At one point we will probably need to refactor that for it to be more general. Maybe.
        self.gene_embedding = nn.Embedding(nb_gene, emb_size)
        self.tissue_embedding = nn.Embedding(nb_tissue, emb_size)

        # The list of layers.
        layers = []
        dim = [emb_size * 2] + layers_size # Adding the emb size.
        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], 1)



    def forward(self, x):

        #import ipdb; ipdb.set_trace()

        gene, tissue = x[:, 0], x[:, 1]

        # Embedding.
        gene = self.gene_embedding(gene.long())
        tissue = self.tissue_embedding(tissue.long())

        # Forward pass.
        mlp_input = torch.cat([gene, tissue], 1)

        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)

        mlp_output = self.last_layer(mlp_input)

        return mlp_output


def get_model(opt, nb_gene, nb_tissue):

    # All of the different models.

    if opt.model == 'factor':
        model = FactorizedMLP(layers_size=opt.layers_size, emb_size=opt.emb_size, nb_gene=nb_gene, nb_tissue=nb_tissue)
    else:
        raise NotImplementedError()

    return model
