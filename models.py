import torch
import torch.nn.functional as F
from torch import nn


class FactorizedMLP(nn.Module):

    def __init__(self, layers_size, nb_gene, nb_patient, emb_size=2):
        super(FactorizedMLP, self).__init__()

        self.layers_size = layers_size
        self.emb_size = emb_size
        self.nb_gene = nb_gene
        self.nb_patient = nb_patient

        # The embedding
        # TODO: At one point we will probably need to refactor that for it to be more general. Maybe.
        self.gene_embedding = nn.Embedding(nb_gene, emb_size)
        self.tissue_embedding = nn.Embedding(nb_patient, emb_size)

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

        gene, patient = x[:, 0], x[:, 1]

        # Embedding.
        gene = self.gene_embedding(gene.long())
        tissue = self.tissue_embedding(patient.long())

        # Forward pass.
        mlp_input = torch.cat([gene, tissue], 1)

        # TODO: the proper way in pytorch is to use a Sequence layer.
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)

        mlp_output = self.last_layer(mlp_input)

        return mlp_output


def get_model(opt, nb_gene, nb_patient):

    # All of the different models.

    # TODO: find a way to remove the if.
    if opt.model == 'factor':
        model = FactorizedMLP(layers_size=opt.layers_size, emb_size=opt.emb_size, nb_gene=nb_gene, nb_patient=nb_patient)
    else:
        raise NotImplementedError()

    return model
