from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class GeneDataset(Dataset):
    """Gene expression dataset"""

    def __init__(self, root_dir='.', data_path='30by30_dataset.npy', data_type_path='30by30_types.npy', data_subtype='30by30_subtypes.npy', transform=None):


        data_path = os.path.join(root_dir, data_path)
        data_type_path = os.path.join(root_dir, data_type_path)
        data_subtype = os.path.join(root_dir, data_subtype)

        # Load the dataset
        data, data_type, data_subtype = np.load(data_path), np.load(data_type_path), np.load(data_subtype)

        self.nb_genes = data.shape[0]
        self.nb_patient = data.shape[1]

        # TODO: for proper pytorch form, we should probably do that on the fly, but heh. todo.
        self.X_data, self.Y_data = self.dataset_make(data, log_transform=True)

        self.root_dir = root_dir
        self.transform = transform # heh

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        sample = self.X_data[idx]
        label = self.Y_data[idx]

        sample = [sample, label]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def dataset_make(self, gene_exp, log_transform=True):

        #    indices_p1 = numpy.random.randint(0, gene_exp.shape[1]-1,nb_examples)
        indices_p1 = np.arange(gene_exp.shape[1])
        indices_g = np.arange(gene_exp.shape[0])
        X_data = np.transpose([np.tile(indices_g, len(indices_p1)), np.repeat(indices_p1, len(indices_g))])
        Y_data = gene_exp[X_data[:, 0], X_data[:, 1]]

        print (Y_data.shape)

        if log_transform:
            Y_data = np.log10(Y_data + 1)
        return X_data, Y_data

def get_dataset(opt):

    # All of the different datasets.

    if opt.dataset == 'gene':
        dataset = GeneDataset(root_dir=opt.data_dir)
    else:
        raise NotImplementedError()

    #TODO: check the num_worker, might be important later on.
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=4)
    return dataloader