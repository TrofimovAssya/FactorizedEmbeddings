from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import pdb
from collections import OrderedDict
import shutil

class GeneDataset(Dataset):
    """Gene expression dataset"""

    def __init__(self,root_dir='.',save_dir='.',data_file='data.npy', transform=None, masked = 0):


        data_path = os.path.join(root_dir, data_file)
        self.masked = masked
        # Load the dataset
        self.data = np.load(data_path)

        self.nb_patient = self.data.shape[0]
        self.nb_gene = self.data.shape[1]
        print (self.nb_gene)
        print (self.nb_patient)
        self.nb_tissue = 1

        self.root_dir = root_dir
        self.transform = transform # heh
        self.X_data, self.Y_data = self.dataset_make(self.data,log_transform=True)

        ### masking the data if needed
        permutation = np.random.permutation(np.arange(self.X_data.shape[0]))
        keep = int((100-self.masked)*self.X_data.shape[0]/100)
        self.X_data = self.X_data[:keep,:]
        self.Y_data = self.Y_data[:keep]


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        sample = self.X_data[idx]
        label = self.Y_data[idx]

        sample = [sample, label]

        return sample

    def dataset_make(self, gene_exp, log_transform=False):
        indices_p1 = np.arange(gene_exp.shape[0])
        indices_g = np.arange(gene_exp.shape[1])
        X_data = np.transpose([np.tile(indices_g, len(indices_p1)), np.repeat(indices_p1, len(indices_g))])
        Y_data = gene_exp[X_data[:, 1], X_data[:, 0]]

        print (f"Total number of examples: {Y_data.shape} ")

        if log_transform:
            Y_data = np.log10(Y_data + 1)
        return X_data, Y_data

    def input_size(self):
        return self.nb_gene, self.nb_patient

    def extra_info(self):
        info = OrderedDict()
        return info


class DomainGeneDataset(Dataset):
    """Gene expression dataset"""

    def __init__(self,root_dir='.',save_dir='.',data_file='data.npy', domain_file = 'domain.npy', transform=True, masked = 0):


        data_path = os.path.join(root_dir, data_file)
        domain_path = os.path.join(root_dir, domain_file)
        # Load the dataset
        self.data = np.load(data_path)
        self.domain = np.load(domain_path)
        self.masked = masked

        self.nb_patient = self.data.shape[0]
        self.nb_gene = self.data.shape[1]
        self.nb_domain = len(set(self.domain))
        print (self.nb_gene)
        print (self.nb_patient)
        print (self.nb_domain)
        self.nb_tissue = 1

        self.root_dir = root_dir
        self.transform = transform
        self.X_data, self.Y_data = self.dataset_make(self.data,self.domain,log_transform=True)
        ### TODO: figure out how to control the transform from main arguments

        ### masking the data if needed
        permutation = np.random.permutation(np.arange(self.X_data.shape[0]))
        keep = int((100-self.masked)*self.X_data.shape[0]/100)
        self.X_data = self.X_data[:keep,:]
        self.Y_data = self.Y_data[:keep]


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        sample = self.X_data[idx]
        label = self.Y_data[idx]

        sample = [sample, label]

        return sample

    def dataset_make(self, gene_exp, domains, log_transform=False):
        indices_p1 = np.arange(gene_exp.shape[0])
        indices_g = np.arange(gene_exp.shape[1])
        indices_d = np.arange(len(set(domains)))

        X_data = np.transpose([np.tile(indices_g, len(indices_p1)), np.repeat(indices_p1, len(indices_g))])
        X_dom = domains[X_data[:,1]]
        X_dom = X_dom.reshape((X_dom.shape[0],1))
        X_data = np.hstack((X_data, X_dom))
        X_data = X_data.astype('int32')
        Y_data = gene_exp[X_data[:, 1], X_data[:, 0]]


        print (f"Total number of examples: {Y_data.shape} ")

        if log_transform:
            Y_data = np.log10(Y_data + 1)
        return X_data, Y_data

    def input_size(self):
        return self.nb_gene, self.nb_patient, self.nb_domain

    def extra_info(self):
        info = OrderedDict()
        return info


class FEDomainsDataset(Dataset):
    """FE domains dataset
    This dataset creates a factorized embedding for each variation to factor out.

    """

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy', domain_file = 'domain.npy', nb_factors = 2, transform=True, masked = 0):


        data_path = os.path.join(root_dir, data_file)
        domain_path = os.path.join(root_dir, domain_file)
        # Load the dataset
        self.data = np.load(data_path)
        self.domain = np.load(domain_path)
        self.masked = masked
        self.nb_factors = nb_factors

        self.nb_patient = len(set(self.domain[:,0]))
        self.nb_gene = self.data.shape[1]
        

        print (self.nb_gene)
        print (self.nb_patient)
        self.nb_tissue = 1

        self.root_dir = root_dir
        self.transform = transform
        self.X_indices, self.Y_data = self.dataset_make(self.data,log_transform=True)

        self.X_data = np.zeros((self.Y_data.shape[0],3))
        self.X_data[:,0] = self.domain[self.X_indixes[:,1],0]
        self.X_data[:,1] = self.domain[self.X_indixes[:,1],1]
        self.X_data[:,2] = self.X_indices[:,0]
        
        ### masking the data if needed
        permutation = np.random.permutation(np.arange(self.X_data.shape[0]))
        keep = int((100-self.masked)*self.X_data.shape[0]/100)
        
        self.X_data = self.X_data[:keep,:]
        self.Y_data = self.Y_data[:keep]


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        sample = self.X_data[idx]
        label = self.Y_data[idx]

        sample = [sample, label]

        return sample

    def dataset_make(self, gene_exp, log_transform=False):
        indices_p1 = np.arange(gene_exp.shape[0])
        indices_g = np.arange(gene_exp.shape[1])

        X_data = np.transpose([np.tile(indices_g, len(indices_p1)), np.repeat(indices_p1, len(indices_g))])
        X_data = X_data.astype('int32')
        Y_data = gene_exp[X_data[:, 1], X_data[:, 0]]


        print (f"Total number of examples: {Y_data.shape} ")

        if log_transform:
            Y_data = np.log10(Y_data + 1)
        return X_data, Y_data

    def input_size(self):

        return np.max(self.X_data[:,2])+1, np.max(self.X_data[:,1])+1, np.max(self.X_data[:,0])+1

    def extra_info(self):
        info = OrderedDict()
        return info

class ImputeGeneDataset(Dataset):
    """Gene expression dataset"""

    def __init__(self,root_dir='.',save_dir='.',data_file='data.npy', transform=None, masked = 0):


        data_path = os.path.join(root_dir, data_file)
        ### masked is the number of random genes that will be part of the dataset.
        self.masked = masked
        # Load the dataset
        self.data = np.load(data_path)

        self.nb_patient = self.data.shape[0]
        self.nb_gene = self.data.shape[1]
        print (self.nb_gene)
        print (self.nb_patient)
        self.nb_tissue = 1

        self.root_dir = root_dir
        self.transform = transform # heh
        self.X_data, self.Y_data = self.dataset_make(self.data,log_transform=True)

        if masked>0:
            ### masking the data if needed
            permutation = np.random.permutation(np.arange(self.nb_gene))
            keep = permutation[:masked]
            import pdb; pdb.set_trace()
            self.X_data = self.X_data[:keep,:]
            self.Y_data = self.Y_data[:keep]


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        sample = self.X_data[idx]
        label = self.Y_data[idx]

        sample = [sample, label]

        return sample

    def dataset_make(self, gene_exp, log_transform=False):
        indices_p1 = np.arange(gene_exp.shape[0])
        indices_g = np.arange(gene_exp.shape[1])
        X_data = np.transpose([np.tile(indices_g, len(indices_p1)), np.repeat(indices_p1, len(indices_g))])
        Y_data = gene_exp[X_data[:, 1], X_data[:, 0]]

        print (f"Total number of examples: {Y_data.shape} ")

        if log_transform:
            Y_data = np.log10(Y_data + 1)
        return X_data, Y_data

    def input_size(self):
        return self.nb_gene, self.nb_patient

    def extra_info(self):
        info = OrderedDict()
        return info


def get_dataset(opt,exp_dir, masked=0):

    # All of the different datasets.

    if opt.dataset == 'gene':
        dataset = GeneDataset(root_dir=opt.data_dir, save_dir = exp_dir, data_file = opt.data_file, transform = opt.transform, masked = opt.mask)
    elif opt.dataset == 'domaingene':
        dataset = DomainGeneDataset(root_dir=opt.data_dir, save_dir = exp_dir,data_file = opt.data_file, 
            domain_file = opt.data_domain, transform = opt.transform, masked = opt.mask)
    elif opt.dataset == 'impute':
        dataset = ImputeGeneDataset(root_dir=opt.data_dir, save_dir = exp_dir, data_file = opt.data_file, transform = opt.transform, masked = masked)
    elif opt.dataset == 'fedomains': 
        dataset = FEDomainsDataset(root_dir=opt.data_dir, save_dir = exp_dir,data_file = opt.data_file, 
            domain_file = opt.data_domain, transform = opt.transform, masked = opt.mask)
    else:
        raise NotImplementedError()

    #TODO: check the num_worker, might be important later on, for when we will use a bunch of big files.
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=1)

    return dataloader
