import os
import numpy as np
import random
import torch
import shutil
import models
import datetime

def save_everything(dir_name, epoch, model, dataset):

    #import ipdb; ipdb.set_trace()

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = os.path.join(dir_name, 'epoch_{}'.format(epoch))

    emb = model.emb_2.weight.cpu().data.numpy()
    dump_emb_size2(emb, file_name, dataset.extra_info(), dataset.nb_patient)


def dump_emb_size2(emb, file_name, extra_info, nb_patient):

    assert emb.shape[1] == 2

    b = open(file_name, 'wb')

    info_keys, info_values = extra_info.keys(), extra_info.values()
    b.write("\t".join(['dimension1', 'dimension2'] + info_keys) + '\n')
    for p in xrange(nb_patient):
        b.write("\t".join([str(emb[p, 0]), str(emb[p, 1])] + [str(info[p]) for info in info_values]) + '\n')
    b.close()

def dump_error_by_gene(pred, data, file_name, dir_name):
    if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    assert pred.shape == data.shape
    a = (data-pred)**2
    meanval = np.mean(a,axis=1)
    np.save(file_name,meanval)

def dump_error_by_tissue(pred, data, file_name, dir_name, data_type, nb_patient):


    #import ipdb; ipdb.set_trace()

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    assert pred.shape == data.shape
    b = open(file_name, 'wb')
    b.write("\t".join(['type','MSE']) + '\n')
    for ttypes in set(data_type):
        meanval = np.mean((pred[:,data_type == ttypes] - data[:,data_type == ttypes])**2)
        b.write("\t".join([ttypes, str(meanval)]) + '\n')
    b.close()    

    
def create_experiment_folder(opt):

    params = vars(opt).copy()
    params = str(params)

    # create a experiment folder
    this_hash = random.getrandbits(128)
    this_hash = "%032x" % this_hash # in hex

    exp_dir = os.path.join(opt.save_dir, this_hash)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    f = open(os.path.join(exp_dir,'run_parameters'), 'wb')
    f.write(params+'\n')
    f.close()
    print vars(opt)
    print "Saving the everything in {}".format(exp_dir)

    with open(os.path.join(opt.save_dir, 'experiment_table.txt'), 'a') as f:
        f.write('time: {} folder: {} experiment: {}\n'.format(datetime.datetime.now(), this_hash, params))

    return exp_dir


def save_checkpoint(model, optimizer, epoch, opt, exp_dir, filename='checkpoint.pth.tar'):

    state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'opt' : opt
        }

    filename = os.path.join(exp_dir, filename)
    torch.save(state, filename)

def load_checkpoint(load_folder, opt, input_size, filename='checkpoint.pth.tar'):

    # Model
    model_state = None

    # Epoch
    epoch = 0

    # Optimizser
    optimizer_state = None

    # Options
    new_opt = opt

    # Load the states if we saved them.
    if opt.load_folder:

        # Loading all the state
        filename = os.path.join(load_folder, filename)
        if os.path.isfile(filename):
            print "=> loading checkpoint '{}'".format(filename)
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']

            # Loading the options
            new_opt = checkpoint['opt']
            print "Loading the model with these parameters: {}".format(new_opt)

            # Loading the state
            model_state = checkpoint['state_dict']
            optimizer_state = checkpoint['optimizer']
            epoch = checkpoint['epoch']

            # We override some of the options between the runs, otherwise it might be a pain.
            new_opt.epoch = opt.epoch

            print"=> loaded checkpoint '{}' (epoch {})".format(filename, epoch)
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    # Get the network
    my_model = models.get_model(new_opt, input_size, model_state)

    # Get the optimizer
    optimizer = torch.optim.RMSprop(my_model.parameters(), lr=new_opt.lr, weight_decay=new_opt.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    print "Our model:"
    print my_model

    return my_model, optimizer, epoch, new_opt

#
# def sample_embedding_dump(emb, epoch, g_emb_size, data_type, data_subtype, pca=False, nmf=False):
#     if pca:
#         b = open('pca_embeddings', 'wb')
#     elif nmf:
#         b = open('nmf_emebddings', 'wb')
#     else:
#         b = open('_'.join(['embeddings', str(epoch), str(g_emb_size)]), 'wb')
#         emb = emb[0]
#
#     b.write("\t".join(['dimension1', 'dimension2', 'type', 'subtype']) + '\n')
#     for p in xrange(nb_patient):
#         b.write("\t".join([str(emb[p, 0]), str(emb[p, 1]), str(data_type[p]), str(data_subtype[p])]) + '\n')
#     b.close()
