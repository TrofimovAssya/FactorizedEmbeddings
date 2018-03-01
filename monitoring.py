import os
import numpy as np
import random

def save_everything(dir_name, epoch, model, dataset):

    #import ipdb; ipdb.set_trace()

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = os.path.join(dir_name, 'epoch_{}'.format(epoch))

    emb = model.emb_2.weight.cpu().data.numpy()
    dump_emb_size2(emb, file_name, dataset.data_type, dataset.data_subtype, dataset.nb_patient)


def dump_emb_size2(emb, file_name, data_type, data_subtype, nb_patient):

    assert emb.shape[1] == 2

    b = open(file_name, 'wb')

    b.write("\t".join(['dimension1', 'dimension2', 'type', 'subtype']) + '\n')
    for p in xrange(nb_patient):
        b.write("\t".join([str(emb[p, 0]), str(emb[p, 1]), str(data_type[p]), str(data_subtype[p])]) + '\n')
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
    return exp_dir

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
