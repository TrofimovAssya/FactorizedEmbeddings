import os

def save_everything(dir_name, epoch, model, dataset):

    #import ipdb; ipdb.set_trace()

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = os.path.join(dir_name, 'epoch_{}'.format(epoch))

    emb = model.tissue_embedding.weight.cpu().data.numpy()
    dump_emb_size2(emb, file_name, dataset.data_type, dataset.data_subtype, dataset.nb_patient)


def dump_emb_size2(emb, file_name, data_type, data_subtype, nb_patient):

    assert emb.shape[1] == 2

    b = open(file_name, 'wb')

    b.write("\t".join(['dimension1', 'dimension2', 'type', 'subtype']) + '\n')
    for p in xrange(nb_patient):
        b.write("\t".join([str(emb[p, 0]), str(emb[p, 1]), str(data_type[p]), str(data_subtype[p])]) + '\n')
    b.close()


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