import jellyfish
import numpy as np
import os
import argparse
import h5py
import time


def build_parser():
    parser = argparse.ArgumentParser(
        description="Filter out the kmers and save everything in a hdf5 array.")

    parser.add_argument('--data-dir', default='/data/milatmp1/dutilfra/dataset/kmer', help='Where the .jf file is.')
    parser.add_argument('--data-file', default='duodenum1.24.jf', help='The .jf file.')
    parser.add_argument('--save-dir', default=None, help='Where to save the .npy array. By default will save at the same place as the original file.')
    parser.add_argument('--save-file', type=str, default=None, help="The specified file name. By default will have the same base name as the original file.")

    # Model specific options
    parser.add_argument('--min', default=0, type=int, help='The minimum number of occurence.')
    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def add_data(dataset, data, name='kmer'):

    #import ipdb; ipdb.set_trace()

    if name not in dataset:
        dataset.create_dataset(name, data.shape, data=data, maxshape=(None, data.shape[1]))
    else:
        dataset[name].resize((dataset[name].shape[0] + data.shape[0]), axis=0)
        dataset[name][-data.shape[0]:] = data


def main(argv=None):

    opt = parse_args(argv)

    data_dir = opt.data_dir
    data_file = opt.data_file
    save_dir = opt.save_dir
    save_file = opt.save_file
    min_count = opt.min

    if save_dir is None:
        save_dir = data_dir

    if save_file is None:
            save_file = data_file[:-3] + '.hdf5' # Same name, differente extention

    data_file = os.path.join(data_dir, data_file)
    save_file = os.path.join(save_dir, save_file)

    print "We will process {}, keep the kmer that has a count >= {}, and save it in {}".format(data_file, min_count, save_file)

    start = time.time()
    mf = jellyfish.ReadMerFile(data_file)
    kmers = []
    kept_kmer = 0
    all_kmer = 0
    tmp_kmers = []
    fmy = h5py.File(save_file, "w")

    for i, [mer, count] in enumerate(mf):

        all_kmer += 1

        if count < min_count:
            continue

        kept_kmer += 1




        #if i > 1000:
        #    break
        mer = str(mer)
        mer = mer.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')

        sample = list(mer)
        sample.append(int(count))
        sample = np.array(sample).astype(int)
        kmers.append(sample)
        #kmers.append((str(mer), int(count)))

        #import ipdb; ipdb.set_trace()

        if i % 1000000 == 0:
            print "Done {} in {} seconds ".format(i, int(time.time() - start))
            print i, mer, count
            add_data(fmy, np.array(kmers))
            kmers = []

    print "Keeping {}/{} kmers".format(kept_kmer, all_kmer)
    #import ipdb; ipdb.set_trace()
    #kmers = np.array(kmers)

    # Save the data here.

    #import ipdb; ipdb.set_trace()

    # TODO, add a tissue and patient group. Right now is only one single thing.
    #fmy.create_dataset("kmer", kmers.shape, data=kmers)
    fmy.close()
    print "Done!"


if __name__ == '__main__':
    main()
