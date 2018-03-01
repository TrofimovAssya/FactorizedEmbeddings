#!/usr/bin/env python
import torch
import pdb
import numpy as np
from torch.autograd import Variable
import os
import argparse
import datasets
import models
import pickle
import time
import monitoring
#
def build_parser():
    parser = argparse.ArgumentParser(
        description="Model for convolution-graph network (CGN)")

    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=1993, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=10000, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--data-dir', default='./data/', help='The folder contening the dataset.')
    parser.add_argument('--save-dir', default='./testing123/', help='The folder where everything will be saved.')
    parser.add_argument('--dataset', choices=['gene', 'kmer'], default='gene', help='Which dataset to use.')
    parser.add_argument('--model', choices=['factor', 'bag'], default='factor', help='Which model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.') # TODO: should probably be cpu instead.
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")

    # Model specific options
    parser.add_argument('--layers-size', default=[150, 100, 100, 50, 25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb_size', default=2, type=int, help='The size of the embeddings.')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')

    # Monitoring options
    parser.add_argument('--save-error', action='store_true', help='If we want to save the error for each tissue and each gene at every epoch.')

    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)
    # TODO: set the seed

    exp_dir = monitoring.create_experiment_folder(opt)

    # creating the dataset
    print "Getting the dataset..."
    dataset = datasets.get_dataset(opt)

    # Creating a model
    print "Getting the model..."
    # I might understand something wrong here, but shouldn't be 30 id, instead of 800?
    # Or is the it is a cross product between tissue and patient?
    my_model = models.get_model(opt, dataset.dataset.input_size())
    print "Our model:"
    print my_model

    # Training optimizer and stuff
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(my_model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay) # TODO use ADAM or something. weight decay.

    if not opt.cpu:
        print "Putting the model on gpu..."
        my_model.cuda()

    # The training.

    print "Start training."
    progress_bar_modulo = len(dataset)/10
    
    for t in range(opt.epoch):

        start_timer = time.time()
        
        outfname_g = '_'.join(['gene_epoch',str(t),'prediction.npy'])
        outfname_g = os.path.join(exp_dir,outfname_g)
        outfname_t = '_'.join(['tissue_epoch',str(t),'prediction.npy'])
        outfname_t = os.path.join(exp_dir,outfname_t)
        train_trace = np.zeros((dataset.dataset.nb_gene, dataset.dataset.nb_patient))

        for no_b, mini in enumerate(dataset):
            if no_b%progress_bar_modulo==0:
                print '#'+str(no_b%progress_bar_modulo),
            inputs, targets = mini[0], mini[1]

            inputs = Variable(inputs, requires_grad=False).float()
            targets = Variable(targets, requires_grad=False).float()

            if not opt.cpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = my_model(inputs).float()

            # Log the predicted values per sample and per gene (S.L. validation)
            batch_inputs = mini[0].numpy()
            predicted_values = y_pred.data.cpu().numpy()
            train_trace[batch_inputs[:,0],batch_inputs[:,1]] = predicted_values[:,0]
            # Compute and print loss
            
            loss = criterion(y_pred, targets)
            # TODO: the logging here.
            if ((no_b*opt.batch_size) % 10000000) == 0:
                print "Doing epoch {}, examples {}/{}. Loss: {}".format(t, no_b, len(dataset), loss.data[0])

                # Saving the emb
                monitoring.save_everything(exp_dir, t, my_model, dataset.dataset)


            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if opt.save_error:
            monitoring.dump_error_by_tissue(train_trace, dataset.dataset.data, outfname_t, exp_dir, dataset.dataset.data_type, dataset.dataset.nb_patient)
            monitoring.dump_error_by_gene(train_trace, dataset.dataset.data, outfname_g, exp_dir)


    print "Done!"

    #TODO: end of training, save the model and blah.


if __name__ == '__main__':
    main()
