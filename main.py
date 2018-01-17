import torch
import numpy as np
from torch.autograd import Variable
import os
import argparse
import datasets
import models
import pickle
import time

def build_parser():
    parser = argparse.ArgumentParser(
        description="Model for convolution-graph network (CGN)")

    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=1993, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=100, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--data-dir', default='./data/', help='The folder contening the dataset.')
    parser.add_argument('--save-dir', default='.', help='The folder where everything will be saved.')
    parser.add_argument('--dataset', choices=['gene'], default='gene', help='Which dataset to use.')
    parser.add_argument('--model', choices=['factor'], default='factor', help='Which model to use.')
    parser.add_argument('--cuda', action='store_true', help='If we want to run on gpu.') # TODO: should probably be cpu instead.
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")

    # Model specific options
    parser.add_argument('--layer_size', default=[10, 10], type=int, nargs='+', help='Number of convolution layer in the CGN.')
    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)

    # The experiment unique id.
    param = vars(opt).copy()
    # Removing a bunch of useless tag
    exp_dir = None # TODO: do

    # TODO: set the seed

    # creating the dataset
    print "Getting the dataset..."
    dataset = datasets.get_dataset(opt)

    # Creating a model
    print "Getting the model..."
    my_model = models.get_model(opt)
    print "Our model:"
    print my_model

    # Training optimizer and stuff
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=opt.lr, momentum=opt.momentum) # TODO use ADAM or something. weight decay.

    if opt.on_cuda:
        print "Putting the model on gpu..."
        my_model.cuda()

    # For the loggind and stuff. TODO: do.
    exp_dir = None
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

        # dumping the options
        pickle.dump(opt, open(os.path.join(exp_dir, 'options.pkl'), 'wb'))
        print "We will log everything in ", exp_dir
    else:
        print "Nothing will be log, everything will only be shown on screen."

    # The training.
    for t in range(opt.epoch):

        start_timer = time.time()

        for no_b, mini in enumerate(dataset):

            inputs, targets = mini[0], mini[1]

            inputs = Variable(inputs, requires_grad=False).float()
            targets = Variable(targets, requires_grad=False).float()

            if opt.on_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = my_model(inputs).float()

            # Compute and print loss
            loss = criterion(y_pred, targets)
            print "Done epoch {}. Loss: {}".format(t, loss)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: the logging here.

    print "Done!"

    #TODO: end of training

if __name__ == '__main__':
    main()
