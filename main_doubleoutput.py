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
    parser = argparse.ArgumentParser(description="")

    ### Hyperparameter options
    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=260389, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=10000, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    

    ### Dataset specific options
    parser.add_argument('--data-dir', default='./data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--dataset', choices=['gene', 'domaingene', 'impute', 'fedomains', 'doubleoutput'], default='gene', help='Which dataset to use.')
    parser.add_argument('--mask', type=int, default=0, help="percentage of masked values")
    parser.add_argument('--missing', type=int, default=0, help="number of held out combinations for FE domains")
    parser.add_argument('--data-domain', default='.', help='Number of domains in the data for triple factemb')
    parser.add_argument('--transform', default=True,help='log10(exp+1)')
    
    # Model specific options
    parser.add_argument('--layers-size', default=[250, 75, 50, 25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb_size', default=2, type=int, help='The size of the embeddings.')
    parser.add_argument('--set-gene-emb', default='.', help='Starting points for gene embeddings.')
    parser.add_argument('--warm_pca', default='.', help='Datafile to use as a PCA warm start for the sample embeddings')

    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['factor', 'triple', 'multiple','doubleoutput', 'choybenchmark'], default='factor', help='Which model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.') # TODO: should probably be cpu instead.
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=1, help="selectgpu")


    # Monitoring options
    parser.add_argument('--save-error', action='store_true', help='If we want to save the error for each tissue and each gene at every epoch.')
    parser.add_argument('--make-grid', default=True, type=bool,  help='If we want to generate fake patients on a meshgrid accross the patient embedding space')
    parser.add_argument('--nb-gridpoints', default=50, type=int, help='Number of points on each side of the meshgrid')
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--save-dir', default='./testing123/', help='The folder where everything will be saved.')

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
    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    exp_dir = opt.load_folder
    if exp_dir is None: # we create a new folder if we don't load.
        exp_dir = monitoring.create_experiment_folder(opt)

    # creating the dataset
    print ("Getting the dataset...")
    dataset = datasets.get_dataset(opt,exp_dir)

    # Creating a model
    print ("Getting the model...")

    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset.dataset.input_size(), dataset.dataset.additional_info())

    # Training optimizer and stuff
    criterion = torch.nn.MSELoss()

    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # The training.
    print ("Start training.")
    #monitoring and predictions
    predictions =np.zeros((dataset.dataset.nb_patient,dataset.dataset.nb_gene))
    indices_patients = np.arange(dataset.dataset.nb_patient)
    indices_genes = np.arange(dataset.dataset.nb_gene)
    xdata = np.transpose([np.tile(indices_genes, len(indices_patients)),
                          np.repeat(indices_patients, len(indices_genes))])
    progress_bar_modulo = len(dataset)/100
    for t in range(epoch, opt.epoch):

        start_timer = time.time()

        if opt.save_error:
            outfname_g = '_'.join(['gene_epoch',str(t),'prediction.npy'])
            outfname_g = os.path.join(exp_dir,outfname_g)
            outfname_t = '_'.join(['tissue_epoch',str(t),'prediction.npy'])
            outfname_t = os.path.join(exp_dir,outfname_t)
            train_trace = np.zeros((dataset.dataset.nb_gene, dataset.dataset.nb_patient))
        ### making predictions:
        nb_proteins = my_model.emb_3.weight.cpu().data.numpy().shape[0]
        nb_patients = my_model.emb_2.weight.cpu().data.numpy().shape[0]
        predictions_protein = np.zeros((nb_patients, nb_proteins))
        patient_embs = my_model.emb_2.weight.cpu().data.numpy()

        for patient in np.arange(nb_patients):
            new = my_model.generate_datapoint_protein(patient_embs[patient,:], gpu=2)
            new = new.cpu().data.numpy()
            predictions_protein[patient,:] = new[:,0]
        np.save(f'predictions_protein_{epoch}.npy', predictions_protein)


        for no_b, mini in enumerate(dataset):

            inputs, targets, inputs2, targets2 = mini[0], mini[1], mini[2], mini[3]
            

            inputs = Variable(inputs, requires_grad=False).float()
            targets = Variable(targets, requires_grad=False).float()
            inputs2 = Variable(inputs2, requires_grad=False).float()
            targets2 = Variable(targets2, requires_grad=False).float()

            if not opt.cpu:
                inputs = inputs.cuda(opt.gpu_selection)
                targets = targets.cuda(opt.gpu_selection)
                inputs2 = inputs2.cuda(opt.gpu_selection)
                targets2 = targets2.cuda(opt.gpu_selection)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = my_model([inputs, inputs2])

            #if opt.save_error:
                # Log the predicted values per sample and per gene (S.L. validation)
            #    batch_inputs = mini[0].numpy()
            #    predicted_values = y_pred.data.cpu().numpy()
            #    train_trace[batch_inputs[:,0],batch_inputs[:,1]] = predicted_values[:,0]
            targets = torch.reshape(targets,(targets.shape[0],1))
            targets2 = torch.reshape(targets2,(targets2.shape[0],1))
            # Compute and print loss

            loss1 = criterion(y_pred[0], targets)
            loss2 = criterion(y_pred[1], targets2)
            loss = loss1+loss2
            if no_b % 5 == 0:
                print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")

                # Saving the emb
                np.save(os.path.join(exp_dir, 'pixel_epoch_{}'.format(t)),my_model.emb_1.weight.cpu().data.numpy() )
                np.save(os.path.join(exp_dir,'digit_epoch_{}'.format(t)),my_model.emb_2.weight.cpu().data.numpy())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #my_model.generate_datapoint([0,0], opt.gpu_selection)
        #monitoring.save_predictions(exp_dir, predictions)


#        for i in range(0,xdata.shape[0],1000):
#            #import pdb; pdb.set_trace()
#            inputs = torch.FloatTensor(xdata[i:i+1000,:])
#            inputs = Variable(inputs, requires_grad=False).float()
#            if not opt.cpu:
#                inputs = inputs.cuda(opt.gpu_selection)
#            y_pred = my_model(inputs).float()
#            predictions[inputs.data.cpu().numpy()[:,1].astype('int32'),inputs.data.cpu().numpy()[:,0].astype('int32')] = y_pred.data.cpu().numpy()[:,0]
        #      monitoring.dump_error_by_tissue(train_trace, dataset.dataset.data, outfname_t, exp_dir, dataset.dataset.data_type, dataset.dataset.nb_patient)
        #      monitoring.dump_error_by_gene(train_trace, dataset.dataset.data, outfname_g, exp_dir)


        #print ("Saving the model...")
        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)


#        if opt.make_grid:
#            print ('generating grid and datapoints')

#            nb_points = opt.nb_gridpoints
#            x_min = min(my_model.emb_2.weight.data.cpu().numpy()[:,0])
#            y_min = min(my_model.emb_2.weight.data.cpu().numpy()[:,1])
#            x_max = max(my_model.emb_2.weight.data.cpu().numpy()[:,0])
#            y_max = max(my_model.emb_2.weight.data.cpu().numpy()[:,1])
#            x = np.linspace((np.floor(x_min*100))/100,(np.ceil(x_max*100))/100,nb_points)
#            y = np.linspace((np.floor(y_min*100))/100,(np.ceil(y_max*100))/100,nb_points)
#            X, Y = np.meshgrid(x,y)
#            T = []
#            print (f"I'll be making {(X.shape[0]*X.shape[1])**2} samples for a grid of {X.shape[0]} by {X.shape[1]} ")
#            count = 0

            #for ix,iy in zip(X.reshape((X.shape[0]*X.shape[1],)),Y.reshape((Y.shape[0]*Y.shape[1],))):
            #    if count%1000==0:
            #        print(f'made {count} samples')
                #import pdb; pdb.set_trace()
                #np.save(os.path.join(exp_dir,'generated_patient{}'.format(count)),my_model.generate_datapoint([ix,iy],opt.gpu_selection).data.cpu().numpy())
#                T.append(my_model.generate_datapoint([ix,iy],opt.gpu_selection).data.cpu().numpy())
#                count+=1
#            np.save(os.path.join(exp_dir,'meshgrid_x.npy'),X)
#            np.save(os.path.join(exp_dir,'meshgrid_y.npy'),Y)
#            np.save(os.path.join(exp_dir,'generated_mesh_samples.npy'),T)



    #getting a datapoint embedding coordinate                                                                                                                                 

    #inputs = torch.FloatTensor(xdata[i:i+1000,:])
    #inputs = Variable(inputs, requires_grad=False).float()
    #if not opt.cpu:
    #    inputs = inputs.cuda(opt.gpu_selection)
    #    y_pred = my_model(inputs).float()
    #predictions[inputs.data.cpu().numpy()[:,1].astype('int32'),inputs.data.cpu().numpy()[:,0].astype('int32')] = y_pred.data.cpu().numpy()[:,0]
    #emb_2 = (np.ones(emb_1.shape[0]*2).reshape((emb_1.shape[0],2)))*[0,0]




    #TODO: end of training, save the model and blah.


if __name__ == '__main__':
    main()
