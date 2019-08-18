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

    ### Loading options
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')

    ### Dataset specific options
    parser.add_argument('--data-dir', default='./data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--dataset', choices=['gene', 'domaingene', 'impute'], default='impute', help='Which dataset to use.')
    parser.add_argument('--data-domain', default='.', help='Number of domains in the data for triple factemb')
    parser.add_argument('--transform', default=True,help='log10(exp+1)')
    parser.add_argument('--imputation-list', default = [1,5,10,25,100,250,500,1000,2500,5000,10000,25000], type=int, nargs='+', help='number of genes to give the algorithm')
    parser.add_argument('--nb-shuffles', default = 25, type=int, help='number of genes to give the algorithm')
    
    # GPU options
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.') # TODO: should probably be cpu instead.
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="selectgpu")


    # Making grids
    parser.add_argument('--make-grid', default=True, type=bool,  help='If we want to generate fake patients on a meshgrid accross the patient embedding space')
    parser.add_argument('--nb-gridpoints', default=50, type=int, help='Number of points on each side of the meshgrid')

    # Saving directory options
    parser.add_argument('--new-save-dir', default='./imputation_shuffles123', help='The folder where everything will be saved.')

    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt


#### replace the embeddings of emb_2 for the N first patients by zeros
#### train on only the new dataset on the number of genes specified
#### generate the rest of samples

def impute(argv=None):

    opt = parse_args(argv)

    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    exp_dir = opt.load_folder
    new_exp_dir = opt.new_save_dir
    if exp_dir is None: 
        print ("Experiment doesn't exist!")
    else:
        # creating the dataset
        print ("Getting the dataset...")
        dataset = datasets.get_dataset(opt,exp_dir)

        # Creating a model
        print ("Getting the model...")
        my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset.dataset.input_size(), )
        
        ### Making sure updates are only on the patient embedding layer
        my_model.freeze_all()
        optimizer = torch.optim.RMSprop(my_model.emb_2.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        ### Replacing the first embeddings as the new number of patients to predict
        my_model.emb_2.weight[:dataset.dataset.nb_patient,:] = Variable(torch.FloatTensor(np.zeros((dataset.dataset.nb_patient,2))), requires_grad=False).float()

        # Training optimizer and stuff
        criterion = torch.nn.MSELoss()

        if not opt.cpu:
            print ("Putting the model on gpu...")
            my_model.cuda(opt.gpu_selection)

        # The training.
        print ("Start training.")




        #monitoring and predictions
        predictions =np.zeros((dataset.dataset.nb_patient,my_model.emb_1.shape[0]))
        indices_patients = np.arange(dataset.dataset.nb_patient)
        indices_genes = np.arange(my_model.emb_1.shape[0])
        xdata = np.transpose([np.tile(indices_genes, len(indices_patients)),
                              np.repeat(indices_patients, len(indices_genes))])


        for nb_genes in opt.imputation_list:
            print (f'Imputation with {nb_genes} genes given...')
            for shuffle in range(opt.nb_shuffles):
                progress_bar_modulo = len(dataset)/100

                print ("Re-getting the dataset...")
                dataset = datasets.get_dataset(opt,exp_dir, nb_genes)

                #monitoring and predictions
                predictions =np.zeros((dataset.dataset.nb_patient,my_model.emb_1.shape[0]))
                indices_patients = np.arange(predictions.shape[0])
                indices_genes = np.arange(predictions.shape[1])
                xdata = np.transpose([np.tile(indices_genes, len(indices_patients)),
                                      np.repeat(indices_patients, len(indices_genes))])

                for t in range(epoch, opt.epoch):

                    start_timer = time.time()

                    #if opt.save_error:
                        #outfname_g = f'shuffle_{shuffle}_{nb_genes}_genes_epoch_{t}_prediction_genes.npy'
                        #outfname_g = os.path.join(new_exp_dir,outfname_g)
                        #outfname_t = f'shuffle_{shuffle}_{nb_genes}_genes_epoch_{t}_prediction_tissue.npy'
                        #outfname_t = os.path.join(new_exp_dir,outfname_t)
                        #train_trace = np.zeros((dataset.dataset.nb_gene, dataset.dataset.nb_patient))

                    for no_b, mini in enumerate(dataset):

                        inputs, targets = mini[0], mini[1]

                        inputs = Variable(inputs, requires_grad=False).float()
                        targets = Variable(targets, requires_grad=False).float()

                        if not opt.cpu:
                            inputs = inputs.cuda(opt.gpu_selection)
                            targets = targets.cuda(opt.gpu_selection)

                        # Forward pass: Compute predicted y by passing x to the model
                        y_pred = my_model(inputs).float()

                        #if opt.save_error:
                            ## Log the predicted values per sample and per gene (S.L. validation)
                            #batch_inputs = mini[0].numpy()
                            #predicted_values = y_pred.data.cpu().numpy()
                            #train_trace[batch_inputs[:,0],batch_inputs[:,1]] = predicted_values[:,0]
                        #import pdb; pdb.set_trace()
                        targets = torch.reshape(targets,(targets.shape[0],1))
                        # Compute and print loss

                        loss = criterion(y_pred, targets)
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
                    

            for i in range(0,xdata.shape[0],1000):
                #import pdb; pdb.set_trace()
                inputs = torch.FloatTensor(xdata[i:i+1000,:])
                inputs = Variable(inputs, requires_grad=False).float()
                if not opt.cpu:
                    inputs = inputs.cuda(opt.gpu_selection)
                y_pred = my_model(inputs).float()
                predictions[inputs.data.cpu().numpy()[:,1].astype('int32'),inputs.data.cpu().numpy()[:,0].astype('int32')] = y_pred.data.cpu().numpy()[:,0]
            outfname_pred = f'shuffle_{shuffle}_{nb_genes}_genes_epoch_{t}_prediction.npy'
            outfname_pred = os.path.join(new_exp_dir,outfname_pred)
            monitoring.save_predictions(outfname_pred, predictions)


            #      monitoring.dump_error_by_tissue(train_trace, dataset.dataset.data, outfname_t, exp_dir, dataset.dataset.data_type, dataset.dataset.nb_patient)
            #      monitoring.dump_error_by_gene(train_trace, dataset.dataset.data, outfname_g, exp_dir)



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

    #            for ix,iy in zip(X.reshape((X.shape[0]*X.shape[1],)),Y.reshape((Y.shape[0]*Y.shape[1],))):
    #                if count%1000==0:
    #                    print(f'made {count} samples')
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
    impute()
