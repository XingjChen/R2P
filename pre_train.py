import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn.functional as F 
from torch.optim import Adam
import argparse
import logging
import sys
from model import network, metric
from utils import util

device = torch.device("cuda") 


def train(args, train_loader, val_loader, optimizer, model, loss_fn):
    logger.info("Begin training!") 

    best_epoch = 0
    best_val_loss_value = float("inf")

    for epoch in range(0, args.epochs): 

        # Training Loop 
        running_train_loss = 0.0
        y_true = []
        y_pred = []
    
        for data in train_loader: 
        #for data in enumerate(train_loader, 0): 
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs] 
            inputs, outputs = inputs.to(device), outputs.to(device)
            # print(inputs.dtype, outputs.dtype)

            optimizer.zero_grad()   # zero the parameter gradients          
            predicted_outputs = model(inputs)   # predict output from the model 
            train_loss = loss_fn(predicted_outputs, outputs)   # calculate loss for the predicted output
        
            train_loss.backward()   # backpropagate the loss, here for ddp, the specific process will wait other processes finished and update the params
            optimizer.step()        # adjust parameters based on the calculated gradients 
            running_train_loss +=train_loss.item()  # track the loss value 

            y_true.append(outputs)
            y_pred.append(predicted_outputs)
            
        # Calculate training loss value 
        train_loss_value = running_train_loss/len(train_loader) 

        y_true = torch.vstack(y_true) 
        y_pred = torch.vstack(y_pred) 

        # calculate rmse, r2, and pcc

        train_rmse = metric.rmse_loss(y_pred, y_true) 
        train_r2 = metric.r2_loss(y_pred, y_true) 
        train_pcc = metric.pcc_loss(y_pred, y_true)[0]


        # Validation Loop
        running_val_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad(): 
            model.eval() 
            for data in val_loader: 
                inputs, outputs = data 
                inputs, outputs = inputs.to(device),  outputs.to(device)
        
                predicted_outputs = model(inputs) 
                val_loss = loss_fn(predicted_outputs, outputs) 
                
                running_val_loss += val_loss.item()

                y_true.append(outputs)
                y_pred.append(predicted_outputs)
            
        # Calculate validation loss value 
        val_loss_value = running_val_loss/len(val_loader) 

        y_true = torch.vstack(y_true) 
        y_pred = torch.vstack(y_pred) 

        # calculate rmse, r2, and pcc
        val_rmse = metric.rmse_loss(y_pred, y_true) 
        val_r2 = metric.r2_loss(y_pred, y_true) 
        val_pcc = metric.pcc_loss(y_pred, y_true)[0]

        if val_loss_value < best_val_loss_value:
            path = 'pretrain/' + str(args.dataset) + "_" + str(args.pretrain) + '.pth'   # replace one by next one and only save one
            torch.save(model.module.state_dict(), path)
            best_epoch = epoch  
            best_val_loss_value = val_loss_value
        
        # logger.info('Epoch %d: Train Loss: %.4f, Train RMSE: %.4f, Train R2: %.4f, Train PCC: %.4f,\
        #     Val Loss: %.4f, Val RMSE: %.4f, Val R2: %.4f, Val PCC: %.4f,\
        #         Best epoch: %d, Best val loss: %.4f',\
        #             epoch, train_loss_value, train_rmse, train_r2, train_pcc,\
        #                 val_loss_value, val_rmse, val_r2, val_pcc,\
        #                     best_epoch, best_val_loss_value)

        logger.info('Epoch %d: Train Loss: %.4f, Train PCC: %.4f, Val Loss: %.4f,Val PCC: %.4f, Best epoch: %d, Best val loss: %.4f',
                    epoch, train_loss_value, train_pcc, val_loss_value, val_pcc, best_epoch, best_val_loss_value)
        
        


def test(test_loader, loss_fn, input_size, output_size):
    # test Loop
    logger.info("\nBegin test!") 
    # Load the model that we saved at the end of the training loop 
    model = network.MLP(input_size, output_size) 
    path = 'pretrain/' + str(args.dataset) + "_" + str(args.pretrain) + '.pth'
    model.load_state_dict(torch.load(path))
    model.to(device)

    running_test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad(): 
        model.eval() 
        for data in test_loader: 
            inputs, outputs = data 
            inputs, outputs = inputs.to(device), outputs.to(device)

            predicted_outputs = model(inputs) 
            test_loss = loss_fn(predicted_outputs, outputs) 
            running_test_loss += test_loss.item()

            y_true.append(outputs)
            y_pred.append(predicted_outputs)

    # Calculate validation loss value 
    test_loss_value = running_test_loss/len(test_loader)

    y_true = torch.vstack(y_true) 
    y_pred = torch.vstack(y_pred) 

    # calculate rmse, r2, and pcc
    test_rmse = metric.rmse_loss(y_pred, y_true) 
    test_r2 = metric.r2_loss(y_pred, y_true) 
    test_pcc = metric.pcc_loss(y_pred, y_true)[0]

    logger.info('Test Loss: %.4f, Test RMSE: %.4f, Test R2: %.4f, Test PCC: %.4f', test_loss_value, test_rmse, test_r2, test_pcc)

def main(args):
    device = torch.device("cuda") 
    ### Scripts start from here

    train_loader, val_loader, test_loader, input_size, output_size = util.get_dataset(args)

    # Instantiate the model 
    model = network.MLP(input_size, output_size).to(device)
    model = nn.DataParallel(model)

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Training Function 

    train(args, train_loader, val_loader, optimizer, model, loss_fn)

    test(test_loader, loss_fn, input_size, output_size)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()    

    parser.add_argument('--dataset', default='cbmc', type=str)   
    parser.add_argument('--pca', action='store_true', default=True)  
    parser.add_argument('--pca_dim', required='--pca' in sys.argv, default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)  
    parser.add_argument('--epochs', default=500, type=int)  
    parser.add_argument('--pretrain', default = 'r2p', choices=['r2p','p2r'])

    args = parser.parse_args() 
    
    logging.basicConfig(level=logging.INFO, filename= 'pretrain/' + str(args.dataset) + '_' +str(args.pretrain) + '.log', filemode='w')
    logger = logging.getLogger('pretrainlogger')

    main(args)

    # python pre_train.py --pretrain r2p
