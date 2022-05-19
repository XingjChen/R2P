import torch
import torch.nn as nn 
from torch.optim import Adam
import argparse
from model import network, metric
import sys
from utils import util
import logging
from itertools import cycle

from torch.utils.tensorboard import SummaryWriter   


# Training Function

device = torch.device("cuda") 

def train(args, train_loader, pair_loader, val_loader, test_loader, optim_r2p, optim_p2r, model_r2p, model_p2r, loss_fn):
    logger.info("Begin training!") 

    best_epoch = 0
    best_val_loss_value = float("inf")

    for epoch in range(0, args.epochs): 

        # Training Loop 
        running_train_loss = 0.0
        y_true = []
        y_pred = []
    
        # for data in train_loader: 

        
        if args.pretrain == 'dual_combined':

            for i, data in enumerate(zip(train_loader[0], cycle(train_loader[1]))):
    
            #for data in enumerate(train_loader, 0): 
                rna, protein = data[0][0],  data[1][0] # get the input and real species as outputs; data is a list of [inputs, outputs] 
                rna, protein = rna.to(device), protein.to(device)
                
                # print(rna.shape, protein.shape)


                optim_r2p.zero_grad()   # zero the parameter gradients
                optim_p2r.zero_grad()   # zero the parameter gradients   

                pro_pred = model_r2p(rna)
                rna_pred = model_p2r(pro_pred)
                rna_loss = loss_fn(rna_pred, rna)

                rna_pred = model_p2r(protein)
                pro_pred = model_r2p(rna_pred)
                pro_loss = loss_fn(pro_pred, protein)

                all_loss = rna_loss + pro_loss

                all_loss.backward()   # backpropagate the loss, here for ddp, the specific process will wait other processes finished and update the params
                
                optim_r2p.step()        # adjust parameters based on the calculated gradients 
                optim_p2r.step()        # adjust parameters based on the calculated gradients 

                running_train_loss +=all_loss.item()  # track the loss value 

                # print('train mse',  metric.mse_loss()   )

            # Calculate training loss value 
            train_loss_value = running_train_loss/len(train_loader) 

            writer.add_scalar('Train/loss', train_loss_value, epoch)

            # Validation Loop
            running_val_loss = 0.0

            with torch.no_grad():

                model_r2p.eval()   
                model_p2r.eval() 

                for i, data in enumerate(zip(train_loader[0], cycle(val_loader[1]))):
                    
                    rna, protein = data[0][0],  data[1][0]  # get the input and real species as outputs; data is a list of [inputs, outputs] 
                    rna, protein = rna.to(device), protein.to(device)
            
                    pro_pred = model_r2p(rna)
                    rna_pred = model_p2r(pro_pred)
                    rna_loss = loss_fn(rna_pred, rna)





                    rna_pred = model_p2r(protein)
                    pro_pred = model_r2p(rna_pred)
                    pro_loss = loss_fn(pro_pred, protein)

                    all_loss = rna_loss + pro_loss
                    
                    running_val_loss += all_loss.item()

            # Calculate validation loss value 
            val_loss_value = running_val_loss/len(val_loader) 

            writer.add_scalar('Val/loss', val_loss_value, epoch)
                

            if val_loss_value < best_val_loss_value:
                
            
                path = 'dual/' + str(args.dataset) + '_r2p.pth'
                torch.save(model_r2p.module.state_dict(), path)

                path = 'dual/' + str(args.dataset) + '_p2r.pth'
                torch.save(model_p2r.module.state_dict(), path)

                best_epoch = epoch  
                best_val_loss_value = val_loss_value
            
            logger.info('Epoch %d: Train Loss: %.4f, Val Loss: %.4f, Best epoch: %d, Best val loss: %.4f',
                        epoch, train_loss_value, val_loss_value, best_epoch, best_val_loss_value)
            


        # test Loop
            running_test_loss = 0.0
            y_true = []
            y_pred = []

            for data in test_loader: 
                inputs, outputs = data 
                inputs, outputs = inputs.to(device), outputs.to(device)

                predicted_outputs = model_r2p(inputs) 
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

            writer.add_scalar('Test/loss', test_loss_value, epoch)

            logger.info('Test Loss: %.4f, Test RMSE: %.4f, Test R2: %.4f, Test PCC: %.4f', test_loss_value, test_rmse, test_r2, test_pcc)




def test(test_loader, loss_fn, input_size, output_size):
    logger.info("\nBegin test!") 
    # Load the model that we saved at the end of the training loop 
    model = network.MLP(input_size, output_size) 
    path = 'dual/' + str(args.dataset) + "_r2p.pth" 
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

   
    train_loader, pair_loader, val_loader, test_loader, rna_dims, pro_dims = util.get_dataset(args)

    # Instantiate the model
    model_r2p = network.MLP(rna_dims, pro_dims)
    path = 'pretrain/' + str(args.dataset) + "_r2p.pth"  
    model_r2p.load_state_dict(torch.load(path))
    model_r2p.to(device)
    model_r2p = nn.DataParallel(model_r2p)

    model_p2r = network.MLP(pro_dims, rna_dims).to(device)
    path = 'pretrain/' + str(args.dataset) + '_p2r.pth'
    model_p2r.load_state_dict(torch.load(path))
    model_p2r.to(device)
    model_p2r = nn.DataParallel(model_p2r)

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    optim_r2p = Adam(model_r2p.parameters(), lr=0.001)
    optim_p2r = Adam(model_p2r.parameters(), lr=0.001)

    loss_fn = nn.MSELoss()
    
    train(args, train_loader, pair_loader, val_loader, test_loader, optim_r2p, optim_p2r, model_r2p, model_p2r, loss_fn)

    test(test_loader, loss_fn, rna_dims, pro_dims)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()    

    parser.add_argument('--dataset', default='cbmc', type=str)   
    parser.add_argument('--pca', action='store_true', default=True)  
    parser.add_argument('--pca_dim', required='--pca' in sys.argv, default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)  
    parser.add_argument('--epochs', default=500, type=int)  
    parser.add_argument('--pretrain', default = 'r2p', choices=['r2p','p2r', 'dual_unlabel', 'dual_combined'])

    args = parser.parse_args() 
    
    logging.basicConfig(level=logging.INFO, filename= 'dual/' + str(args.dataset) + '_dual.log', filemode='w')
    logger = logging.getLogger('duallogger')

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter('dual/tensorboard/' + str(args.dataset))

    main(args)

    # python pre_train.py --pretrain r2p


# tensorboard --logdir=./path/to/the/folder --port 8123
# tensorboard --logdir=runsimport torch
import torch.nn as nn 
from torch.optim import Adam
import argparse
from model import network, metric
import sys
from utils import util
import logging
from itertools import cycle

from torch.utils.tensorboard import SummaryWriter   


# Training Function

device = torch.device("cuda") 

def train(args, train_loader, pair_loader, val_loader, test_loader, optim_r2p, optim_p2r, model_r2p, model_p2r, loss_fn):
    logger.info("Begin training!") 

    best_epoch = 0
    best_val_loss_value = float("inf")

    for epoch in range(0, args.epochs): 

        # Training Loop 
        running_train_loss = 0.0
        y_true = []
        y_pred = []
    
        # for data in train_loader: 

        
        if args.pretrain == 'dual_combined':

            for i, data in enumerate(zip(train_loader[0], train_loader[1]), cycle(pair_loader)):  # ux, xy, uy
    
            #for data in enumerate(train_loader, 0): 
                rna, protein = data[0][0],  data[1][0] # get the input and real species as outputs; data is a list of [inputs, outputs] 
                rna, protein = rna.to(device), protein.to(device)
                

                prna, pprotein = data[3][0]
                print(prna.shape, pprotein.shape)


                optim_r2p.zero_grad()   # zero the parameter gradients
                optim_p2r.zero_grad()   # zero the parameter gradients   

                pro_pred = model_r2p(rna)
                rna_pred = model_p2r(pro_pred)
                rna_loss = loss_fn(rna_pred, rna)

                rna_pred = model_p2r(protein)
                pro_pred = model_r2p(rna_pred)
                pro_loss = loss_fn(pro_pred, protein)

                all_loss = rna_loss + pro_loss

                all_loss.backward()   # backpropagate the loss, here for ddp, the specific process will wait other processes finished and update the params
                
                optim_r2p.step()        # adjust parameters based on the calculated gradients 
                optim_p2r.step()        # adjust parameters based on the calculated gradients 

                running_train_loss +=all_loss.item()  # track the loss value 

                # print('train mse',  metric.mse_loss()   )

            # Calculate training loss value 
            train_loss_value = running_train_loss/len(train_loader) 

            writer.add_scalar('Train/loss', train_loss_value, epoch)

            # Validation Loop
            running_val_loss = 0.0

            with torch.no_grad():

                model_r2p.eval()   
                model_p2r.eval() 

                for i, data in enumerate(zip(train_loader[0], cycle(val_loader[1]))):
                    
                    rna, protein = data[0][0],  data[1][0]  # get the input and real species as outputs; data is a list of [inputs, outputs] 
                    rna, protein = rna.to(device), protein.to(device)
            
                    pro_pred = model_r2p(rna)
                    rna_pred = model_p2r(pro_pred)
                    rna_loss = loss_fn(rna_pred, rna)





                    rna_pred = model_p2r(protein)
                    pro_pred = model_r2p(rna_pred)
                    pro_loss = loss_fn(pro_pred, protein)

                    all_loss = rna_loss + pro_loss
                    
                    running_val_loss += all_loss.item()

            # Calculate validation loss value 
            val_loss_value = running_val_loss/len(val_loader) 

            writer.add_scalar('Val/loss', val_loss_value, epoch)
                

            if val_loss_value < best_val_loss_value:
                
            
                path = 'dual/' + str(args.dataset) + '_r2p.pth'
                torch.save(model_r2p.module.state_dict(), path)

                path = 'dual/' + str(args.dataset) + '_p2r.pth'
                torch.save(model_p2r.module.state_dict(), path)

                best_epoch = epoch  
                best_val_loss_value = val_loss_value
            
            logger.info('Epoch %d: Train Loss: %.4f, Val Loss: %.4f, Best epoch: %d, Best val loss: %.4f',
                        epoch, train_loss_value, val_loss_value, best_epoch, best_val_loss_value)
            


        # test Loop
            running_test_loss = 0.0
            y_true = []
            y_pred = []

            for data in test_loader: 
                inputs, outputs = data 
                inputs, outputs = inputs.to(device), outputs.to(device)

                predicted_outputs = model_r2p(inputs) 
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

            writer.add_scalar('Test/loss', test_loss_value, epoch)

            logger.info('Test Loss: %.4f, Test RMSE: %.4f, Test R2: %.4f, Test PCC: %.4f', test_loss_value, test_rmse, test_r2, test_pcc)




def test(test_loader, loss_fn, input_size, output_size):
    logger.info("\nBegin test!") 
    # Load the model that we saved at the end of the training loop 
    model = network.MLP(input_size, output_size) 
    path = 'dual/' + str(args.dataset) + "_r2p.pth" 
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

   
    train_loader, pair_loader, val_loader, test_loader, rna_dims, pro_dims = util.get_dataset(args)

    # Instantiate the model
    model_r2p = network.MLP(rna_dims, pro_dims)
    path = 'pretrain/' + str(args.dataset) + "_r2p.pth"  
    model_r2p.load_state_dict(torch.load(path))
    model_r2p.to(device)
    model_r2p = nn.DataParallel(model_r2p)

    model_p2r = network.MLP(pro_dims, rna_dims).to(device)
    path = 'pretrain/' + str(args.dataset) + '_p2r.pth'
    model_p2r.load_state_dict(torch.load(path))
    model_p2r.to(device)
    model_p2r = nn.DataParallel(model_p2r)

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    optim_r2p = Adam(model_r2p.parameters(), lr=0.001)
    optim_p2r = Adam(model_p2r.parameters(), lr=0.001)

    loss_fn = nn.MSELoss()
    
    train(args, train_loader, pair_loader, val_loader, test_loader, optim_r2p, optim_p2r, model_r2p, model_p2r, loss_fn)

    test(test_loader, loss_fn, rna_dims, pro_dims)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()    

    parser.add_argument('--dataset', default='cbmc', type=str)   
    parser.add_argument('--pca', action='store_true', default=True)  
    parser.add_argument('--pca_dim', required='--pca' in sys.argv, default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)  
    parser.add_argument('--epochs', default=500, type=int)  
    parser.add_argument('--pretrain', default = 'r2p', choices=['r2p','p2r', 'dual_unlabel', 'dual_combined'])

    args = parser.parse_args() 
    
    logging.basicConfig(level=logging.INFO, filename= 'dual_combine/' + str(args.dataset) + '_dual.log', filemode='w')
    logger = logging.getLogger('duallogger')

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter('dual/tensorboard/' + str(args.dataset))

    main(args)

    # python pre_train.py --pretrain r2p


# tensorboard --logdir=./path/to/the/folder --port 8123
# tensorboard --logdir=runs