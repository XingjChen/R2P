import numpy as np
import torch 
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.decomposition import PCA
import os
import logging
logger = logging.getLogger('mainlogger.util')

def split_data(dataset):
    # print(os.getcwd())
    
    logger.info("Dataset: %s", dataset)

    ### 5991 training samples and 2561 test samples, while 20501 gene abundances mapping to 13 protein abundances.
    train_X = np.load('./data/' + str(dataset) + '/train_x.npy') # gene  
    train_y = np.load('./data/' + str(dataset) + '/train_y.npy') # protein

    # split 10% as paired data and 90% as unlabeled data
    label_ratio = 0.1 # the ratio of labelled data and unlabelled data
    sample_nums = train_X.shape[0]
    pair_X = train_X[ :int(label_ratio*sample_nums), :] # get 10% training data as pair data and other 90% as unlabelled data
    pair_y = train_y[ :int(label_ratio*sample_nums), :]
    # print(pair_X.shape, pair_y.shape) # (599, 20501) (599, 13)
    unlabel_X = train_X[int(label_ratio*sample_nums):, :]
    unlabel_y = train_y[int(label_ratio*sample_nums):, :]

    # for paired and unlabeled gene data, we use pca to reduce the dimension to 200
    all_X = np.vstack((pair_X, unlabel_X)) # (5182, 20501)  
    pca = PCA(n_components=200)
    pca.fit(all_X)   # here we fit for rna data, and together with matched rna and unlabelled rna, (5182, 38)  
    all_X = pca.transform(all_X)
    logger.info('The dimension after PCA is: %4d', all_X.shape[1])
    pair_X = all_X[ :pair_X.shape[0], :]
    unlabel_X = all_X[pair_X.shape[0]:, :]

    # use the same pca to fit test gene data
    test_X = np.load('./data/' + str(dataset) + '/test_x.npy')
    test_y = np.load('./data/' + str(dataset) + '/test_y.npy')
    test_X =  pca.transform(test_X)

    # devide the inlabeled data with the ratio of 0.85:0.15 while first 0.85 for gene and last 0.15 for protein
    unlabel_ratio = 0.85
    unlabel_nums = unlabel_X.shape[0]
    unlabel_X = unlabel_X[ :int(unlabel_ratio*unlabel_nums), :]
    unlabel_y = unlabel_y[int(unlabel_ratio*unlabel_nums):, :]

    # print(pair_X.shape, pair_y.shape, unlabel_X.shape, unlabel_y.shape, test_X.shape, test_y.shape)
    # (599, 200) (599, 13) (4583, 200) (809, 13) (2561, 200) (2561, 13)
    
    logger.info("There are %d pairred samples with %d RNAs (after PCA) and %d proteins for training.",\
    pair_X.shape[0], pair_X.shape[1], pair_y.shape[1])

    logger.info("There are %d unlabeled mRNAs amples and %d unlabeled protein samples for training.",\
     unlabel_X.shape[0], unlabel_y.shape[0])

    logger.info("There are %d pairred samples for testing.",test_X.shape[0])

    logger.info('\n')

    return pair_X, pair_y, unlabel_X, unlabel_y, test_X, test_y

def get_dataset(args):

    pair_X, pair_y, unlabel_X, unlabel_y, test_Xp, test_yp = split_data(args.dataset)  # (599, 20501) (599, 13)
    
    if args.pretrain == 'r2p':

        train_X = pair_X 
        train_y = pair_y

        test_X = test_Xp 
        test_y = test_yp

     # Convert Input and Output data to Tensors and create a TensorDataset 
        input_size = train_X.shape[1] # num of gene features
        output_size = train_y.shape[1]

        train_X, train_y = torch.Tensor(train_X), torch.Tensor(train_y)  # Double 64-bit while folat 32-bit, Tensor-> to 32, tensor-> auto

        # print((train_X.dtype), (train_y.dtype))     
        train_dataset = TensorDataset(train_X, train_y)
        split_ratio = [int(len(train_dataset) * 0.9), int(len(train_dataset) * 0.1) + 1] # set 10% training data as validation data
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, split_ratio)

        
        # Create Dataloader to read the data within batch sizes and put into memory. 
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True) 
        val_loader = DataLoader(val_dataset, batch_size = 32) 


        test_X, test_y = torch.Tensor(test_X), torch.Tensor(test_y)
        test_dataset = TensorDataset(test_X, test_y)  
        test_loader = DataLoader(test_dataset, batch_size = 32)

        return train_loader, val_loader, test_loader, input_size, output_size

    elif args.pretrain == 'p2r':

        train_X = pair_y 
        train_y = pair_X

        test_y = test_Xp 
        test_X = test_yp

         # Convert Input and Output data to Tensors and create a TensorDataset 
        input_size = train_X.shape[1] # num of gene features
        output_size = train_y.shape[1]

        train_X, train_y = torch.Tensor(train_X), torch.Tensor(train_y)  # Double 64-bit while folat 32-bit, Tensor-> to 32, tensor-> auto

        # print((train_X.dtype), (train_y.dtype))     
        train_dataset = TensorDataset(train_X, train_y)
        split_ratio = [int(len(train_dataset) * 0.9), int(len(train_dataset) * 0.1) + 1] # set 10% training data as validation data
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, split_ratio)

        
        # Create Dataloader to read the data within batch sizes and put into memory. 
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True) 
        val_loader = DataLoader(val_dataset, batch_size = 32) 

        test_X, test_y = torch.Tensor(test_X), torch.Tensor(test_y)
        test_dataset = TensorDataset(test_X, test_y)  
        test_loader = DataLoader(test_dataset, batch_size = 32)

        return train_loader, val_loader, test_loader, input_size, output_size


    elif args.pretrain == 'dual_unlabel':

        train_X = unlabel_X
        train_y = unlabel_y 

        test_X = test_Xp 
        test_y = test_yp

          # Convert Input and Output data to Tensors and create a TensorDataset 
        input_size = train_X.shape[1] # num of gene features
        output_size = train_y.shape[1]

        train_X, train_y = torch.Tensor(train_X), torch.Tensor(train_y)  # Double 64-bit while folat 32-bit, Tensor-> to 32, tensor-> auto

        # print((train_X.dtype), (train_y.dtype))     
        train_X_dataset = TensorDataset(train_X)
        train_y_dataset = TensorDataset(train_y)

        split_ratio = [int(len(train_X_dataset) * 0.9), int(len(train_X_dataset) * 0.1) + 1] # set 10% training data as validation data
       
        train_X_dataset, val_X_dataset = torch.utils.data.random_split(train_X_dataset, split_ratio, generator=torch.Generator().manual_seed(42))
        
        train_y_dataset, val_y_dataset = torch.utils.data.random_split(train_y_dataset, [int(len(train_y_dataset) * 0.9), int(len(train_y_dataset) * 0.1) + 1], generator=torch.Generator().manual_seed(42))
        
        # Create Dataloader to read the data within batch sizes and put into memory. 
        train_X_loader = DataLoader(train_X_dataset, batch_size = args.batch_size, shuffle = True) 
        train_y_loader = DataLoader(train_y_dataset, batch_size = args.batch_size, shuffle = True)

        val_X_loader = DataLoader(val_X_dataset, batch_size = 32) 
        val_y_loader = DataLoader(val_y_dataset, batch_size = 32) 

        test_X, test_y = torch.Tensor(test_X), torch.Tensor(test_y)
        test_dataset = TensorDataset(test_X, test_y)  
        test_loader = DataLoader(test_dataset, batch_size = 32)

        train_loader = [train_X_loader, train_y_loader]

        val_loader = [val_X_loader, val_y_loader]

        return train_loader, val_loader, test_loader, input_size, output_size


    elif args.pretrain == 'dual_combined':

        train_X = unlabel_X
        train_y = unlabel_y 

        pair_X = pair_X
        pair_y = pair_y 

        test_X = test_Xp 
        test_y = test_yp

          # Convert Input and Output data to Tensors and create a TensorDataset 
        input_size = train_X.shape[1] # num of gene features
        output_size = train_y.shape[1]

        train_X, train_y, pair_X, pair_y = torch.Tensor(train_X), torch.Tensor(train_y), torch.Tensor(pair_X), torch.Tensor(pair_y)  # Double 64-bit while folat 32-bit, Tensor-> to 32, tensor-> auto

        # print((train_X.dtype), (train_y.dtype))     
        train_X_dataset = TensorDataset(train_X)
        train_y_dataset = TensorDataset(train_y)

        pair_dataset = TensorDataset(pair_X, pair_y)

        split_ratio = [int(len(train_X_dataset) * 0.9), int(len(train_X_dataset) * 0.1) + 1] # set 10% training data as validation data
       
        train_X_dataset, val_X_dataset = torch.utils.data.random_split(train_X_dataset, split_ratio, generator=torch.Generator().manual_seed(42))
        
        train_y_dataset, val_y_dataset = torch.utils.data.random_split(train_y_dataset, [int(len(train_y_dataset) * 0.9), int(len(train_y_dataset) * 0.1) + 1], generator=torch.Generator().manual_seed(42))
        

        # Create Dataloader to read the data within batch sizes and put into memory. 
        train_X_loader = DataLoader(train_X_dataset, batch_size = args.batch_size, shuffle = True) 
        train_y_loader = DataLoader(train_y_dataset, batch_size = args.batch_size, shuffle = True)
        
        val_X_loader = DataLoader(val_X_dataset, batch_size = 32) 
        val_y_loader = DataLoader(val_y_dataset, batch_size = 32)
        
        pair_loader = DataLoader(pair_dataset, batch_size = args.batch_size, shuffle = True) 

        test_X, test_y = torch.Tensor(test_X), torch.Tensor(test_y)
        test_dataset = TensorDataset(test_X, test_y)  
        test_loader = DataLoader(test_dataset, batch_size = 32)

        train_loader = [train_X_loader, train_y_loader]

        val_loader = [val_X_loader, val_y_loader]

        return train_loader, pair_loader, val_loader, test_loader, input_size, output_size
