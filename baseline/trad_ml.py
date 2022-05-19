import numpy as np
import logging

from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import sys
sys.path.append('../')
from model import metric
from sklearn.decomposition import PCA
import argparse   

def get_dataset(file_name):

    train_X = np.load('../data/' + str(file_name) + '/train_x.npy')
    train_y = np.load('../data/' + str(file_name) + '/train_y.npy')
    test_X = np.load('../data/' + str(file_name) + '/test_x.npy')
    test_y = np.load('../data/' + str(file_name) + '/test_y.npy')
    logging.info("Dataset: %s", file_name)
    logging.info("There are %d training samples and %d test samples with %d RNAs and %d proteins.",\
    train_X.shape[0], test_X.shape[0], train_X.shape[1], train_y.shape[1])
    
    return train_X, train_y, test_X, test_y

def evaluate(args):

    logging.basicConfig(level=logging.INFO, filename='./' +args.dataset + '_baseline.log', filemode='w')
    ### choose the dataset
    train_X, train_y, test_X, test_y = get_dataset(args.dataset)

    # we use pca to reduce the dimension to 200
    if args.pca:
        pca = PCA(n_components=args.pca_dim)
        pca.fit(train_X) # here we fit for dna data, and together with matched dna and unlabelled dna, (5182, 38)  
        train_X = pca.transform(train_X)
        logging.info('The dimension after PCA is: %d', train_X.shape[1])
        # use the same pca to fit test gene data
        test_X =  pca.transform(test_X)


    logging.info('\n')

    en = ElasticNet(random_state=42, max_iter=1000)
    en.fit(train_X, train_y)
    en_y_pred = en.predict(test_X)

    en_test_mse = metric.mse_loss(en_y_pred, test_y) 
    en_test_rmse = metric.rmse_loss(en_y_pred, test_y) 
    en_test_r2 = metric.r2_loss(en_y_pred, test_y) 
    en_test_pcc = metric.pcc_loss(en_y_pred, test_y)[0]

    logging.info('EN: Test Loss: %.4f, Test RMSE: %.4f, Test R2: %.4f, Test PCC: %.4f', en_test_mse, en_test_rmse, en_test_r2, en_test_pcc)
    logging.info('\n')

    knn = KNeighborsRegressor()
    knn.fit(train_X, train_y)
    knn_y_pred = knn.predict(test_X)

    knn_test_mse = metric.mse_loss(knn_y_pred, test_y) 
    knn_test_rmse = metric.rmse_loss(knn_y_pred, test_y) 
    knn_test_r2 = metric.r2_loss(knn_y_pred, test_y) 
    knn_test_pcc = metric.pcc_loss(knn_y_pred, test_y)[0]

    logging.info('KNN: Test Loss: %.4f, Test RMSE: %.4f, Test R2: %.4f, Test PCC: %.4f', knn_test_mse, knn_test_rmse, knn_test_r2, knn_test_pcc)
    logging.info('\n')

    rf = RandomForestRegressor(random_state=42, n_estimators = 1000)
    rf.fit(train_X, train_y)
    rf_y_pred = rf.predict(test_X)

    rf_test_mse = metric.mse_loss(rf_y_pred, test_y) 
    rf_test_rmse = metric.rmse_loss(rf_y_pred, test_y) 
    rf_test_r2 = metric.r2_loss(rf_y_pred, test_y) 
    rf_test_pcc = metric.pcc_loss(rf_y_pred, test_y)[0]

    logging.info('RF: Test Loss: %.4f, Test RMSE: %.4f, Test R2: %.4f, Test PCC: %.4f', rf_test_mse, rf_test_rmse, rf_test_r2, rf_test_pcc)
    logging.info('\n')

    xgb = XGBRegressor(random_state=42, n_estimators = 1000)
    xgb.fit(train_X, train_y)
    xgb_y_pred = xgb.predict(test_X)

    xgb_test_mse = metric.mse_loss(xgb_y_pred, test_y) 
    xgb_test_rmse = metric.rmse_loss(xgb_y_pred, test_y) 
    xgb_test_r2 = metric.r2_loss(xgb_y_pred, test_y) 
    xgb_test_pcc = metric.pcc_loss(xgb_y_pred, test_y)[0]

    logging.info('XGB: Test Loss: %.4f, Test RMSE: %.4f, Test R2: %.4f, Test PCC: %.4f', xgb_test_mse, xgb_test_rmse, xgb_test_r2, xgb_test_pcc)



def main():
    
    parser = argparse.ArgumentParser()    

    parser.add_argument('--dataset', default='cbmc', type=str)   
    parser.add_argument('--pca', action='store_true', default=True)  
    parser.add_argument('--pca_dim', required='--pca' in sys.argv, default=200, type=int)

    args = parser.parse_args() 

    evaluate(args)
    
if __name__ == '__main__':
    main()


# python trad_ml.py --dataset cbmc --pca_dim 200
# python trad_ml.py --dataset sln --pca_dim 200