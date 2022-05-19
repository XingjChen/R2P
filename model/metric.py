# here the data format is m * n, while m is the cell/sample and n is the gene
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def mse_loss(output, target):

    if torch.is_tensor(output):
        loss = torch.nn.MSELoss()
        return loss(output, target)
    else:
        return mean_squared_error(target, output)

    # all_mse = torch.mean((output-target) ** 2, 0)
    # mse = torch.mean(all_mse)

def rmse_loss(output, target):

    if torch.is_tensor(output):
        loss = torch.nn.MSELoss()
        return torch.sqrt(loss(output, target))
    else:
        return mean_squared_error(target, output, squared = False)


def r2_loss(output, target):  # pred and ground truth

    if torch.is_tensor(output):
        target_mean = torch.mean(target, 0)
        ss_tot = torch.sum((target - target_mean) ** 2, 0)
        ss_res = torch.sum((target - output) ** 2, 0)
        r2 = 1 - ss_res / ss_tot
        return torch.mean(r2)
    else:
        return r2_score(target, output)


def pcc_loss(output, target):
    
    if torch.is_tensor(output):
        output_var = output - torch.mean(output, 0)
        target_var = target - torch.mean(target, 0)
        pcc = torch.sum(output_var * target_var, 0) / (torch.sqrt(torch.sum(output_var ** 2, 0)) * torch.sqrt(torch.sum(target_var ** 2, 0)))
        
        return torch.mean(pcc), torch.median(pcc), pcc

    else:
        all_pcc = []
        for i in range(output.shape[1]):
            pcc = pearsonr(output[:,i], target[:,i])[0]
            all_pcc.append(pcc)

        return np.mean(all_pcc), np.median(all_pcc), all_pcc
