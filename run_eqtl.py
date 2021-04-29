import argparse
import torch
import os
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import fit_model, project_latent

def parse_args():
    parser = argparse.ArgumentParser(description='MPCCA for eQTL Pipeline')
    parser.add_argument('dataset_1', type=str, help='Matrix 1 csv file (no header) + samples aligned as rows')
    parser.add_argument('dataset_2', type=str, help='Matrix 2 csv file (no header) + samples aligned as rows')
    parser.add_argument('-d', type=int, default=5, help='Dimension of shared space')
    parser.add_argument('-x1', type=int, default=5, help='Dimension of private space for dataset 1')
    parser.add_argument('-x2', type=int, default=5, help='Dimension of private space for dataset 2')
    parser.add_argument('--steps', type=int, default=2000, help='Max # of E/M Steps to run')
    parser.add_argument('--outdir', type=str, default='./', help='Output directory to store .npy files')

    return parser.parse_args()

if __name__ == '__main__':
    # load command line args 
    args = parse_args()
    outdir = args.outdir

    # get model information from arguments 
    d = args.d
    x_dims = torch.tensor([args.x1, args.x2])

    # load datasets and prepare for model 
    dataset_1 = pd.read_csv(args.dataset_1).to_numpy()
    dataset_2 = pd.read_csv(args.dataset_2).to_numpy()
    dataset_1 = torch.tensor(dataset_1)
    dataset_2 = torch.tensor(dataset_2)
    datasets = [dataset_1, dataset_2]
    assert dataset_1.shape[0] == dataset_2.shape[0]
    print(dataset_1.shape)
    print(dataset_2.shape)

    # get y dimensions and number of samples
    y_dims = torch.tensor([dataset_1.shape[1], dataset_2.shape[1]])
    n_samples = dataset_1.shape[0]

    # concatenate y's for each sample together to prepare to fit model 
    y_concat = torch.cat(datasets, axis=1).float()
    W_model, L_model, Phi_model = fit_model(y_dims, x_dims, datasets, d, y_concat, n_samples, steps=args.steps) 
    posterior_z, posterior_x = project_latent(W_model, L_model, Phi_model, d, y_concat, x_dims)

    # extract data-set specific projection matrices 
    W_model_d1 = W_model[:y_dims[0], :].numpy()
    W_model_d2 = W_model[y_dims[0]:, :].numpy()
    L_model_d1 = L_model[:y_dims[0], :x_dims[0]].numpy()
    L_model_d2 = L_model[y_dims[0]:, x_dims[0]:].numpy()
    hidden_x_d1 = posterior_x[:, :x_dims[0]].numpy()
    hidden_x_d2 = posterior_x[:, x_dims[0]:].numpy()
    posterior_z = posterior_z.numpy()

    np.save(os.path.join(outdir, 'dataset1_W.npy'), W_model_d1)
    np.save(os.path.join(outdir, 'dataset2_W.npy'), W_model_d2)
    np.save(os.path.join(outdir, 'dataset1_L.npy'), L_model_d1)
    np.save(os.path.join(outdir, 'dataset2_L.npy'), L_model_d2)
    np.save(os.path.join(outdir, 'dataset1_x.npy'), hidden_x_d1)
    np.save(os.path.join(outdir, 'dataset2_x.npy'), hidden_x_d2)
    np.save(os.path.join(outdir, 'posterior_z.npy'), posterior_z)
