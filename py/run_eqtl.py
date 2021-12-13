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
    parser.add_argument('--isotropic', action='store_true', help='Use isotropic covariance model')
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
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # concatenate y's for each sample together to prepare to fit model 
    y_concat = torch.cat(datasets, axis=1).double()
    W_model, L_model, Phi_model = fit_model(y_dims, x_dims, datasets, d, y_concat, n_samples, steps=args.steps, device=device)
    W_model = W_model.cpu()
    L_model = L_model.cpu()
    Phi_model = Phi_model.cpu()

    # default setting for isotropic is False (i.e. diagonal covariance model with potentially non-iso variance)
    posterior_z, posterior_x = project_latent(W_model, L_model, Phi_model, d, y_concat, x_dims, isotropic=args.isotropic)

    # extract data-set specific projection matrices 
    pd.DataFrame(W_model[:y_dims[0], :].numpy()).to_csv(os.path.join(outdir, 'dataset1_W.csv'), header=False, index=False)
    pd.DataFrame(W_model[y_dims[0]:, :].numpy()).to_csv(os.path.join(outdir, 'dataset2_W.csv'), header=False, index=False)
    pd.DataFrame(L_model[:y_dims[0], :x_dims[0]].numpy()).to_csv(os.path.join(outdir, 'dataset1_L.csv'), header=False, index=False)
    pd.DataFrame(L_model[y_dims[0]:, x_dims[0]:].numpy()).to_csv(os.path.join(outdir, 'dataset2_L.csv'), header=False, index=False)
    pd.DataFrame(posterior_x[:, :x_dims[0]].numpy()).to_csv(os.path.join(outdir, 'dataset1_x.csv'), header=False, index=False)
    pd.DataFrame(posterior_x[:, x_dims[0]:].numpy()).to_csv(os.path.join(outdir, 'dataset2_x.csv'), header=False, index=False)
    pd.DataFrame(posterior_z.numpy()).to_csv(os.path.join(outdir, 'posterior_z.csv'), header=False, index=False)
