import torch
import numpy as np
import matplotlib.pyplot as plt


def E_step(W, L, Phi, x_dims, d, y_i, device='cpu'):
    # schur-complement (M/D)^{-1}; need to make sure that this is not blowing up!
    sigma_22_inv = torch.inverse(W@W.T + L @ L.T + Phi).to(device)
    #     print(sigma_22_inv)

    # other necessary block matrices
    sigma_12 = torch.cat([W.T, L.T], axis=0).to(device)
    sigma_11 = torch.eye(torch.sum(x_dims)+d).to(device)

    # compute the posterior mean of z and x; y should be a matrix with all samples aligned as columns
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ (y_i)
    posterior_z_mean = posterior_z_x_mean[:d].to(device)
    posterior_x_mean = posterior_z_x_mean[d:].to(device)

    # posterior covariance
    posterior_x1_cov = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
    posterior_z_x_cov = posterior_x1_cov[:d, d:].to(device)  # cross covariance
    posterior_z_z_cov = posterior_x1_cov[:d, :d].to(device)  # upper left block matrix
    posterior_x_x_cov = posterior_x1_cov[d:, d:].to(device)  # bottom right block matrix
    
    # need to batch zmu and xmu: [n_samples, <[z, x]>.shape, 1]
    zmu_batched = posterior_z_mean.T[:, :, None].to(device)
    xmu_batched = posterior_x_mean.T[:, :, None].to(device)

    # posterior <zx.T> = cov(z, x) + <z><x.T>
    posterior_zxT = (posterior_z_x_cov + zmu_batched @ xmu_batched.permute(0, 2, 1)).to(device)  # shape: (n_samples, z_dim, x_dim)
    posterior_zzT = (posterior_z_z_cov + zmu_batched @ zmu_batched.permute(0, 2, 1)).to(device)  # shape: (n_samples, z_dim, z_dim)
    posterior_xxT = (posterior_x_x_cov + xmu_batched @ xmu_batched.permute(0, 2, 1)).to(device)  # shape: (n_samples, x_dim, x_dim)

    return posterior_zxT, posterior_zzT, posterior_xxT, zmu_batched, xmu_batched


def M_step(zxT, zzT, xxT, zmu, xmu, y_i, Phi_model, L_model, W_model, N, device='cpu'):
    # note that in the M-Step, we assume that the variance is diagonal but potentially
    # non-isotropic
    y_i_batched = y_i[:, :, None].to(device)  # (n_samples, batch_dim, 1)
    new_L = torch.sum(y_i_batched @ xmu.permute(0, 2, 1) - W_model @ zxT, axis=0) @ torch.inverse(torch.sum(xxT, axis=0)).to(device)
    new_W = torch.sum(y_i_batched @ zmu.permute(0, 2, 1) - L_model @ zxT.permute(0, 2, 1), axis=0) @ torch.inverse(torch.sum(zzT, axis=0)).to(device)
    new_Phi = (1 / N * torch.sum(y_i_batched @ y_i_batched.permute(0, 2, 1) + \
                                L_model @ xxT @ L_model.T + \
                                W_model @ zzT @ W_model.T + \
                                2 * L_model @ zxT.permute(0, 2, 1) @ W_model.T + \
                                -2 * y_i_batched @ zmu.permute(0, 2, 1) @ W_model.T + \
                                -2 * y_i_batched @ xmu.permute(0, 2, 1) @ L_model.T, axis=0)).to(device)

    new_Phi = torch.diag(torch.diagonal(new_Phi)).to(device)  # only update the diagonal components
    return new_W, new_L, new_Phi


def initialize_isotropic_model(y_dims, x_dims, datasets, d=5, std=2, mean=0):
    Ws_to_stack = []
    Phis_to_stack = []
    Ls_to_stack = []
    
    for i, y_dim in enumerate(y_dims):
        # set W and L to standard normal initialization around 0 
        cur_W = torch.nn.init.normal_(torch.zeros(y_dim, d), mean=mean, std=std)
        cur_L = torch.nn.init.normal_(torch.zeros(y_dim, x_dims[i]), mean=mean, std=std)

        # set Phi to the empirical covariance matrix 
        cur_dataset = datasets[i] # (n_samples x dimension)
        ymu = torch.mean(cur_dataset, axis=0, keepdim=True)
        cur_dataset = cur_dataset - ymu
        
        # due to design matrix construction, covariance = 1/n(Y^T Y)
        cur_Phi = (1/cur_dataset.shape[0]) * cur_dataset.T @ cur_dataset

        Ws_to_stack.append(cur_W)
        Ls_to_stack.append(cur_L)
        Phis_to_stack.append(cur_Phi)
    
    # get the current phi matrix
    cur_Phi = torch.block_diag(*Phis_to_stack)

    # we're fixing an *isotropic* variance assumption here, so the model variance will be captured by a *scalar*
    variance = 1/cur_Phi.shape[0] * torch.einsum('ii->', cur_Phi)
    assert variance >= 0 
    return torch.cat(Ws_to_stack, axis=0), torch.block_diag(*Ls_to_stack), variance


def initialize_model(y_dims, x_dims, datasets, d=5, std=2, mean=0):
    Ws_to_stack = []
    Phis_to_stack = []
    Ls_to_stack = []
    
    for i, y_dim in enumerate(y_dims):
        # set W and L to standard normal initialization around 0 
        cur_W = torch.nn.init.normal_(torch.zeros(y_dim, d), mean=mean, std=std)
        cur_L = torch.nn.init.normal_(torch.zeros(y_dim, x_dims[i]), mean=mean, std=std)

        # set Phi to the empirical covariance matrix 
        cur_dataset = datasets[i] # (n_samples x dimension)
        ymu = torch.mean(cur_dataset, axis=0, keepdim=True)
        demean_dataset = cur_dataset - ymu
        # print(demean_dataset.shape)
        # due to design matrix construction, covariance = 1/n(Y^T Y)
        cur_Phi = torch.diag(torch.diagonal((1/demean_dataset.shape[0]) * demean_dataset.T @ demean_dataset))

        Ws_to_stack.append(cur_W)
        Ls_to_stack.append(cur_L)
        Phis_to_stack.append(cur_Phi)

    return torch.cat(Ws_to_stack, axis=0).double(), torch.block_diag(*Ls_to_stack).double(), torch.block_diag(*Phis_to_stack).double()


# the E-Step required values (is there a way to batch this intelligently?)
def isotropic_E_step(W, L, Phi, x_dims, d, y_i):
    # schur-complement (M/D)^{-1}; need to make sure that this is not blowing up!
    # Phi is a scalar now! (Phi = \sigma^2)
    sigma_22_inv = torch.inverse(W@W.T + L @ L.T + Phi*torch.eye(W.shape[0]))
    #     print(sigma_22_inv)

    # other necessary block matrices
    sigma_12 = torch.cat([W.T, L.T], axis=0)
    sigma_11 = torch.eye(torch.sum(x_dims)+d)

    # compute the posterior mean of z and x; y should be a matrix with all samples aligned as columns
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ (y_i)
    posterior_z_mean = posterior_z_x_mean[:d]
    posterior_x_mean = posterior_z_x_mean[d:]

    # posterior covariance
    posterior_x1_cov = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
    posterior_z_x_cov = posterior_x1_cov[:d, d:]  # cross covariance
    posterior_z_z_cov = posterior_x1_cov[:d, :d]  # upper left block matrix
    posterior_x_x_cov = posterior_x1_cov[d:, d:]  # bottom right block matrix
    
    # need to batch zmu and xmu: [n_samples, <[z, x]>.shape, 1]
    zmu_batched = posterior_z_mean.T[:, :, None]
    xmu_batched = posterior_x_mean.T[:, :, None]

    # posterior <zx.T> = cov(z, x) + <z><x.T>
    posterior_zxT = posterior_z_x_cov + zmu_batched @ xmu_batched.permute(0, 2, 1)  # shape: (n_samples, z_dim, x_dim)
    posterior_zzT = posterior_z_z_cov + zmu_batched @ zmu_batched.permute(0, 2, 1)  # shape: (n_samples, z_dim, z_dim)
    posterior_xxT = posterior_x_x_cov + xmu_batched @ xmu_batched.permute(0, 2, 1)  # shape: (n_samples, x_dim, x_dim)

    return posterior_zxT, posterior_zzT, posterior_xxT, zmu_batched, xmu_batched


def isotropic_M_step(zxT, zzT, xxT, zmu, xmu, y_i, Phi_model, L_model, W_model, N):
    y_i_batched = y_i[:, :, None]  # (n_samples, batch_dim, 1)
    new_L = torch.sum(y_i_batched @ xmu.permute(0, 2, 1) - W_model @ zxT, axis=0) @ torch.inverse(torch.sum(xxT, axis=0))
    new_W = torch.sum(y_i_batched @ zmu.permute(0, 2, 1) - L_model @ zxT.permute(0, 2, 1), axis=0) @ torch.inverse(torch.sum(zzT, axis=0))

    # compute terms involving y_i_batched 
    var_terms = []
    var_terms.append(1/2*torch.sum(y_i_batched.permute(0, 2, 1) @ y_i_batched))
    var_terms.append(-1 * torch.sum(y_i_batched.permute(0, 2, 1) @ W_model @ zmu))
    var_terms.append(-1 * torch.sum(y_i_batched.permute(0, 2, 1) @ L_model @ xmu))

    # einsum does a batched trace
    var_terms.append(1/2 * torch.sum(torch.einsum('bii->b', W_model.T @ W_model @ zzT)))
    var_terms.append(1/2 * torch.sum(torch.einsum('bii->b', L_model.T @ L_model @ xxT)))
    var_terms.append(1/2 * torch.sum(torch.einsum('bii->b', L_model.T @ W_model @ zxT)))

    new_Phi =  torch.sum(2 / (N * y_i.shape[1]) * torch.tensor(var_terms)) 
    return new_W, new_L, new_Phi


def fit_isotropic_model(y_dims, x_dims, datasets, d, y_concat, N, eps=1e-6, steps=5000):
    torch.set_num_threads(20)

    y_concat_T = y_concat.T

    # initialize the model parameters
    W_model, L_model, Phi_model = initialize_isotropic_model(y_dims, x_dims, datasets, d=d)

    #print(W_model.shape)
    #print(L_model.shape)
    #print(Phi_model)

    # store the update size 
    W_diffs= []
    L_diffs = []
    Phi_diffs = []

    # iterate through E/M Steps
    for i in range(steps):
        # E-Step, then M-Step
        zxT, zzT, xxT, zmu, xmu = isotropic_E_step(W_model, L_model, Phi_model, x_dims, d, y_concat_T)
        W_tprime, L_tprime, Phi_tprime = isotropic_M_step(zxT, zzT, xxT, zmu, xmu, y_concat, Phi_model, L_model, W_model, N)

        # compute updated L 
        L_tupdate = torch.zeros_like(L_tprime)

        # careful when updating L_model; have to make sure to keep terms that allow
        # interaction of private structure across datasets fixed at 0 
        #     L_model = L_tprime
        for j in range(len(y_dims)):
            # only update the specific L_i corresponding to each dataset; keep 
            # zero-padded values that make the matrix multiplication work as 0
            bot_y = 0 if j == 0 else torch.sum(y_dims[:j])
            bot_x = 0 if j == 0 else torch.sum(x_dims[:j])
            L_tupdate[bot_y:bot_y+y_dims[j], bot_x:bot_x+x_dims[j]] = L_tprime[bot_y:bot_y+y_dims[j], bot_x:bot_x+x_dims[j]]

        # F-norm between prev params and updated
        W_diff = torch.norm(W_tprime-W_model).item() 
        L_diff = torch.norm(L_tupdate-L_model).item()
        Phi_diff = torch.norm(torch.tensor(Phi_tprime - Phi_model)).item()

        # store training stats
        W_diffs.append(W_diff)
        L_diffs.append(L_diff)
        Phi_diffs.append(Phi_diff)
        
        # update parameters
        W_model = W_tprime
        Phi_model = Phi_tprime
        L_model = L_tupdate

        # check for convergence
        if W_diff <= eps and L_diff <= eps and Phi_diff <= eps:
            print('Stopping... step between W_t, W_t+1 <= {}'.format(eps))
            break

        # debugging
        if (i % 100 == 0):
            print("{}/{}: (Wtprime-Wt)_F: {} (Ltprime-Lt)_F: {} (Phi_tprime-Phi_t)_F: {}".format(i, steps, W_diff, L_diff, Phi_diff))

    # Save Training Stats 
    n_steps = np.arange(len(W_diffs)) 
    plt.plot(n_steps, W_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(W_t - W_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./mesa_W_updates.png')
    plt.clf()

    n_steps = np.arange(len(L_diffs)) 
    plt.plot(n_steps, L_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(L_t - L_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./mesa_L_updates.png')
    plt.clf()

    n_steps = np.arange(len(Phi_diffs)) 
    plt.plot(n_steps, Phi_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(Phi_t - Phi_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./mesa_Phi_updates.png')
    plt.clf()

    return W_model, L_model, Phi_model


def project_latent(W, L, Phi, d, y_concat, x_dims, isotropic=False):
    y_concat_T = y_concat.T
    var_term = Phi*torch.eye(W.shape[0]) if isotropic else Phi
    sigma_22_inv = torch.inverse(W@W.T + L @ L.T + var_term)
    #     print(sigma_22_inv)

    # other necessary block matrices
    sigma_12 = torch.cat([W.T, L.T], axis=0)
    sigma_11 = torch.eye(torch.sum(x_dims)+d)

    # compute the posterior mean of z and x; y should be a matrix with all samples aligned as columns
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ y_concat_T
    posterior_z_mean = posterior_z_x_mean[:d]
    posterior_x_mean = posterior_z_x_mean[d:]

    return posterior_z_mean.T, posterior_x_mean.T


def project_latent_individual(W, L, Phi, d, y_concat, x_dims, y_dims, nth_dataset, isotropic=False):
    # get dimension of dataset of interest
    prev_dims = torch.sum(y_dims[:nth_dataset]).item()
    cur_dim = y_dims[nth_dataset]
    cur_W = W[prev_dims:prev_dims+cur_dim, :]
    cur_L = L[prev_dims:prev_dims+cur_dim, :]
    cur_Phi = Phi
    
    var_term = cur_Phi * torch.eye(cur_W.shape[0]) if isotropic else cur_Phi[prev_dims:prev_dims+cur_dim, prev_dims:prev_dims+cur_dim]
    # transpose dataset for easier manipulation
    y_concat_T = y_concat[:, prev_dims:prev_dims+cur_dim].T
    sigma_22_inv = torch.inverse(cur_W@cur_W.T + cur_L @ cur_L.T + var_term)

    # other necessary block matrices
    sigma_12 = torch.cat([cur_W.T, cur_L.T], axis=0)
    sigma_11 = torch.eye(y_dims[nth_dataset]+d)

    # compute the posterior mean of z and x; y should be a matrix with all samples aligned as columns
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ y_concat_T
    posterior_z_mean = posterior_z_x_mean[:d]
    posterior_x_mean = posterior_z_x_mean[d:]

    # slice x to contain only the inferred private space of the specified dataset
    prev_x_dims = torch.sum(x_dims[:nth_dataset]).item()
    cur_x_dim = x_dims[nth_dataset]
    posterior_x_mean = posterior_x_mean[prev_x_dims:prev_x_dims+cur_x_dim, :]

    return posterior_z_mean.T, posterior_x_mean.T


def compute_ISC(*all_ys):
    # setup for processing and computation of ISC
    datasets = [*all_ys]
    N_sets = len(datasets)

    # de-mean each individual CCA projection
    for i in range(N_sets):
        datasets[i] = datasets[i] - torch.mean(datasets[i], axis=1, keepdim=True)

    # ISC explained by each canonical component
    rb = torch.zeros(datasets[0].shape[1])    
    rw = torch.zeros(datasets[0].shape[1])
    # compute the b/w set correlation 
    for i in range(N_sets):
        for j in range(i+1, N_sets):
            rb += torch.sum(torch.mul(datasets[i], datasets[j]), axis=0)
        rw += torch.sum(torch.pow(datasets[i], 2), axis=0)
    rho = rb / rw
    return rho, rb, rw


def fit_model(y_dims, x_dims, datasets, d, y_concat, N, eps=1e-6, steps=5000, device='cpu'):
    torch.set_num_threads(20)
    device = torch.device(device)

    y_concat_T = y_concat.T.to(device)

    # initialize the model parameters
    W_model, L_model, Phi_model = initialize_model(y_dims, x_dims, datasets, d=d)

    # move parameters to device 
    W_model = W_model.to(device)
    L_model = L_model.to(device)
    Phi_model = Phi_model.to(device)
    y_concat = y_concat.to(device)
    y_concat_T = y_concat_T.to(device)

    #print(W_model.shape)
    #print(L_model.shape)
    #print(Phi_model)

    # store the update size 
    W_diffs= []
    L_diffs = []
    Phi_diffs = []

    # iterate through E/M Steps
    for i in range(steps):
        # E-Step, then M-Step
        zxT, zzT, xxT, zmu, xmu = E_step(W_model, L_model, Phi_model, x_dims, d, y_concat_T, device=device)
        W_tprime, L_tprime, Phi_tprime = M_step(zxT, zzT, xxT, zmu, xmu, y_concat, Phi_model, L_model, W_model, N, device=device)

        # compute updated L 
        L_tupdate = torch.zeros_like(L_tprime).to(device)

        # careful when updating L_model; have to make sure to keep terms that allow
        # interaction of private structure across datasets fixed at 0 
        #     L_model = L_tprime
        for j in range(len(y_dims)):
            # only update the specific L_i corresponding to each dataset; keep 
            # zero-padded values that make the matrix multiplication work as 0
            bot_y = 0 if j == 0 else torch.sum(y_dims[:j])
            bot_x = 0 if j == 0 else torch.sum(x_dims[:j])
            L_tupdate[bot_y:bot_y+y_dims[j], bot_x:bot_x+x_dims[j]] = L_tprime[bot_y:bot_y+y_dims[j], bot_x:bot_x+x_dims[j]]

        # F-norm between prev params and updated
        W_diff = torch.norm(W_tprime-W_model).item() 
        L_diff = torch.norm(L_tupdate-L_model).item()
        Phi_diff = torch.norm(Phi_tprime - Phi_model)

        # store training stats
        W_diffs.append(W_diff)
        L_diffs.append(L_diff)
        Phi_diffs.append(Phi_diff)
        
        # update parameters
        W_model = W_tprime.to(device)
        Phi_model = Phi_tprime.to(device)
        L_model = L_tupdate.to(device)

        # check for convergence
        if W_diff <= eps and L_diff <= eps and Phi_diff <= eps:
            print('Stopping... step between W_t, W_t+1 <= {}'.format(eps))
            break

        # debugging
        if (i % 100 == 0):
            print("{}/{}: (Wtprime-Wt)_F: {} (Ltprime-Lt)_F: {} (Phi_tprime-Phi_t)_F: {}".format(i, steps, W_diff, L_diff, Phi_diff))

    # Save Training Stats 
    n_steps = np.arange(len(W_diffs)) 
    plt.plot(n_steps, W_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(W_t - W_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./W_updates.png')
    plt.clf()

    n_steps = np.arange(len(L_diffs)) 
    plt.plot(n_steps, L_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(L_t - L_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./L_updates.png')
    plt.clf()

    n_steps = np.arange(len(Phi_diffs)) 
    plt.plot(n_steps, Phi_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(Phi_t - Phi_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./Phi_updates.png')
    plt.clf()

    return W_model, L_model, Phi_model


def compute_x_cov(posterior_x_mu):
    # compute empirical covariance of predicted private space
    # posterior_x_mu must be of dimension (n_samples x private_x_dim)
    # returns: 
    #         posterior_cov: posterior covariance matrix 
    #         torch.diagonal(^): diagonal view of posterior cov
    posterior_cov = posterior_x_mu.T @ posterior_x_mu

    return posterior_cov, torch.diagonal(posterior_cov)
