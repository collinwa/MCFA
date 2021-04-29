import torch
import numpy as np
import matplotlib.pyplot as plt

# EXPERIMENT: Use isotropic noise variance that is the same across all of the y_i, allowing for the L to account for 
# different true variance and covariance conditioned on the y_i

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


def fit_model(y_dims, x_dims, datasets, d, y_concat, N, eps=1e-6, steps=5000):
    torch.set_num_threads(20)

    y_concat_T = y_concat.T

    # initialize the model parameters
    W_model, L_model, Phi_model = initialize_isotropic_model(y_dims, x_dims, datasets, d=d)

    print(W_model.shape)
    print(L_model.shape)
    print(Phi_model)

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


def project_latent(W, L, Phi, d, y_concat, x_dims):
    y_concat_T = y_concat.T
    sigma_22_inv = torch.inverse(W@W.T + L @ L.T + Phi*torch.eye(W.shape[0]))
    #     print(sigma_22_inv)

    # other necessary block matrices
    sigma_12 = torch.cat([W.T, L.T], axis=0)
    sigma_11 = torch.eye(torch.sum(x_dims)+d)

    # compute the posterior mean of z and x; y should be a matrix with all samples aligned as columns
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ y_concat_T
    posterior_z_mean = posterior_z_x_mean[:d]
    posterior_x_mean = posterior_z_x_mean[d:]

    return posterior_z_mean.T, posterior_x_mean.T