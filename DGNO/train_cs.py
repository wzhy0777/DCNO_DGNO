# Import standard libraries
import os  # Miscellaneous operating system interfaces
import time  # Time access and conversions
import warnings  # Warning control
import argparse
import scipy.io as io

import numpy as np  # NumPy library for numerical operations
import torch
# from torch.autograd import grad
from functorch import vmap, grad

# Import custom modules
from deeponet import *  # Physics Informed Neural Networks utilities
import pandas as pd
import torch.nn as nn
from scipy.interpolate import griddata
from scipy.io import loadmat
import csv
# from torch.autograd import grad
from generate_branch_inputs_train import *
from generate_branch_inputs_test import *

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--d', type=int, default=4, help='depth')
parser.add_argument('--n', type=int, default=70, help='width')
parser.add_argument('--k', type=int, default=90, help='width of the output layer')
parser.add_argument('--nx', type=int, default=64, help='Sampling')
args = parser.parse_args()

# ---------- begin: greedy basis training & projection utilities ----------




def generate_branch_inputs(params, Xq, Yq, K=1, L=1):

    X_grid, Y_grid = Xq, Yq
    n = X_grid.shape[0]
    points = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)
    sensor_x = np.linspace(0, params["Lx"], params["num_sensors"])
    sensor_y = np.linspace(0, params["Ly"], params["num_sensors"])
    sensor_X, sensor_Y = np.meshgrid(sensor_x, sensor_y)
    sensor_points = np.stack([sensor_X.flatten(), sensor_Y.flatten()], axis=-1)
    branch_inputs = []
    f_values = []
    coeffs_all = []

    for _ in range(params["num_f"]):
        coeffs = np.random.uniform(6, 8, size=(K, L)) / (
            np.arange(1, K + 1)[:, None] ** 2 + np.arange(1, L + 1)[None, :]
        )

        f_grid = np.zeros((n, n))
        for k in range(1, K + 1):
            for l in range(1, L + 1):
                f_grid += (
                    coeffs[k - 1, l - 1]
                    * (k ** 2 + l ** 2)
                    * np.pi ** 2
                    * np.sin(k * np.pi * X_grid)
                    * np.sin(l * np.pi * Y_grid)
                )

        values = f_grid.flatten()
        branch_input = griddata(points, values, sensor_points, method="cubic")
        branch_inputs.append(branch_input)
        f_values.append(values)
        coeffs_all.append(coeffs.flatten()) 

    return np.array(f_values),np.array(branch_inputs),np.array(coeffs_all)  







def eval_bases_and_grads(basis_models, coeffs_tensor, trunk_input_pts):
    device = coeffs_tensor.device
    B = coeffs_tensor.shape[0]
    N = trunk_input_pts.shape[0]
    M = len(basis_models)
    assert trunk_input_pts.requires_grad, "trunk_input_pts must have requires_grad=True"

    psi_vals = []
    grad_vals = []

    for net in basis_models:
        # ------ psi values: (B, N) ------
        def get_psi(coeff):
            return net(coeff.unsqueeze(0), trunk_input_pts).squeeze(0)  # (N,)
        psi_b = vmap(get_psi)(coeffs_tensor)  # (B, N)
        psi_vals.append(psi_b)

        def get_grad(coeff):
            def psi_fn(x):
                return net(coeff.unsqueeze(0), x).squeeze(0)  # (N,)
            grad_output = grad(lambda x: psi_fn(x).sum(), has_aux=False)(trunk_input_pts)  # (N, 2)
            return grad_output
        grad_b = vmap(get_grad)(coeffs_tensor)  # (B, N, 2)
        grad_vals.append(grad_b)

    psi_vals = torch.stack(psi_vals, dim=1)  # (B, M, N)
    grad_vals = torch.stack(grad_vals, dim=1)  # (B, M, N, 2)

    return psi_vals, grad_vals








def solve_coeffs_for_batch(psi_vals, grad_vals, f_vals, gauss_weights, eps_reg=1e-12):

    B, M, N = psi_vals.shape
    betas = torch.zeros(B, M, device=psi_vals.device)
    weights = gauss_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    
    for b in range(B):
        ψ = psi_vals[b]      # (M, N)
        grad_ψ = grad_vals[b]    # (M, N, 2)
        f = f_vals[b]        # (N,)
        A = torch.einsum('ind,jnd,n->ij', grad_ψ, grad_ψ, gauss_weights)
        A = A + eps_reg * torch.eye(M, device=ψ.device)
        F = torch.einsum('n,jn,n->j', f, ψ, gauss_weights)
        β = torch.linalg.solve(A, F.unsqueeze(-1)).squeeze(-1)
        betas[b] = β
    
    return betas



def greedy_train_bases_and_project(initial_bases, coeffs_train, F_train, Trunk_input, Wq, coeffs_test, F_values_test, Trunk_input_test, Wq_test, params, 
                                   num_total_bases=6, epochs_each=300, lr=5e-4, reg=1e-6,
                                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), save_path=None):


    Coeffs_train = torch.tensor(coeffs_train, dtype=torch.float32, device=device)  # (B, branch_dim)
    Coeffs_test = torch.tensor(coeffs_test, dtype=torch.float32, device=device) 
    basis_models = list(initial_bases)  # copy
    M0 = len(basis_models)
    
    
    psi_vals_test, grad_vals_test = eval_bases_and_grads(basis_models, Coeffs_test, Trunk_input_test)
    alphas_test = solve_coeffs_for_batch(psi_vals_test, grad_vals_test, F_values_test, Wq_test)
    u_test_pred = torch.einsum('bm,bmn->bn', alphas_test, psi_vals_test)
    Total_rel_l2_test = torch.sqrt(torch.sum((u_test_pred + torch.tensor(Phi1, device=device) - torch.tensor(Phi_total_exact, device=device))**2, dim=1)) / \
                  torch.sqrt(torch.sum(torch.tensor(Phi_total_exact, device=device)**2, dim=1))
    Total_mean_val = Total_rel_l2_test.mean()
    print("Total_mean rel_l2_test:", Total_mean_val.item())
    mean_values = []
    

    for j in range(M0, num_total_bases):
        print(f"\n=== Greedy training for basis {j+1}/{num_total_bases} ===")
        psi_vals, grad_vals = eval_bases_and_grads(basis_models, Coeffs_train, Trunk_input)  # (B, M, N)
        alphas = solve_coeffs_for_batch(psi_vals, grad_vals, F_train, Wq)  # (B,M)
        u_vals = torch.einsum('bm,bmn->bn', alphas, psi_vals)
        grad_u_prev = torch.einsum('bm,bmnd->bnd', alphas, grad_vals).detach()


        branch = MLP(5, params["k"]+20*j, params["width"]+20*j, params["depth"]).to(device)
        branch.apply(init_weights)
        trunk = MLP(2, params["k"]+20*j, params["width"]+20*j, params["depth"]).to(device)
        trunk.apply(init_weights)
        psi_new = DeepONet(branch, trunk).to(device)


        opt = torch.optim.Adam(psi_new.parameters(), lr=lr, weight_decay=1e-9)

        for ep in range(1, epochs_each + 1): 
            loss = loss_function(psi_new, Coeffs_train,  F_train,  basis_models, grad_u_prev, Trunk_input, Wq, params, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ep % 1000 == 0 or ep == 1:
                print(f"  basis {j+1} ep {ep}/{epochs_each} loss {loss.item():.4e}")


        if save_path:
            model_path = f"{save_path}/basis_{j}.pth"
            torch.save(psi_new, model_path)  
            print(f"Saved complete model for basis {j} to {model_path}")
            

        basis_models.append(psi_new)
        psi_vals_all, grad_vals_all = eval_bases_and_grads(basis_models, Coeffs_train, Trunk_input)
        alphas_all =  solve_coeffs_for_batch(psi_vals_all, grad_vals_all, F_train, Wq)

        
        # ------------------------------------
        psi_vals_test, grad_vals_test = eval_bases_and_grads(basis_models, Coeffs_test, Trunk_input_test)
        alphas_test = solve_coeffs_for_batch(psi_vals_test, grad_vals_test, F_values_test, Wq_test)
        u_test_pred = torch.einsum('bm,bmn->bn', alphas_test, psi_vals_test)  # (3, N)
        Total_rel_l2_test = torch.sqrt(torch.sum((u_test_pred + torch.tensor(Phi1, device=device) - torch.tensor(Phi_total_exact, device=device))**2, dim=1)) / \
                      torch.sqrt(torch.sum(torch.tensor(Phi_total_exact, device=device)**2, dim=1))
        Total_mean_val = Total_rel_l2_test.mean()
        print("Total_mean rel_l2_test:", Total_mean_val.item())


    
    with open('IO_DeepONet_training_d{}_n{}_nx{}_k{}/mean rel_l2_test.csv'.format(args.d, args.n,
                                                                                           args.nx,
                                                                                           args.k),
              'w') as file:
        writer = csv.writer(file)
        writer.writerows([[value] for value in mean_values])
        
    psi_vals_final = psi_vals_all
    alphas_final = alphas_all  # (B,M)
    u_pred_final = torch.einsum('bm,bmn->bn', alphas_final, psi_vals_final)  # (B,N)

    return basis_models, alphas_final.detach().cpu().numpy(), u_pred_final.detach().cpu().numpy()





def errorFun(output, target):
    error = output - target
    error = np.sqrt(np.mean(error * error))
    ref = np.sqrt(np.mean(target * target))
    return error / (ref + 1E-14)






def loss_function(deeponet, coeffs_all, f_values, basis_models, grad_u_prev, trunk_pts, Wq, params, device,
                  eps_reg=1e-8):

    B = coeffs_all.shape[0]
    device = trunk_pts.device
    quad_weights = Wq  # (nq*nq,)
    psi_new, grad_new = eval_bases_and_grads([deeponet], coeffs_all, trunk_pts)
    psi_new = psi_new[:, 0, :]          # (B, nq*nq)
    grad_new = grad_new[:, 0, :, :]     # (B, nq*nq, 2)
    
    psi_norm = torch.sqrt(torch.sum(psi_new**2 * quad_weights[None, :], dim=1) + 1e-12)  # (B,)
    psi_new = psi_new / psi_norm[:, None]
    grad_new = grad_new / psi_norm[:, None, None]
    
    inner_f_psi = torch.sum(f_values * psi_new * quad_weights[None, :], dim=1)
    inner_grad_u_grad_psi = torch.sum(
        torch.sum(grad_u_prev * grad_new, dim=-1) * quad_weights[None, :],
        dim=1
    )
    numerator = inner_f_psi - inner_grad_u_grad_psi
    psi_norm = torch.sum(
        torch.sum(grad_new**2, dim=-1) * quad_weights[None, :],
        dim=1
    ) 
    return -(numerator / psi_norm).mean()






def train_adam(deeponet, coeffs_all, f_values, coeffs_test, f_values_test, u_test, Phi1, Phi_total_exact, Wq, Wq_test, Trunk_input, Trunk_input_test, params, device, num_iter=100000):
     
    Branch_input = torch.from_numpy(coeffs_all).float().to(device)
    Branch_input_test = torch.from_numpy(coeffs_test).float().to(device)
    F_values = torch.from_numpy(f_values).float().to(device)
    F_values_test = torch.from_numpy(f_values_test).float().to(device)
    basis_models = []

    optimizer = torch.optim.Adam(deeponet.parameters(), lr=0.01)
    grad_u_prev = torch.zeros(params["num_f"], Trunk_input.shape[0], 2, device=device)
    global iter

    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        loss = loss_function(deeponet, Branch_input, F_values,  basis_models, grad_u_prev, Trunk_input, Wq, params, device) 
        loss.backward()
        optimizer.step()
        results.append([iter, loss.item()])  # , error
        iter += 1
        

    
    psi_prev_test, grad_prev_test = eval_bases_and_grads([deeponet], Branch_input_test, Trunk_input_test) # (B, M, N), (B, M, N, 2)
    alphas_test = solve_coeffs_for_batch(psi_prev_test, grad_prev_test, F_values_test, Wq_test) # (B, M) 
    U_pred_last = alphas_test[:, None] * psi_prev_test # (B, N)
    U_pred_last = U_pred_last.squeeze(1)
    

    Total_rel_l2_test = torch.sqrt(torch.sum((U_pred_last + torch.tensor(Phi1, device=device) - torch.tensor(Phi_total_exact, device=device))**2, dim=1)) / \
                  torch.sqrt(torch.sum(torch.tensor(Phi_total_exact, device=device)**2, dim=1))
    Total_mean_val = Total_rel_l2_test.mean()
    print("Total_mean rel_l2_test:", Total_mean_val.item())
    


if __name__ == "__main__":
    set_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if not os.path.exists(
            'IO_DeepONet_training_d{}_n{}_nx{}_k{}'.format(args.d, args.n, args.nx, args.k)):
        os.makedirs(
            'IO_DeepONet_training_d{}_n{}_nx{}_k{}'.format(args.d, args.n, args.nx, args.k))
    # Initialize variables
    results = []
    iter = 0 
    params = dict()
    params["k"] = args.k
    params["width"] = args.n
    params["depth"] = args.d
    params["Lx"] = 12
    params["Ly"] = 12
    params["nx"] = args.nx
    params["ny"] = args.nx
    params["num_sensors"] = 64
    params["num_f"] = 100
    params["batch_size"] = 4096
    params["nq"] = 150
    params["nq_test"] = 191
    
    Xq, Yq, wq, f_values, coeffs_all, u_star = generate_branch_inputs_train(params["Lx"],params["nq"],params["num_f"])
    gauss_weights_flat = wq.reshape(-1)
    Wq = torch.tensor(gauss_weights_flat, dtype=torch.float32, device=device)
    trunk_input = np.stack([Xq.ravel(), Yq.ravel()], axis=-1) 
    Trunk_input = torch.from_numpy(trunk_input).float().to(device)
    Trunk_input.requires_grad_(True)
   
    
    
    params["num_f"] = 50
    Xq_test, Yq_test, dxdy_test, f_values_test, coeffs_test, u_test, Phi1, Phi_total_exact = generate_branch_inputs_test(params["Lx"],params["nq_test"],params["num_f"])
    wq_test = np.ones(Xq_test.shape)*dxdy_test
    Wq_test = torch.tensor(wq_test.reshape(-1), dtype=torch.float32, device=device)
    trunk_input_test = np.stack([Xq_test.ravel(), Yq_test.ravel()], axis=-1)
    Trunk_input_test = torch.from_numpy(trunk_input_test).float().to(device)
    Trunk_input_test.requires_grad_(True)

    

    branch = MLP(5, params["k"], params["width"], params["depth"])
    branch.apply(init_weights)
    trunk = MLP(2, params["k"], params["width"], params["depth"])
    trunk.apply(init_weights)
    model = DeepONet(branch, trunk).to(device)

    # Adam optimizer
    train_adam(model, coeffs_all, f_values, coeffs_test, f_values_test, u_test, Phi1, Phi_total_exact, Wq, Wq_test, Trunk_input, Trunk_input_test, params, device, num_iter=2000)
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6e}")
    with open('IO_DeepONet_training_d{}_n{}_nx{}_k{}/Phase_training_summary.txt'.format(args.d, args.n,
                                                                                            args.nx,
                                                                                            args.k),
              'w') as file:
        file.write(f"Total iterations: {iter:.6e}\n")
        file.write(f"Final Loss: {final_loss:.6e}\n")
    results = np.array(results)
    np.savetxt(
        "IO_DeepONet_training_d{}_n{}_nx{}_k{}/Phase_training_data.csv".format(args.d, args.n, args.nx, args.k), results,
        delimiter=",",
        header="Iter,Loss", comments="")
    torch.save(model,
                'IO_DeepONet_training_d{}_n{}_nx{}_k{}/Phase_new.pth'.format(args.d, args.n, args.nx, args.k))
    

    

    F_values_test = torch.from_numpy(f_values_test).float().to(device)
    F_values_train = torch.from_numpy(f_values).float().to(device)
    

    initial_bases = [model]   
    

    basis_models, alphas_all, u_pred_all = greedy_train_bases_and_project(
        initial_bases=initial_bases,
        coeffs_train=coeffs_all.reshape(coeffs_all.shape[0], -1),
        F_train = F_values_train,
        Trunk_input=Trunk_input,
        Wq = Wq,
        coeffs_test=coeffs_test,
        F_values_test = F_values_test,
        Trunk_input_test=Trunk_input_test,
        Wq_test = Wq_test,
        params=params,
        num_total_bases=5,       
        epochs_each=3000,        
        lr=1e-3,
        reg=1e-14,
        device=device,
        save_path="IO_DeepONet_training_d4_n70_nx64_k90")