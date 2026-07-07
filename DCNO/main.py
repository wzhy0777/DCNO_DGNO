# Import standard libraries
import os  # Miscellaneous operating system interfaces
import time  # Time access and conversions
import warnings  
import argparse
import scipy.io as io
import scipy.special as sp
import numpy as np  
import torch
from functorch import vmap,grad

from deeponet import *  
import pandas as pd
import torch.nn as nn
from scipy.interpolate import griddata
import torch.nn.functional as F
from scipy.special import i0
from scipy.io import loadmat
import csv


# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--d', type=int, default=4, help='depth')
parser.add_argument('--n', type=int, default=70, help='width')
parser.add_argument('--k', type=int, default=90, help='width of the output layer')
parser.add_argument('--nx', type=int, default=100, help='Sampling')
args = parser.parse_args()

def generate_data(params, X, Y, nx, ny, a, b, seed):
    np.random.seed(seed)
    num_points = nx * ny
    rho_all = np.zeros((params["num_f"], num_points))
    sigma_all = np.random.uniform(a, b, params["num_f"]).reshape(-1,1)
    phi_all = np.zeros((params["num_f"], num_points))
    r = np.sqrt(X**2+Y**2)
    for k in range(0,params["num_f"]):
        sigma = sigma_all[k,0]
        rho_all[k,:] = np.exp(-r**2/(sigma**2)).reshape(1,-1)
        
        phi_all[k, :] = analytical_solution(X, Y, sigma).reshape(1,-1)    
    return rho_all, sigma_all, phi_all


def analytical_solution(x, y, sigma):
    r2 = x**2 + y**2
    r = np.sqrt(r2)   
    gamma_e = 0.5772156649015328
    safe_r2 = np.where(r2 < 1e-12, 1e-12, r2)
    safe_r = np.sqrt(safe_r2)
    result = np.zeros_like(x)
    mask = r2 > 1e-12
    result[mask] = -sigma**2/4 * (-sp.expi(-r2[mask]/sigma**2) + np.log(safe_r2[mask]))
    
    result[~mask] = sigma**2/4 * (gamma_e - 2*np.log(sigma))
    
    return result






def eval_bases(basis_models, coeffs_tensor, trunk_input_pts):

    device = coeffs_tensor.device
    B = coeffs_tensor.shape[0]
    N = trunk_input_pts.shape[0]
    M = len(basis_models)
    if M == 0:
        return torch.zeros(B,0,N,device=device), torch.zeros(B,0,N,device=device)
    psi_list = []
    with torch.no_grad(): 
        for mdl in basis_models:
            psi_bn = mdl(coeffs_tensor, trunk_input_pts)  # (B,N)
            psi_list.append(psi_bn)  # keep as float tensors
   
    psi_vals = torch.stack(psi_list, dim=1)  # (B, M, N)
    return psi_vals


def solve_coeffs_for_batch(psi_vals, eps_reg=1e-3):
    
    B, M, N = psi_vals.shape
    device = psi_vals.device
    dtype = psi_vals.dtype
    alphas = torch.zeros(B, M, device=device, dtype=dtype)

    scale_values = params["scales"][:M]
    scale_tensor = torch.tensor(scale_values, device=device, dtype=dtype)
    alphas = scale_tensor.unsqueeze(0).expand(alphas.size(0), -1)

    return alphas  # (B,M)



def greedy_train_bases_and_project(initial_bases, coeffs_train, f_train, trunk_input, trunk_input_test, params,
                                   num_total_bases=6, epochs_each=300, lr=5e-4, reg=1e-6,
                                   device=torch.device('cpu'), save_path=None):

    device = device
    trunk_pts = torch.tensor(trunk_input, dtype=torch.float32, device=device, requires_grad=True)  # (N,2)
    coeffs_t = torch.tensor(coeffs_train, dtype=torch.float32, device=device)  # (B, branch_dim)
    f_t = torch.tensor(f_train, dtype=torch.float32, device=device)  # (B, N)
    basis_models = list(initial_bases)  
    M0 = len(basis_models)
    
    # ------------------------------------
    coeffs_test_tensor = torch.tensor(coeffs_test, dtype=torch.float32, device=device)  # (3, branch_dim)
    trunk_pts_test = torch.tensor(trunk_input_test, dtype=torch.float32, device=device, requires_grad=True)
    psi_vals_test = eval_bases(basis_models, coeffs_test_tensor, trunk_pts_test)
    f_test_tensor = torch.tensor(f_values_test, dtype=torch.float32, device=device)  # (3, N)
    alphas_test = solve_coeffs_for_batch(psi_vals_test)
    u_test_pred = torch.einsum('bm,bmn->bn', alphas_test, psi_vals_test)  # (3, N)
    rel_l2_test = torch.sqrt(torch.sum((u_test_pred - torch.tensor(u_test, device=device))**2, dim=1)) / \
                  torch.sqrt(torch.sum(torch.tensor(u_test, device=device)**2, dim=1))
    mean_val = rel_l2_test.mean()
    print("mean rel_l2_test:", mean_val.item())
    
    mean_values = []
    

    for j in range(M0, num_total_bases):
        print(f"\n=== Greedy training for basis {j+1}/{num_total_bases} ===")
        if len(basis_models) == 0:
            I_lap = torch.zeros_like(f_t)  # (B,N)
        else:
            psi_vals= eval_bases(basis_models, coeffs_t, trunk_pts)  # (B, M, N)
            alphas = solve_coeffs_for_batch(psi_vals)  # (B,M)
            u_vals = torch.einsum('bm,bmn->bn', alphas, psi_vals)
            u_vals.detach()


        branch = MLP(1, params["k"]+20*j, params["width"]+20*j, params["depth"], j).to(device)
        branch.apply(init_weights)
        trunk = MLP(2, params["k"]+20*j, params["width"]+20*j, params["depth"], j).to(device)
        trunk.apply(init_weights)
        psi_new = DeepONet(branch, trunk).to(device)
        
        norm = np.sqrt(np.mean(np.square(u_star-u_vals.cpu().numpy())))
        params["scales"][j] = norm
        

        
    
        opt = torch.optim.Adam(psi_new.parameters(), lr=lr, weight_decay=1e-6)
        u_test_sub = torch.tensor(u_star, dtype=torch.float32, device=device)
        u_vals_sub = u_vals
        
        for ep in range(1, epochs_each + 1):
            coeffs_b = coeffs_t      # (B_s, branch_dim)
            trunk_pts_sub = trunk_pts
                        
            u_new_sub = psi_new(coeffs_b, trunk_pts_sub)
            loss = torch.mean(torch.square(u_new_sub - (u_test_sub - u_vals_sub)/norm))

            opt.zero_grad()
            loss.backward()
            opt.step()
            results.append([ep, loss.item()])

            if ep % 5000 == 0 or ep == 1:
                print(f"  basis {j+1} ep {ep}/{epochs_each} loss {loss.item():.4e}")


        if save_path:
            model_path = f"{save_path}/basis_{j}.pth"
            torch.save(psi_new, model_path)  # 保存完整模型
            print(f"Saved complete model for basis {j} to {model_path}")
            
            torch.save({"model": psi_new,"scale": params["scales"][j],}, 
                       model_path)
            
            

        basis_models.append(psi_new)
        psi_vals_all = eval_bases(basis_models, coeffs_t, trunk_pts)
        
        # ------------------------------------
        psi_vals_test = eval_bases(basis_models, coeffs_test_tensor, trunk_pts_test)
        alphas_test = solve_coeffs_for_batch(psi_vals_test)
        u_test_pred = torch.einsum('bm,bmn->bn', alphas_test, psi_vals_test)  # (3, N)
        rel_l2_test = torch.sqrt(torch.sum((u_test_pred - torch.tensor(u_test, device=device))**2, dim=1)) / \
                      torch.sqrt(torch.sum(torch.tensor(u_test, device=device)**2, dim=1))
        mean_val = rel_l2_test.mean()
        print("mean rel_l2_test:", mean_val.item())
        
        mean_values.append(mean_val.item())

    
    with open('IO_DeepONet_training_d{}_n{}_nx{}_k{}/mean rel_l2_test.csv'.format(args.d, args.n,
                                                                                           args.nx,
                                                                                           args.k),
              'w') as file:
        writer = csv.writer(file)
        writer.writerows([[value] for value in mean_values])
        
    # after greedy loop: compute final projection on full training set
    psi_vals_final = psi_vals_test
    alphas_final =alphas_test  # (B,M)
    u_pred_final = torch.einsum('bm,bmn->bn', alphas_final, psi_vals_final)  # (B,N)
    return basis_models, alphas_final.detach().cpu().numpy(), u_pred_final.detach().cpu().numpy()




def GetGradients(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True, allow_unused=True)[0]


def errorFun(output, target):
    error = output - target
    error = np.sqrt(np.mean(error * error))
    ref = np.sqrt(np.mean(target * target))
    return error / (ref + 1E-14)




def loss_function(deeponet, coeffs_all, f_values, u_star, params, device):

    x_grid = np.linspace(-1*params["Lx"], params["Lx"], params["nx"])
    y_grid = np.linspace(-1*params["Ly"], params["Ly"], params["ny"])
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    trunk_input = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)

    Branch_input = torch.from_numpy(coeffs_all).float().to(device)   # (num_f, coeff_dim)
    Trunk_input = torch.tensor(trunk_input, dtype=torch.float32, device=device, requires_grad=True)  # (num_points, 2)
    

    norm = np.sqrt(np.mean(np.square(u_star)))
    params["scales"][0] = norm

    U_pred = deeponet(Branch_input, Trunk_input)
    loss = torch.mean(torch.square(U_pred - (1/norm)*torch.tensor(u_star, device=device)))
    
    return loss


def train_adam(deeponet, coeffs_all, f_values, u_star, coeffs_test, u_test, params, X_grid_test, Y_grid_test, device, num_iter=10000):
    trunk_input_test = np.stack([X_grid_test.ravel(), Y_grid_test.ravel()], axis=-1)
    optimizer = torch.optim.Adam(deeponet.parameters(), lr=0.001)
    global iter
    


    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        loss = loss_function(deeponet, coeffs_all, f_values, u_star, params, device)
        loss.backward()
        optimizer.step()


        results.append([iter, loss.item()])  # , error
        iter += 1
        if iter == 1 or iter % 3000 == 0:
            print(
                f"Train adam - Iter: {iter:.6e} - Loss: {loss.item():.6e}")  # - test error: {error:.6e}

    Branch_input_test = torch.from_numpy(coeffs_test).float().to(device)
    Trunk_input_test = torch.from_numpy(trunk_input_test).float().to(device)
    U_test_pred = deeponet(Branch_input_test, Trunk_input_test) 
    rel_l2_test = torch.sqrt(torch.sum((params["scales"][0]*U_test_pred - torch.tensor(u_test, device=device))**2, dim=1)) / \
                  torch.sqrt(torch.sum(torch.tensor(u_test, device=device)**2, dim=1))
    print(f"  -> Relative L2 error : {rel_l2_test.mean().item()}")
    

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
    params["nx_test"] = 150
    params["ny_test"] = 150
    params["num_f"] = 100
    params["batch_size"] = 4096
    params["scales"] = [1, 1, 1, 1, 1]

    
    x_grid = np.linspace(-1*params["Lx"], params["Lx"], params["nx"])
    y_grid = np.linspace(-1*params["Ly"], params["Ly"], params["ny"])
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    x_grid_test = np.linspace(-1*params["Lx"], params["Lx"], params["nx_test"])
    y_grid_test = np.linspace(-1*params["Ly"], params["Ly"], params["ny_test"])
    X_grid_test, Y_grid_test = np.meshgrid(x_grid_test, y_grid_test)


    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    f_values, coeffs_all, u_star = generate_data(params, X_flat, Y_flat, params["nx"], params["ny"], 1, 5, 46)


    params["num_f"] = 50
    X_flat_test = X_grid_test.flatten()
    Y_flat_test = Y_grid_test.flatten()
    f_values_test, coeffs_test, u_test = generate_data(params, X_flat_test, Y_flat_test, params["nx_test"], params["ny_test"], 1, 5, 42)

    


    branch = MLP(1, params["k"], params["width"], params["depth"], 1)
    branch.apply(init_weights)
    trunk = MLP(2, params["k"], params["width"], params["depth"], 1)
    trunk.apply(init_weights)
    model = DeepONet(branch, trunk).to(device)
    
    print(params)

    # Adam optimizer
    train_adam(model, coeffs_all, f_values, u_star, coeffs_test, u_test, params, X_grid_test, Y_grid_test, device, num_iter=5000)
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6e}")
    with open('IO_DeepONet_training_d{}_n{}_nx{}_k{}/Phase_training_summary.txt'.format(args.d, args.n,
                                                                                            args.nx,
                                                                                            args.k),
              'w') as file:
        file.write(f"Total iterations: {iter:.6e}\n")
        file.write(f"Final Loss: {final_loss:.6e}\n")
    

    torch.save({"model": model,"scale": params["scales"][0],}, 
                'IO_DeepONet_training_d{}_n{}_nx{}_k{}/Phase_new.pth'.format(args.d, args.n, args.nx, args.k))



    trunk_input = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)  # (N,2)
    trunk_input_test = np.stack([X_grid_test.ravel(), Y_grid_test.ravel()], axis=-1)
    
    initial_bases = [model]    
    f_vals_arr = f_values  
    coeffs_arr = coeffs_all.reshape(coeffs_all.shape[0], -1)  # (num_f, branch_dim)

    basis_models, alphas_all, u_pred_all = greedy_train_bases_and_project(
        initial_bases=initial_bases,
        coeffs_train=coeffs_arr,
        f_train=f_vals_arr,
        trunk_input=trunk_input,
        trunk_input_test = trunk_input_test,
        params=params,
        num_total_bases=5,       # S
        epochs_each=5000,        
        lr=5e-4,
        reg=1e-14,
        device=device,
        save_path='IO_DeepONet_training_d{}_n{}_nx{}_k{}'.format(args.d, args.n, args.nx, args.k)
    )
    print(params)
    
    
    results = np.array(results)
    np.savetxt(
        "IO_DeepONet_training_d{}_n{}_nx{}_k{}/Phase_training_data.csv".format(args.d, args.n, args.nx, args.k), results,
        delimiter=",",
        header="Iter,Loss", comments="")
    
    
