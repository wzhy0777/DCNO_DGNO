import numpy as np
import scipy.special as sp
from scipy.fft import dst, idst
import matplotlib.pyplot as plt


def generate_gauss_points(Lx, Ly, nq):
    x1d, w1d = np.polynomial.legendre.leggauss(nq)
    
    x = x1d * Lx
    y = x1d * Ly

    Wx, Wy = np.meshgrid(w1d * Lx, w1d * Ly, indexing="ij")
    W2d = Wx * Wy  
    
    Xq, Yq = np.meshgrid(x, y, indexing="ij")
    return Xq, Yq, W2d, x, y

def SPMM_2D_Poisson_explicit(m, sigma, X, Y , W , xq, yq):

    


    

    r2 = X**2 + Y**2
    r = np.sqrt(r2)
    
    # ==================== 2====================
    rho = np.exp(-r2 / sigma**2)

    
    # 
    moments = {}
    
    # 
    moments[(0,0)] = np.sum(rho* W) 
    
    # 
    moments[(1,0)] = np.sum(rho * X * W) 
    moments[(0,1)] = np.sum(rho * Y * W)
    
    # 
    moments[(2,0)] = np.sum(rho * X**2 * W) 
    moments[(1,1)] = np.sum(rho * X * Y * W)
    moments[(0,2)] = np.sum(rho * Y**2 * W) 
    
    # 
    moments[(3,0)] = np.sum(rho * X**3 * W) 
    moments[(2,1)] = np.sum(rho * X**2 * Y * W) 
    moments[(1,2)] = np.sum(rho * X * Y**2 * W) 
    moments[(0,3)] = np.sum(rho * Y**3 * W) 
    
    # 
    moments[(4,0)] = np.sum(rho * X**4 * W) 
    moments[(3,1)] = np.sum(rho * X**3 * Y * W)
    moments[(2,2)] = np.sum(rho * X**2 * Y**2 * W)
    moments[(1,3)] = np.sum(rho * X * Y**3 * W) 
    moments[(0,4)] = np.sum(rho * Y**4 * W) 
    

    

    sigma_g = 2
    
    coefficients = {}
    coefficients_np = np.zeros([1,4])
    

    P00 = moments[(0,0)]
    coefficients[(0,0)] = P00
    coefficients_np[0,0] = coefficients[(0,0)]
    
    if m >= 1:
        # 一阶系数
        coefficients[(1,0)] = -moments[(1,0)]
        coefficients[(0,1)] = -moments[(0,1)]
    
    if m >= 2:
        coefficients[(2,0)] = (moments[(2,0)] - sigma_g**2 * coefficients[(0,0)]) / 2
        coefficients[(1,1)] = moments[(1,1)]
        coefficients[(0,2)] = (moments[(0,2)] - sigma_g**2 * coefficients[(0,0)]) / 2

        coefficients_np[0,1] = coefficients[(0,2)]
    
    if m >= 3:
        
        coefficients[(3,0)] = (-moments[(3,0)] - 3*sigma_g**2*coefficients[(1,0)]) / 6
        coefficients[(2,1)] = (-moments[(2,1)] - sigma_g**2*coefficients[(0,1)]) / 2
        coefficients[(1,2)] = (-moments[(1,2)] - sigma_g**2*coefficients[(1,0)]) / 2
        coefficients[(0,3)] = (-moments[(0,3)] - 3*sigma_g**2*coefficients[(0,1)]) / 6

    
    if m >= 4:
        coefficients[(4,0)] = (moments[(4,0)] - 3*sigma_g**4*coefficients[(0,0)] - 
                             12*sigma_g**2*coefficients[(2,0)]) / 24
        coefficients[(3,1)] = (moments[(3,1)] - 3*sigma_g**2*coefficients[(1,1)]) / 6
        coefficients[(2,2)] = (moments[(2,2)] - sigma_g**4*coefficients[(0,0)] - 
                             2*sigma_g**2*(coefficients[(2,0)] + coefficients[(0,2)])) / 4
        coefficients[(1,3)] = (moments[(1,3)] - 3*sigma_g**2*coefficients[(1,1)]) / 6
        coefficients[(0,4)] = (moments[(0,4)] - 3*sigma_g**4*coefficients[(0,0)] - 
                             12*sigma_g**2*coefficients[(0,2)]) / 24

        coefficients_np[0,2] = coefficients[(0,4)]
        coefficients_np[0,3] = coefficients[(2,2)]

    
    def gaussian_derivative(x, y, deriv_x, deriv_y, sigma_g):
        """计算高斯函数的偏导数（使用Hermite多项式）"""
        r2 = x**2 + y**2
        G = 1/(2*np.pi*sigma_g**2) * np.exp(-r2/(2*sigma_g**2))
        
        if deriv_x == 0 and deriv_y == 0:
            return G

        def hermite(n, t):
            if n == 0: return 1
            elif n == 1: return t
            elif n == 2: return t**2 - 1
            elif n == 3: return t**3 - 3*t
            elif n == 4: return t**4 - 6*t**2 + 3
            else: return t * hermite(n-1, t) - (n-1) * hermite(n-2, t)
        
        scale = 1.0 / (sigma_g )
        H_x = (-scale)**deriv_x * hermite(deriv_x, x * scale) if deriv_x > 0 else 1
        H_y = (-scale)**deriv_y * hermite(deriv_y, y * scale) if deriv_y > 0 else 1
        
        return H_x * H_y * G
    
    # 构建ρ1
    rho1 = np.zeros_like(X)
    for (i, j), gamma in coefficients.items():
        if abs(gamma) > 1e-12:  
            g_deriv = gaussian_derivative(X, Y, i, j, sigma_g)
            rho1 += gamma * g_deriv
    
    rho2 = rho - rho1
    
  
    
    L = 32
    h = 0.25
    N = 256
    x_int = np.linspace(-L + h, L - h, N - 1)
    y_int = x_int.copy()
    X_uniform, Y_uniform = np.meshgrid(x_int, y_int, indexing='ij')
    
    r2_uniform = X_uniform**2 + Y_uniform**2
    rho_uniform = np.exp(-r2_uniform / sigma**2)
    
    rho1_uniform = np.zeros_like(X_uniform)
    for (i, j), gamma in coefficients.items():
        if abs(gamma) > 1e-12:  
            g_deriv_uniform = gaussian_derivative(X_uniform, Y_uniform, i, j, sigma_g)
            rho1_uniform += gamma * g_deriv_uniform
            
    
    rho2_uniform = rho_uniform - rho1_uniform

    def sine_solver(rho, L, h):

        N = int(2 * L / h)
        n_int = N - 1  # 内部点数
        
        # 波数
        p = np.arange(1, n_int + 1)
        q = np.arange(1, n_int + 1)
        mu_p = np.pi * p / (2 * L)
        mu_q = np.pi * q / (2 * L)
        MU_P, MU_Q = np.meshgrid(mu_p, mu_q, indexing='ij')
        eigenvalues = MU_P**2 + MU_Q**2

        def dst2(u):
            return dst(dst(u, type=1, axis=0), type=1, axis=1)
        
        def idst2(u):
            return idst(idst(u, type=1, axis=0), type=1, axis=1)
        
        rho_hat = dst2(rho)
        phi_hat = rho_hat / eigenvalues
        phi = idst2(phi_hat)
        
        
        return phi, phi_hat
    
    def sine_reconstruct(U_hat, xg, yg, L):
        n = U_hat.shape[0]
        p = np.arange(1, n+1)
        q = np.arange(1, n+1)
    
        # basis values
        sin_px = np.sin(np.pi * np.outer(p, xg + L) / (2*L))
        sin_qy = np.sin(np.pi * np.outer(q, yg + L) / (2*L))
        

        scale = (1/(n+1))**2
  
        return scale * np.einsum("pq,pi,qj->ij", U_hat, sin_px, sin_qy)

    
    

    
    phi2_uniform, phi2_hat_uniform = sine_solver(rho2_uniform, L, h)
    phi_gauss = sine_reconstruct(phi2_hat_uniform, xq, yq, L)

    
    
    
    return rho, rho1, rho2, phi_gauss, coefficients_np
    

def generate_branch_inputs_train(L,N,sample_num):
    Xq, Yq, wq, xq, yq = generate_gauss_points(L, L, N)
    Sigma = np.linspace(1, 5, sample_num)
    Coeffs = np.zeros([sample_num,5])
    f_values = np.zeros([sample_num,N*N])
    u_star = np.zeros([sample_num,N*N])
    Coeffs[:,0] = Sigma
    for i in range(sample_num):
        rho, rho1, rho2, phi2_gauss, coeffs = SPMM_2D_Poisson_explicit(4, Sigma[i], Xq, Yq, wq, xq, yq)
        Coeffs[i,1:] = coeffs
        f_values[i,:] = rho2.reshape(1,N*N)
        u_star[i,:] = phi2_gauss.reshape(1,N*N)
    return Xq, Yq, wq, f_values, Coeffs, u_star
        
    
    
    
    
    
    
    
    