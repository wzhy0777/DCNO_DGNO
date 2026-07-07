import numpy as np
import scipy.special as sp
from scipy.fft import dst, idst
import matplotlib.pyplot as plt




def SPMM_2D_Poisson_explicit(m=4, sigma=2.0, L=16.0, h=0.25):
    """
    2D泊松方程的SP-MM方法实现
    """
    
    # ==================== 1 ====================
    N = int(2 * L / h)  # 每个方向的网格点数
    #print(f"网格参数: L={L}, h={h}, N={N}")
    x_int = np.linspace(-L + h, L - h, N - 1)
    y_int = x_int.copy()
    X, Y = np.meshgrid(x_int, y_int, indexing='ij')
    r2 = X**2 + Y**2
    r = np.sqrt(r2)
    dxdy = h**2
    
    # ==================== 2 ====================
    rho = np.exp(-r2 / sigma**2)

 
    moments = {}
    
    # 零阶矩
    moments[(0,0)] = np.sum(rho) * dxdy
    
    # 
    moments[(1,0)] = np.sum(rho * X) * dxdy
    moments[(0,1)] = np.sum(rho * Y) * dxdy
    
    # 
    moments[(2,0)] = np.sum(rho * X**2) * dxdy
    moments[(1,1)] = np.sum(rho * X * Y) * dxdy
    moments[(0,2)] = np.sum(rho * Y**2) * dxdy
    
    # 
    moments[(3,0)] = np.sum(rho * X**3) * dxdy
    moments[(2,1)] = np.sum(rho * X**2 * Y) * dxdy
    moments[(1,2)] = np.sum(rho * X * Y**2) * dxdy
    moments[(0,3)] = np.sum(rho * Y**3) * dxdy
    
    # 
    moments[(4,0)] = np.sum(rho * X**4) * dxdy
    moments[(3,1)] = np.sum(rho * X**3 * Y) * dxdy
    moments[(2,2)] = np.sum(rho * X**2 * Y**2) * dxdy
    moments[(1,3)] = np.sum(rho * X * Y**3) * dxdy
    moments[(0,4)] = np.sum(rho * Y**4) * dxdy
    

    
    # ==================== 4 ====================

    sigma_g = 2
    
    coefficients = {}
    coefficients_np = np.zeros([1,4])
    

    P00 = moments[(0,0)]
    coefficients[(0,0)] = P00
    coefficients_np[0,0] = coefficients[(0,0)]
    
    if m >= 1:
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
        r2 = x**2 + y**2
        G = 1/(2*np.pi*sigma_g**2) * np.exp(-r2/(2*sigma_g**2))
        
        if deriv_x == 0 and deriv_y == 0:
            return G
        
        # Hermite多项式
        def hermite(n, t):
            if n == 0: return 1
            elif n == 1: return t
            elif n == 2: return t**2 - 1
            elif n == 3: return t**3 - 3*t
            elif n == 4: return t**4 - 6*t**2 + 3
            else: return t * hermite(n-1, t) - (n-1) * hermite(n-2, t)
        
        scale = 1.0 / (sigma_g)
        H_x = (-scale)**deriv_x * hermite(deriv_x, x * scale) if deriv_x > 0 else 1
        H_y = (-scale)**deriv_y * hermite(deriv_y, y * scale) if deriv_y > 0 else 1
        
        return H_x * H_y * G
    

    rho1 = np.zeros_like(X)
    for (i, j), gamma in coefficients.items():
        if abs(gamma) > 1e-12:  
            g_deriv = gaussian_derivative(X, Y, i, j, sigma_g)
            rho1 += gamma * g_deriv
    
    def W_function(x, y, sigma_g):
        r = np.sqrt(x**2 + y**2)
        small_r_threshold = 1e-10
        
        mask = r < small_r_threshold
        result = np.zeros_like(r)
        

        r_valid = r[~mask]
        if len(r_valid) > 0:
            term = r_valid**2 / (2 * sigma_g**2)

            result[~mask] = -1/(4*np.pi) * (sp.exp1(term) + 2*np.log(r_valid))
        

        if np.any(mask):
            gamma_e = 0.5772156649  
            result[mask] = -1/(4*np.pi) * (-gamma_e + np.log(2*sigma_g**2))
            
        return result
    


    def W_derivative(x, y, deriv_x, deriv_y, sigma_g):
       
    
        x = np.asarray(x)
        y = np.asarray(y)
    
        r2 = x**2 + y**2
        r = np.sqrt(r2)
    
        small_r_threshold = 1e-10
        mask = r < small_r_threshold
    
        result = np.zeros_like(r, dtype=float)
    
        sigma2 = sigma_g**2
        exp_term = np.exp(-r2 / (2 * sigma2))
    

        if np.any(~mask):
            xv = x[~mask]
            yv = y[~mask]
            rv = r[~mask]
            r2v = r2[~mask]
            expv = exp_term[~mask]
    
            if (deriv_x, deriv_y) == (2, 0):
                result[~mask] = (
                    expv * (
                        -xv**2 * (yv**2 - sigma2 * (np.exp(r2v/(2*sigma2)) - 1))
                        - sigma2 * yv**2 * (np.exp(r2v/(2*sigma2)) - 1)
                        - xv**4
                    )
                    / (2 * np.pi * rv**4 * sigma2)
                )
    
            elif (deriv_x, deriv_y) == (0, 2):
                result[~mask] = (
                    expv * (
                        -xv**2 * (sigma2 * (np.exp(r2v/(2*sigma2)) - 1) + yv**2)
                        + sigma2 * yv**2 * (np.exp(r2v/(2*sigma2)) - 1)
                        - yv**4
                    )
                    / (2 * np.pi * rv**4 * sigma2)
                )
    
            elif (deriv_x, deriv_y) == (4, 0):
                result[~mask] = (
                    expv * (
                        xv**4 * (
                            6*sigma2**3*(np.exp(r2v/(2*sigma2)) - 1)
                            - yv**6 + 12*sigma2*yv**4 + 15*sigma2**2*yv**2
                        )
                        + 3*sigma2*xv**2*yv**2 * (
                            -12*sigma2**2*(np.exp(r2v/(2*sigma2)) - 1)
                            + 2*yv**4 + 5*sigma2*yv**2
                        )
                        + 6*sigma2**3*yv**4*(np.exp(r2v/(2*sigma2)) - 1)
                        - xv**10 - 3*xv**8*yv**2
                        - 3*xv**6*(yv**2 - sigma2)**2
                        - 3*sigma2**2*yv**6
                    )
                    / (2 * np.pi * rv**8 * sigma2**3)
                )
    
            elif (deriv_x, deriv_y) == (0, 4):
                result[~mask] = (
                    expv * (
                        -3*xv**4 * (
                            -2*sigma2**3*(np.exp(r2v/(2*sigma2)) - 1)
                            + yv**6 - 4*sigma2*yv**4 - 5*sigma2**2*yv**2
                        )
                        -3*xv**2*yv**2 * (
                            12*sigma2**3*(np.exp(r2v/(2*sigma2)) - 1)
                            + yv**6 - 2*sigma2*yv**4 - 5*sigma2**2*yv**2
                        )
                        - yv**4 * (
                            -6*sigma2**3*(np.exp(r2v/(2*sigma2)) - 1)
                            + yv**6 + 3*sigma2**2*yv**2
                        )
                        - xv**6 * (3*sigma2**2 + yv**4 - 6*sigma2*yv**2)
                    )
                    / (2 * np.pi * rv**8 * sigma2**3)
                )
    
            elif (deriv_x, deriv_y) == (2, 2):
                result[~mask] = (
                    expv * (
                        -3*xv**4 * (
                            2*sigma2**3*(np.exp(r2v/(2*sigma2)) - 1)
                            + yv**6 + 2*sigma2*yv**4 + 5*sigma2**2*yv**2
                        )
                        - xv**2*yv**2 * (
                            -36*sigma2**3*(np.exp(r2v/(2*sigma2)) - 1)
                            + yv**6 + 2*sigma2*yv**4 + 15*sigma2**2*yv**2
                        )
                        + sigma2*yv**4 * (
                            -6*sigma2**2*(np.exp(r2v/(2*sigma2)) - 1)
                            + yv**4 + 3*sigma2*yv**2
                        )
                        + xv**8*(sigma2 - yv**2)
                        + xv**6*(3*sigma2**2 - 3*yv**4 - 2*sigma2*yv**2)
                    )
                    / (2 * np.pi * rv**8 * sigma2**3)
                )
            elif (deriv_x, deriv_y) == (0, 0):
                result0 = 1
    
            else:
                raise ValueError("Unsupported derivative order.")
    

        if np.any(mask):
            if (deriv_x, deriv_y) in [(2, 0), (0, 2)]:
                result[mask] = -1.0 / (4 * np.pi * sigma2)
    
            elif (deriv_x, deriv_y) in [(4, 0), (0, 4)]:
                result[mask] = 3.0 / (8 * np.pi * sigma2**2)
    
            elif (deriv_x, deriv_y) == (2, 2):
                result[mask] = 1.0 / (8 * np.pi * sigma2**2)
            elif (deriv_x, deriv_y) == (0, 0):
                result = W_function(x, y, sigma_g)
            else:
                raise ValueError("Unsupported derivative order at origin.")
    
        return result

    

    phi1 = np.zeros_like(X)
    for (i, j), gamma in coefficients.items():
        if abs(gamma) > 1e-12:
            w_deriv = W_derivative(X, Y, i, j, sigma_g)
            phi1 += gamma * w_deriv
    
    
    rho2 = rho - rho1

    def sine_solver(rho, L, h):

        N = int(2 * L / h)
        n_int = N - 1  # 
        

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
        
        
        return phi
    
    
    
    phi2 = sine_solver(rho2, L, h)
    
    phi_total = phi1 + phi2
    

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
    
    phi_exact = analytical_solution(X, Y, sigma)
    
    return X,Y,dxdy, rho, rho1, rho2, phi2, phi1, phi_exact,  coefficients_np
    

def generate_branch_inputs_test(L,N,sample_num):
    np.random.seed(42)
    Sigma = np.random.uniform(1, 5, sample_num)
    Coeffs = np.zeros([sample_num,5])
    f_values = np.zeros([sample_num,N*N])
    u_test = np.zeros([sample_num,N*N])
    Phi1 = np.zeros([sample_num,N*N])
    Phi_total_exact = np.zeros([sample_num,N*N])
    Coeffs[:,0] = Sigma
    for i in range(sample_num):
        X, Y, dxdy, rho, rho1, rho2, phi2, phi1, phi_exact, coeffs = SPMM_2D_Poisson_explicit(m=4, sigma=Sigma[i], L=12, h=0.125)
        Coeffs[i,1:] = coeffs
        f_values[i,:] = rho2.reshape(1,N*N)
        u_test[i,:] = phi2.reshape(1,N*N)
        Phi1[i,:] = phi1.reshape(1,N*N)
        Phi_total_exact[i,:] = phi_exact.reshape(1,N*N)
    return X, Y, dxdy, f_values, Coeffs, u_test, Phi1, Phi_total_exact
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    