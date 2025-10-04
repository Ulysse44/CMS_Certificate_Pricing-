import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint,fsolve
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution
from Functions import *
from tqdm import tqdm

class Calibration_SABR :


    def sigma_computation(k, alpha, rho, nu, t, f):
        """
        SABR implied volatility computation.
        
        Parameters:
        -----------
        k : array-like or float
            Strike prices.
        alpha : float
            Alpha parameter of the SABR model.
        rho : float
            Rho parameter of the SABR model.
        nu : float
            Nu (vol-of-vol) parameter of the SABR model.
        t : float
            Time to maturity.
        f : float
            Forward rate.

        Returns:
        --------
        sig : np.ndarray
            SABR model-implied volatilities.
        """
        k = np.asarray(k, dtype=np.float64)
        
       
        epsilon = 1e-12
        alpha = np.maximum(alpha, epsilon)

        xi = (nu / alpha) * (f - k)
        sqrt_term = np.sqrt(1 - 2 * rho * xi + xi**2)
        x_hat = np.log((sqrt_term - rho + xi) / (1 - rho + epsilon))
        
        x_hat = np.where(np.abs(x_hat) < epsilon, epsilon, x_hat)

        sig = alpha * (xi / x_hat) * (1 + ((2 - 3 * rho**2) / 24) * t * nu**2)

        return sig
    
    def sigma_computation_2(k, nu_prime, rho, f):
        """
        SABR implied volatility computation with nu_prime.
        Parameters:
        -----------
        k : array-like or float
            Strike prices.
        nu_prime : float
            Nu prime parameter of the SABR model.
        rho : float
            Rho parameter of the SABR model.
        f : float
            Forward rate.
        Returns:
        --------
        sig : np.ndarray
            SABR model-implied volatilities.
        """
        xi = nu_prime * (f - k)
        epsilon = 1e-12
        sqrt_term = np.sqrt(1 - 2 * rho * xi + xi**2)
        x_hat = np.log((sqrt_term - rho + xi) / (1 - rho + epsilon))
        x_hat = np.where(np.abs(x_hat) < epsilon, epsilon, x_hat)
        return xi / x_hat

    def sabr_calibration(dates, discounts, T, Exp, atm_sigma, gridB, today2, K,flag):
        """
        SABR calibration function.
        Parameters:
        ----------
        dates : array of datetime
            Dates.
        discounts : list of float
            Discount factors corresponding to the dates.
        T : list of int
            List of tenors in months.
        Exp : list of int
            List of expiries in months.
        atm_sigma : np.ndarray
            ATM volatilities.
        gridB : np.ndarray
            Grid of market volatilities.
        today2 : datetime
            Reference date for the calibration.
        K : list of float
            Strike prices.
        flag : str
            Calibration method, either "Basic" or "Cascade".
        Returns:
        -------
        alpha_vec : np.ndarray
            Alpha parameters for the SABR model.
        rho_vec : np.ndarray
            Rho parameters for the SABR model.
        nu_vec : np.ndarray
            Nu parameters for the SABR model.
        sigma_model : np.ndarray
            Model-implied volatilities.
        sigma_mkt : np.ndarray
            Market volatilities.
        obj_value : np.ndarray
            Objective function values for the calibration.
        """


        N = len(T)
        alpha_vec = np.zeros(N)
        rho_vec = np.zeros(N)
        nu_vec = np.zeros(N)
        sigma_model = np.zeros((N, len(K)))
        sigma_mkt = np.zeros((N, len(K)))
        obj_value = np.zeros(N)

        grid_A_mat = np.tile(atm_sigma.reshape(-1, 1), (1, gridB.shape[1]))
        grid_B_new = grid_A_mat * 1e-4 + gridB * 1e-4
 
        for i in tqdm(range(N)):
            x0 = np.array([0.0896, 0.0492, 0.0795])
            sigma_mkt_i = grid_B_new[i, :]

            expiry_months = Exp[i]
            tenor_months = T[i]

            expiry_day = today2 + relativedelta(months=+expiry_months)
            tenor_day = today2 + relativedelta(months=+expiry_months + tenor_months)
            
            
            t= Functions.yearfrac(today2,expiry_day, 2) 
            f = Functions.get_fwd_swap_rate(today2, dates, discounts, expiry_day, tenor_months //12, q=2)
            k = f + np.array(K) / 100


            if flag == "Basic":
                def obj_func(x):
                    alpha, rho, nu = x
                    sigma_mod = Calibration_SABR.sigma_computation(k, alpha, rho, nu, t, f)
                    return np.sum((sigma_mod - sigma_mkt_i)**2)

                def atm_constraint(x):
                    alpha, rho, nu = x
                    return alpha * (1 + ((2 - 3 * rho**2) / 24) * t * nu**2) - atm_sigma[i] * 1e-4

                

                bounds = Bounds([1e-7, -0.999, 1e-6], [1.0, 0.999, 2.0])
                constraint = NonlinearConstraint(atm_constraint, 0, 0)
        
                res = minimize(obj_func, x0, method='trust-constr',
                            bounds=bounds,
                            constraints=[constraint],
                            options={'verbose': 0})
                if res.fun> 1e-5:
               
                    x0= np.array([0.00215182, 0.32767745, 1.07639157])
                    res = minimize(obj_func, x0, method='trust-constr',
                            bounds=bounds,
                            constraints=[constraint],
                            options={'verbose': 0})
             
                alpha_vec[i], rho_vec[i], nu_vec[i] = res.x
                sigma_model[i, :] = Calibration_SABR.sigma_computation(k, *res.x, t, f)
                sigma_mkt[i, :] = sigma_mkt_i
                obj_value[i] = res.fun

            
        
            elif flag == "Cascade":
                x0 = np.array([0.01, 0.1])
                sigma_mkt_i = grid_B_new[i, :] / (atm_sigma[i] * 1e-4)
                def obj_func_rho_nuprime(x):
                    nu_prime, rho = x
                    return np.sum((Calibration_SABR.sigma_computation_2(k, nu_prime, rho, f) - sigma_mkt_i)**2)

                if i>0 :
                    x0 = np.array([nu_vec[i-1]/alpha_vec[i-1], rho_vec[i-1]])
                res = minimize(obj_func_rho_nuprime, x0, method='SLSQP')
                nu_prime = res.x[0]
                rho = res.x[1]
                rho_vec[i] = rho
                obj_value[i] = res.fun

                def find_nu_alpha(x):
                    nu, alpha = x
                    eq1 = nu / alpha - nu_prime
                    eq2 = alpha * (1 + ((2 - 3 * rho**2) / 24) * t * nu**2) - atm_sigma[i] * 1e-4
                    return [eq1, eq2]

                sol = fsolve(find_nu_alpha, [1.0, 0.001])
                nu_vec[i], alpha_vec[i] = sol
                sigma_model[i, :] = Calibration_SABR.sigma_computation_2(k, nu_prime, rho, f) * (atm_sigma[i] * 1e-4)
                sigma_mkt[i, :] = sigma_mkt_i * (atm_sigma[i] * 1e-4)
            else:
                raise ValueError("Flag must be either 'Basic' or 'Cascade'.")
            
        return alpha_vec, rho_vec, nu_vec, sigma_model, sigma_mkt, obj_value