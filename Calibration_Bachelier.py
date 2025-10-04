import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.optimize import minimize, fsolve
from Functions import *
from tqdm import tqdm





class Calibration_Bachelier:
    def Bachelier_swaption_price(y, vol):
        """
        Computes the Bachelier swaption price using the normal distribution.
        Parameters
        ----------
        y : np.ndarray
            moneyness /sqrt(t) where t is the time to maturity.
        vol : np.ndarray
            Volatility of the underlying swap rate.
        Returns
        -------
        np.ndarray
            Bachelier swaption price.
        """
       

        y = np.asarray(y)
        vol = np.asarray(vol)
        return -y * norm.cdf(-y / vol) + vol * np.exp(-y**2 / (2 * vol**2)) / np.sqrt(2 * np.pi)
    
    def integral_fft(x1, parameters, t, M, x, alpha):
        """
        Computes the integral using FFT method.

        Parameters
        ----------
        x1 : float
            Starting point of the log-moneyness grid
        parameters : list or array
            [sigma, k, eta]
        t : float
            Time to maturity
        M : int
            Exponent for FFT size, N = 2^M
        x : float or np.ndarray
            Log-moneyness at which to interpolate
        alpha : float
            Damping factor

        Returns
        -------
        np.ndarray
            Interpolated value of the Fourier integral
        """
        sigma, k_param, eta = parameters
        N = 2 ** M
        x_N = -x1
        dx = (x_N - x1) / N
        dz = 2 * np.pi / (N * dx)
        z_N = dz * N / 2
        z1 = -z_N
        z = z1 + dz * np.arange(N)
        x_tot = x1 + dx * np.arange(N)
        j = np.arange(1, N + 1)

        f = Calibration_Bachelier.integrand_function_nig(z, eta, k_param, sigma, t, alpha) * np.exp(-1j * x1 * dz * (j - 1))
        fft_result = np.fft.fft(f)
        integral = dz * np.exp(-1j * z1 * x_tot) * fft_result

        interpolator = interp1d(x_tot, integral.real, kind='cubic', fill_value='extrapolate')
        return interpolator(np.asarray(x))
    
    def integrand_function_nig(xi, eta, k, sigma, t, alpha_val):
        """
        Compute the integrand used in FFT or quadrature pricing methods under a NIG-type model.

        Parameters
        ----------
        xi : np.ndarray
            Vector of integration points (frequencies)
        eta : float
            Exponential tilting parameter
        k : float
            Shape parameter of the Gamma time change
        sigma : float
            Volatility parameter
        t : float
            Time to maturity
        alpha_val : float
            Damping parameter (alpha=0 for VG, otherwise CGMY/NIG-like)

        Returns
        -------
        integrand : np.ndarray
            Complex-valued integrand to be used in Fourier inversion
        """
        a = 0.5 

       
        def laplace_exponent(u):
            return (t / k) * ((1 - alpha_val) / alpha_val) * (1 - (1 + u * k / (1 - alpha_val)) ** alpha_val)

       
        def phi(u):
            return np.exp(laplace_exponent(1j * u * sigma * eta + 0.5 * u ** 2 * sigma ** 2) + 1j * u * sigma * eta)

        shifted_xi = xi - 1j * a
        numerator = phi(shifted_xi)
        denominator = (1j * xi + a) ** 2

        integrand = numerator / denominator
        return integrand

    def model_price(x, t, parameters, flag):
        """
        Computes the normalized price of a European swaption using Lewis formula.

        Parameters:
        -----------
        x : array-like
            Moneyness 
        t : float
            Time to maturity
        parameters : list or array
            [sigma, k, eta]
        flag : str
            'Quad' or 'FFT'

        Returns:
        --------
        price_normalized : np.ndarray
            Normalized swaption price
        """
       
        sigma, k_param, eta = parameters
       
        y = np.asarray(x) / np.sqrt(t)
   
        alpha_val = 1 / 2
        a = 1 / 2

        if flag == "Quad":
            def laplace_exp(u, k, alpha):
                return (1 / k) * ((1 - alpha) / alpha) * (1 - (1 + u * k / (1 - alpha)) ** alpha)

            def phi(u):
                return np.exp(
                    laplace_exp(1j * u * sigma * eta + (u ** 2) * sigma ** 2 / 2, k_param, alpha_val)
                    + 1j * u * sigma * eta
                )

            def integrand(csi, yy):
                return phi(csi - 1j * a) * np.exp(-1j * csi * yy) / ((1j * csi + a) ** 2)

            integral_val = np.zeros_like(y, dtype=np.complex128)
            for idx, yy in enumerate(y):
                f_aux = lambda z: integrand(z, yy)
                integral_val[idx], _ = quad(f_aux, -10, 60)
           
        elif flag == "FFT":
            
            x1 = -50
            M = 15
            integral_val = Calibration_Bachelier.integral_fft(x1, parameters, t, M, y, alpha_val)
        else:
            raise ValueError("Invalid flag. Use 'Quad' or 'FFT'.")

        price_normalized = np.exp(-a * y) / (2 * np.pi) * integral_val
       
        return price_normalized
    


    def additive_bachelier_calibration(dates, discounts, T, Exp, atm_sigma, gridB, today2, K, flag1, flag2):
        """
        Calibrates the Bachelier model parameters for a set of swaptions using either a basic or cascade method.
        Parameters
        ----------
        dates : np.ndarray
            Dates.
        discounts : np.ndarray
            Array of discount factors corresponding to the dates.
        T : np.ndarray
            Array of tenors.
        Exp : np.ndarray
            Array of expiry months.
        atm_sigma : np.ndarray
            Array of ATM volatilities.
        gridB : np.ndarray
            Array of market swaption volatilities.
        today2 : datetime.datetime
            Reference date for the calibration.
        K : list
            List of relative strikes.
        flag1 : str
            Calibration method, either "Basic" or "Cascade".
        flag2 : str
            Pricing method, either "Quad" or "FFT".
        Returns
        -------
        sigma_vec : np.ndarray
            Array of calibrated sigma values for the Bachelier model.
        k_vec : np.ndarray
            Array of calibrated k values for the Bachelier model.
        eta_vec : np.ndarray
            Array of calibrated eta values for the Bachelier model.
        I : np.ndarray
            Array of implied volatilities.
        """
        N = len(T)
        I = np.zeros((N, len(K)))
        sigma_vec = np.zeros(N)
        k_vec = np.zeros(N)
        eta_vec = np.zeros(N)
        
        grid_A_mat = np.tile(atm_sigma.reshape(-1, 1), (1, gridB.shape[1]))
        grid_B_new = grid_A_mat * 1e-4 + gridB * 1e-4
        
       
        for i in tqdm(range(N)):
            sigma_mkt_i = grid_B_new[i, :]

            expiry_months = Exp[i]
            tenor_months = T[i]

            expiry_day = today2 + relativedelta(months=+expiry_months)
            tenor_day = today2 + relativedelta(months=+expiry_months + tenor_months)
          
            t= Functions.yearfrac(today2,expiry_day, 2)  # ACT/360
            f = Functions.get_fwd_swap_rate(today2, dates, discounts, expiry_day, tenor_months //12)
            x = np.array(K) / 100
         
            if flag1 == "Basic":
                x0 = [0.001, 0.002, -0.1]
               
                def obj_func_price(par):
                    model_vals = np.real(Calibration_Bachelier.model_price(x, t, par, flag2))
                    if np.any(np.isnan(model_vals)):
                        print("NaN encountered in model values. Parameters:", par)
                        return np.inf
                    market_vals = Calibration_Bachelier.Bachelier_swaption_price(x / np.sqrt(t), sigma_mkt_i)
                    
                    return np.sum(np.abs(model_vals - market_vals) ** 2)
                
                bounds = [(0, 1), (0, None), (None, None)]
                #use the best method for optimization
         
                res = minimize(obj_func_price, x0, method='trust-constr',
                            bounds=bounds, options={'disp': False})
                
                
               
                
                x_opt = res.x
                
                sigma_vec[i], k_vec[i], eta_vec[i] = x_opt
                
                def obj_func_price2(imp_vol):
                    return abs(np.real(Calibration_Bachelier.model_price(x, t, x_opt, flag2)) - Calibration_Bachelier.Bachelier_swaption_price(x / np.sqrt(t), imp_vol))
                
                IV, _, flag, _ = fsolve(obj_func_price2, sigma_mkt_i, full_output=True, xtol=1e-6, maxfev=100)
                I[i, :] = IV
    
            elif flag1 == "Cascade":
            
                x0 = [0.2, 0.1]
          
                
                def I0(z):
                    return np.sqrt(2 * np.pi) * np.real(Calibration_Bachelier.model_price([0], t, [1, z[0], z[1]], flag2))
                
                y_hat = x / (atm_sigma[i] * 1e-4)
               
                
                def obj_func_price2(z):
                   
                    I0_val = np.sqrt(2 * np.pi) * np.real(Calibration_Bachelier.model_price([0], t, [1, z[0], z[1]], flag2))[0]
                    
                    model_vals = np.real(Calibration_Bachelier.model_price(y_hat * I0_val, t, [1, z[0], z[1]], flag2)) / I0_val
                    target = Calibration_Bachelier.Bachelier_swaption_price(x / np.sqrt(t), sigma_mkt_i) / (atm_sigma[i] * 1e-4)
                    return np.sum(np.abs(model_vals - target) ** 2)
                
                if i > 0:
                    x0 = [k_vec[i-1], eta_vec[i-1]]
                
                bounds = [(1e-7, None), (None, None)]
                res = minimize(obj_func_price2, x0, method='SLSQP',
                            options={ 'disp': False}, bounds=bounds)
                x_opt = res.x
                k_vec[i], eta_vec[i] = x_opt
                sigma_vec[i] = atm_sigma[i] / I0(x_opt)
            
            
            
                
                def obj_func_price3(H):
                    z_best = np.array([k_vec[i], eta_vec[i]])
                    
                    I0_val = I0(z_best)
                    y_hat = x / (atm_sigma[i] * 1e-4 * np.sqrt(t)) 

                    residuals = []

                    for h in H:
                        model_val = np.real(Calibration_Bachelier.model_price(
                        y_hat * I0_val * np.sqrt(t), t, [1, z_best[0], z_best[1]], flag2)) / I0_val
                        market_val = Calibration_Bachelier.Bachelier_swaption_price(y_hat, h)
                        residuals.append(model_val - market_val)
                        return np.sum(np.abs(residuals[0]) ** 2)
                
                H_guess = sigma_mkt_i / (atm_sigma[i] * 1e-4)
                
                H  = minimize(obj_func_price3, H_guess, method='SLSQP',options={'disp':False}).x
               

                I[i, :] = atm_sigma[i] * 1e-4 * H
              

        return sigma_vec, k_vec, eta_vec, I