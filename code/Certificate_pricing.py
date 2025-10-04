import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas.tseries.holiday as holiday
from pandas.tseries.offsets import BDay
import pandas as pd
import QuantLib as ql
from scipy.interpolate import interp1d
from Functions import *
from code.Calibration_Bachelier import *

class Certificate_pricing :


    def schedule_following_BD_convention(sett_date, q=2):
        """
        Generate a schedule of coupon payment dates and reset dates based on the settlement date and frequency.
        Parameters
        ----------
        sett_date : datetime.datetime
            The settlement date for the schedule.
        q : int, optional
            The number of reset dates per year (default is 2).
        Returns
        -------
        coupon_payment_dates : list of datetime.datetime
            List of coupon payment dates.
        reset_dates : list of datetime.datetime
            List of reset dates corresponding to the coupon payment dates.
        """

        calendar = ql.TARGET()

        
        ql_sett_date = ql.Date(sett_date.day, sett_date.month, sett_date.year)

        coupon_payment_dates = []

    
        step = relativedelta(months=12 // q)
        current_date = sett_date

        for _ in range(1, 3 * q + 1):
            current_date += step
            ql_current = ql.Date(current_date.day, current_date.month, current_date.year)

        
            adjusted_ql_date = calendar.adjust(ql_current, ql.Following)
            adjusted_date = datetime.datetime(adjusted_ql_date.year(), adjusted_ql_date.month(), adjusted_ql_date.dayOfMonth())

            coupon_payment_dates.append(adjusted_date)

       
        reset_dates = []
        for dt in coupon_payment_dates:
            ql_dt = ql.Date(dt.day, dt.month, dt.year)
            first = calendar.advance(ql_dt, -1, ql.Days, ql.Preceding)
            second = calendar.advance(first, -1, ql.Days, ql.Preceding)
            reset_dt = datetime.datetime(second.year(), second.month(), second.dayOfMonth())
            reset_dates.append(reset_dt)

        return coupon_payment_dates, reset_dates   





    def vol_smile_from_vol_surf(vol_surf, expiry_dates, target_expiry, tenor):
        """
        Extract a volatility smile for a specific expiry and tenor from a volatility surface.
        Parameters
        ----------
        vol_surf : np.ndarray
            The volatility surface 
        expiry_dates : list of datetime.datetime
            List of expiry dates 
        target_expiry : datetime.datetime
            The target expiry date for which to extract the volatility smile.
        tenor : int
            The tenor in years for which to extract the volatility smile.
        Returns
        -------
        vol_smile : np.ndarray
            The extracted volatility smile as a 1D array.
        """
        if tenor == 2:
            rem = 2
        elif tenor == 5:
            rem = 3
        elif tenor == 10:
            rem = 4
        else:
            raise ValueError("Unsupported tenor")

        vol_surf_fixed_tenor = vol_surf[rem::6, :]

        expiry_ordinal = np.array([d.toordinal() for d in expiry_dates])
        target_ordinal = target_expiry.toordinal()
       
        interp_func = interp1d(expiry_ordinal, vol_surf_fixed_tenor, axis=0, kind='linear')
        vol_smile = interp_func(target_ordinal)

        return vol_smile






    def value_portfolio_suPreplica(settDate, coupon_dates, reset_dates, vol_smile, dates, discounts, q, n, N, K):
        """
        Calculate the value of a portfolio of caplets using the super-replica method.
        Parameters
        ----------
        settDate : datetime.datetime
            The settlement date for the portfolio.
        coupon_dates : list of datetime.datetime
            List of coupon payment dates.
        reset_dates : list of datetime.datetime
            List of reset dates corresponding to the coupon payment dates.
        vol_smile : np.ndarray
            The volatility smile for the caplets.
        dates : list of datetime.datetime
            List of dates corresponding to the discount factors.
        discounts : np.ndarray
            The discount factors corresponding to the dates.
        q : int
            The number of reset dates per year.
        n : int
            Tenor in years.
        N : int
            The number of captlets considered.
        K : float
            The strike price for the caplets.
        Returns
        -------
        value : float
            The value of the portfolio.
        """


        off_set_CMS = 0.005
        M = 0.08
        z_0 = -off_set_CMS
        z_N = np.sqrt(M/100) - off_set_CMS
        delta_z = (z_N + off_set_CMS) / N
        caplet_strikes_ub = np.arange(z_0, z_N + delta_z, delta_z)
        A = 2 * delta_z

        caplet_prices_all = Certificate_pricing.caplet_price(caplet_strikes_ub, coupon_dates, reset_dates, q, n, settDate, dates, discounts, vol_smile, K)
        caplet_long = caplet_prices_all[:-1]
        caplet_short = caplet_prices_all[-1]

        value = 100 * delta_z * caplet_long[0] + 100 * A * np.sum(caplet_long[1:]) - 100 * ((2*N - 1) * delta_z) * caplet_short
        return value
    
    def value_portfolio_suBreplica(sett_date, coupon_payment_date, reset_date,vol_smile, dates, discounts, q, n, N, K):
        """
        Calculate the value of a portfolio of caplets using the sub-replica method.
        Parameters
        ----------
        sett_date : datetime.datetime
            The settlement date for the portfolio.
        coupon_payment_date : datetime.datetime
            The coupon payment date for the caplets.
        reset_date : datetime.datetime
            The reset date corresponding to the coupon payment date.
        vol_smile : np.ndarray
            The volatility smile for the caplets. 
        dates : list of datetime.datetime
            List of dates corresponding to the discount factors.
        discounts : np.ndarray
            The discount factors corresponding to the dates.
        q : int
            The number of reset dates per year.
        n : int
            Tenor in years.
        N : int
            The number of caplets considered.
        K : float
            The strike price for the caplets.
        Returns
        -------
        value_portfolio : float
            The value of the portfolio.
        """

        off_set_CMS = 0.005
        M = 0.08  
        z_0 = -off_set_CMS
        z_N = np.sqrt(M / 100) - off_set_CMS
        delta_z = (z_N + off_set_CMS) / N

      
        caplet_strikes_lb = np.arange(z_0 + delta_z / 2, z_N, delta_z)
        caplet_strikes_lb = np.append(caplet_strikes_lb, z_N)  

        A = 2 * delta_z  

      
        caplet_prices = Certificate_pricing.caplet_price(caplet_strikes_lb,coupon_payment_date,reset_date,
            q,
            n,
            sett_date,
            dates,
            discounts,
            vol_smile,
            K
        )

        caplet_long = caplet_prices[:-1]
        caplet_short = caplet_prices[-1]

        value_portfolio = (
            100 * A * caplet_long[0] +
            100 * A * np.sum(caplet_long[1:]) -
            100 * 2 * (N * delta_z) * caplet_short
        )

        return value_portfolio
    
    def caplet_price(strike_target, payment_dates, reset_dates, q, n, settDate, dates, discounts, vol_smile, K):
        """
        Calculate the price of a caplet using the Bachelier model.
        Parameters
        ----------
        strike_target : float or np.ndarray
            The target strike price for the caplet.
        payment_dates : list of datetime.datetime
            List of payment dates for the caplet.
        reset_dates : list of datetime.datetime
            List of reset dates corresponding to the payment dates.
        q : int
            The number of reset dates per year.
        n : int
            tenor in years.
        settDate : datetime.datetime
            The settlement date for the caplet.
        dates : list of datetime.datetime
            List of dates corresponding to the discount factors.
        discounts : np.ndarray
            The discount factors corresponding to the dates.
        vol_smile : np.ndarray
            The volatility smile for the caplet.
        K : float
            The strike price for the caplet.
        Returns
        -------
        price : float
            The price of the caplet.
        """
        act365 = 3  
        
        disc = Functions.get_discount(settDate, dates, discounts, [payment_dates])[0]
       
        t = Functions.yearfrac(settDate, reset_dates, act365)
        fwd_rate = Functions.get_fwd_swap_rate(settDate, dates, discounts, reset_dates,n, q)
        abs_strikes = fwd_rate + K/100
        y = (strike_target - fwd_rate) / np.sqrt(t)

        sigma = np.interp(strike_target, abs_strikes, vol_smile)
        BPV_fwd = Certificate_pricing.get_fwd_BPV(settDate, dates, discounts, reset_dates,12*n, q)
        swaption_price = BPV_fwd * disc * np.sqrt(t) * Calibration_Bachelier.Bachelier_swaption_price(y, sigma)

        diff_G = lambda x: -1/(1/(x/q+1)**n - 1) - (n*x) / (q*(x/q+1)**(n+1)*(1/(x/q+1)**n - 1)**2)
        price = disc * (swaption_price / BPV_fwd + BPV_fwd * diff_G(fwd_rate) * sigma**2 * t * norm.cdf((fwd_rate - strike_target) / (sigma * np.sqrt(t))))

        return price








    def get_fwd_BPV(settDate, dates, discounts, start_date, tenor_months, q):
        """
        Calculate the forward BPV (Basis Point Value) for a given start date and tenor.
        Parameters
        ----------
        settDate : datetime.datetime
            The settlement date.
        dates : list of datetime.datetime
            The dates corresponding to the discount factors.
        discounts : np.ndarray
            The discount factors corresponding to the dates.
        start_date : datetime.datetime
            The start date for the forward BPV calculation.
        tenor_months : int
            The tenor in months.
        q : int
            The number of reset dates per year.
        Returns
        -------
        BPV : float
            The forward BPV for the specified start date and tenor.
        """
        reset_dates = pd.date_range(start=start_date, periods=int(tenor_months*q/12), freq=pd.DateOffset(months=12//q))
        reset_dates = reset_dates[1:]

        all_dates = [start_date] + list(reset_dates)
        
        discounts_reset = Functions.get_discount(settDate, dates, discounts, all_dates)
        fwd_discounts = discounts_reset[1:] / discounts_reset[0]

        delta = Functions.yearfrac(all_dates[:-1], reset_dates, 3)
        BPV = np.sum(delta * fwd_discounts)
        return BPV

    def compute_NPV_coupons(sett_date, dates, discounts,coupon_payment_dates, reset_dates,expiry_dates, tenors, q, Z, strike_mkt, NN):
        """
        Compute the NPV of coupons for a portfolio of caplets.
        Parameters
        ----------
        sett_date : datetime.datetime
            The settlement date for the portfolio.
        dates : list of datetime.datetime
            List of dates corresponding to the discount factors.
        discounts : np.ndarray
            The discount factors corresponding to the dates.
        coupon_payment_dates : list of datetime.datetime
            List of coupon payment dates.
        reset_dates : list of datetime.datetime
            List of reset dates corresponding to the coupon payment dates.
        expiry_dates : list of datetime.datetime
            List of expiry dates for the caplets.
        tenors : list of int
            List of tenors in years for the caplets.
        q : int
            The number of reset dates per year.
        Z : np.ndarray
            The volatility surface for the caplets.
        strike_mkt : float
            The market strike price for the caplets.
        NN : list of int
            List of numbers of caplets considered for each coupon date.
        Returns
        -------
        NPV_sup : np.ndarray
            The NPV of the sup-portfolio .
        NPV_sub : np.ndarray
            The NPV of the sub-portfolio.
        """
        num_coupons = len(coupon_payment_dates)
        num_N = len(NN)

        coupon_values_ub = np.zeros((num_coupons, num_N))
        coupon_values_lb = np.zeros((num_coupons, num_N))
        delta_dates =[sett_date] + coupon_payment_dates[:-1]
        delta=[Functions.yearfrac(d,coupon_date,6) for d,coupon_date in zip(delta_dates,coupon_payment_dates)]
        for tm in range(num_coupons):
           
            vol_smile = Certificate_pricing.vol_smile_from_vol_surf(Z, expiry_dates, reset_dates[tm], tenors[tm])
            
            value_portfolio_ub = np.ones(num_N)
            value_portfolio_lb = np.ones(num_N)
            print("Computing NPV for coupon date:", coupon_payment_dates[tm])
            print('reset date:', reset_dates[tm])
            print("tenor:", tenors[tm])
            for count, N_val in enumerate(NN):
                value_portfolio_ub[count] = Certificate_pricing.value_portfolio_suPreplica(
                    sett_date,
                    coupon_payment_dates[tm],
                    reset_dates[tm],
                    vol_smile,
                    dates,
                    discounts,
                    q,
                    tenors[tm],
                    N_val,
                    strike_mkt
                )
                value_portfolio_lb[count] = Certificate_pricing.value_portfolio_suBreplica(
                    sett_date,
                    coupon_payment_dates[tm],
                    reset_dates[tm],
                    vol_smile,
                    dates,
                    discounts,
                    q,
                    tenors[tm],
                    N_val,
                    strike_mkt
                )

            coupon_values_ub[tm, :] = delta[tm]*value_portfolio_ub
            coupon_values_lb[tm, :] = delta[tm]*value_portfolio_lb

        # Aggregate NPV over all coupon dates
        NPV_sup = np.sum(coupon_values_ub, axis=0)
        NPV_sub = np.sum(coupon_values_lb, axis=0)

        return NPV_sup, NPV_sub
    


    def compute_protection(NPV_coupons, today, settDate, payment_dates, dates, discounts):
        """
        Compute the protection value of a portfolio of caplets.
        Parameters
        ----------
        NPV_coupons : float
            The NPV of the coupons for the portfolio.
        today : datetime.datetime
            The current date for the protection calculation.
        settDate : datetime.datetime
            The settlement date for the portfolio.
        payment_dates : list of datetime.datetime
            List of payment dates for the portfolio.
        dates : list of datetime.datetime
            List of dates corresponding to the discount factors.
        discounts : np.ndarray
            The discount factors corresponding to the dates.
        Returns
        -------
        P : float
            The protection value of the portfolio.
        """
        act360 = 2
        spread = 130 * 10**(-4)

    
        discounts_payment_dates = Functions.get_discount(today, dates, discounts, payment_dates)

       
        payment_dates_full = [settDate] + payment_dates

        
        delta = Functions.yearfrac(payment_dates_full[:-1], payment_dates_full[1:], act360)

        X = 0.01 + NPV_coupons - (1 - discounts_payment_dates[-1]) - spread * np.sum(delta * discounts_payment_dates)
        P = 1 - X / discounts_payment_dates[-1]

        return P