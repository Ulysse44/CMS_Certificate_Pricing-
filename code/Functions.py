import numpy as np
import QuantLib as ql
from scipy.interpolate import interp1d
import datetime as datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
class Functions :

    def converter(date):
        """Convert a date in the format datenum to a datetime.
        Parameters
        ----------
        date : float
            The date in datenum.
        Returns
        -------
        date : datetime.datetime
            The corresponding datetime object.

        """
        ordinal = int(date)
        fractional = float(date % 1)
        return datetime.datetime.fromordinal(ordinal) + datetime.timedelta(days=fractional) - datetime.timedelta(days=366)







    def  yearfrac(start_date, end_date, basis):
        """Calculate the year fraction between two dates based on the specified basis.
        Parameters
        ----------
        start_date : datetime.datetime or list of datetime.datetime
            The start date(s) for the calculation.
        end_date : datetime.datetime or list of datetime.datetime
            The end date(s) for the calculation.
        basis : int 
            The basis for the year fraction calculation.
        Returns
        -------
        year_fraction : float or np.ndarray
            The year fraction between the start and end dates.
        """


        if isinstance(start_date, list) or isinstance(end_date, list):
             return np.array([Functions.yearfrac(s, e, basis) for s, e in zip(start_date, end_date)])
       
        
        if basis == 0 or basis == 8:
            convention = ql.ActualActual()
        elif basis == 2 or basis == 9:
            convention = ql.Actual360()
        elif basis == 3 or basis == 10:
            convention = ql.Actual365Fixed()
        elif basis == 6 or basis == 11:
            convention = ql.Thirty360(ql.Thirty360.European)
        else:
            raise ValueError(" ")

       
        ql_start = ql.Date(start_date.day, start_date.month, start_date.year)
        ql_end = ql.Date(end_date.day, end_date.month, end_date.year)

        return convention.yearFraction(ql_start, ql_end)
        






    def zero_rates(dates, discounts, sett_date):
        """Calculate zero rates from discount factors.
        Parameters
        ----------
        dates : list of datetime.datetime
            The dates corresponding to the discount factors.
        discounts : list of float
            The discount factors.
        sett_date : datetime.datetime
            The settlement date.
        Returns
        -------
        zero_rates : array
            The zero rates corresponding to the dates.
        """

        time_fracs = np.array([Functions.yearfrac(sett_date, d, 3) for d in dates])
        time_fracs[time_fracs == 0] = 1e-10  
        return -np.log(discounts) / time_fracs
    







    def get_discount(sett_date, dates, discounts, maturities):
        """Calculate discount factors for given maturities based on zero rates.
        Parameters
        ----------
        sett_date : datetime.datetime
            The settlement date.
        dates : list of datetime.datetime
            The dates corresponding to the discount factors.
        discounts : list of float
            The discount factors.
        maturities : list of datetime.datetime
        The maturities for which to calculate the discount factors.
        Returns
        -------
        discount_factors : array
            The discount factors for the specified maturities.
        """
        z = Functions.zero_rates(dates, discounts, sett_date)

        dates_num = np.array([d.toordinal() for d in dates])
        mat_num = np.array([m.toordinal() for m in maturities])
        
        z_interp = interp1d(dates_num, z, kind='linear', fill_value="extrapolate")
        z_mat = z_interp(mat_num)

        delta = np.array([Functions.yearfrac(dates[0], m, 3) for m in maturities])
        return np.exp(-z_mat * delta)
    


    def get_fwd_swap_rate(sett_date, dates, discounts, start_date, tenor_years,q=2):
        """Calculate the forward swap rate for a given start date and tenor.
        Parameters
        ----------
        sett_date : datetime.datetime
            The settlement date.
        dates : list of datetime.datetime
            The dates corresponding to the discount factors.
        discounts : list of float
            The discount factors.
        start_date : datetime.datetime
          
        tenor_years : int
            The tenor in years.
        q : int, optional
            The number of reset dates per year (default is 2).
        Returns
        -------
        forward_rate : float
            The forward swap rate.
        """
        reset_dates = pd.date_range(start=start_date, periods=int(tenor_years*q), freq=pd.DateOffset(months=12//q))
        reset_dates = reset_dates[1:]  

        all_dates = [start_date] + list(reset_dates)
        
        discounts_reset = Functions.get_discount(sett_date, dates, discounts, all_dates)

        delta = Functions.yearfrac(all_dates[:-1], all_dates[1:], 3)  
        BPV = np.sum(delta * discounts_reset[1:])

        return (discounts_reset[0] - discounts_reset[-1]) / BPV
    
    
    