from Functions import Functions
import numpy as np
from scipy.interpolate import interp1d

class bootstrap: 

    def bootstrap(today, RT, Dates, Fixing, Settle, Expiry, Type):
        """
        Bootstrap the discount curve 
        Parameters
        ----------
        today : datetime
            The reference date for the bootstrap.
        RT : list
            List of rates corresponding to the financial instruments.
        Dates : list
            List of dates corresponding to the financial instruments.
        Fixing : list
            List of fixing dates for the financial instruments.
        Settle : list
            List of settlement dates for the financial instruments.
        Expiry : list
            List of expiry dates for the financial instruments.
        Type : list
            List of types corresponding to the financial instruments (e.g., 'DEPOSIT', 'FRA', 'SWAP').
        KeysFRA : list
            List of keys for FRA instruments.
        Returns
        -------
        dates : list
            List of dates for the discount curve.
        discounts : list
            List of discount factors corresponding to the dates.
        """
        dates = [today]
        discounts = [1.0]

        # 1. Bootstrap DEPOSIT
        idx_dep = [i for i, t in enumerate(Type) if t == 'DEPOSIT']
        dates_dep = [Dates[i] for i in idx_dep]
    
        rates_dep = [RT[i] for i in idx_dep]
        frac_dep = [Functions.yearfrac(today, d, 2) for d in dates_dep]
        disc_dep = [1 / (1 + f * r) for f, r in zip(frac_dep, rates_dep)]
        dates += dates_dep
        discounts += disc_dep

        # 2. Bootstrap FRA
        final_keys_fra = [i for i, t in enumerate(Type) if t == 'FRA']
        rates_fra = [RT[i] for i in final_keys_fra[:-1]]  
        fra_settle = [Settle[i - 1] for i in final_keys_fra[:-1]]
        fra_fixing = [Fixing[i - 1] for i in final_keys_fra[:-1]]
        fra_expiry = [Expiry[i - 1] for i in final_keys_fra[:-1]]

        for i in range(len(rates_fra)):
            delta = Functions.yearfrac(fra_settle[i], fra_expiry[i], 2)
            zr = Functions.zero_rates(dates, discounts, today)
            zr[0] = 0.0

            settle_ord = fra_settle[i].toordinal()
            interp_fn = interp1d( [d.toordinal() for d in dates],zr, kind='linear', fill_value='extrapolate')
            z_settle = interp_fn(settle_ord)
          
            DF_settle = np.exp(-z_settle * Functions.yearfrac(today, fra_settle[i], 3))

            DF_exp = DF_settle / (1 + delta * rates_fra[i])
            dates.append(fra_expiry[i])
            discounts.append(DF_exp)

        # 3. Bootstrap SWAP
        idx_swap = [i for i, t in enumerate(Type) if t == 'SWAP']
        dates_swap = [Dates[i] for i in idx_swap]
        rates_swap = [RT[i] for i in idx_swap]

       
        sorted_idx = np.argsort(dates)
        dates = [dates[i] for i in sorted_idx]
        discounts = [discounts[i] for i in sorted_idx]

        for i in range(len(rates_swap)):
            swap_end = dates_swap[i]
            settle_dt = fra_settle[5]
            flow_dates = [settle_dt] + dates_swap[:i + 1]
    
            flow_dates = [d for d in flow_dates if d <= swap_end]

            n_flows = len(flow_dates)
            inter_yfrac = [Functions.yearfrac(today, flow_dates[j], 6) - Functions.yearfrac(today, flow_dates[j - 1], 6) for j in range(1, n_flows)]
            inter_yfrac.insert(0, Functions.yearfrac(today, flow_dates[0], 6))

            BPV_tr = 0
            zr = Functions.zero_rates(dates, discounts, today)
            for j in range(n_flows - 1):
                z = np.interp(flow_dates[j].toordinal(), [d.toordinal() for d in dates], zr)
                DF_j = np.exp(-z * Functions.yearfrac(today, flow_dates[j], 3))
                BPV_tr += inter_yfrac[j] * DF_j

            delta_last = inter_yfrac[-1]
            DF_last = (1 - rates_swap[i] * BPV_tr) / (1 + rates_swap[i] * delta_last)

            dates.append(flow_dates[-1])
            discounts.append(DF_last)

       
        sorted_idx = np.argsort(dates)
        dates = [dates[i] for i in sorted_idx]
        discounts = [discounts[i] for i in sorted_idx]

        return dates, discounts