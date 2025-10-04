# CMS Certificate Pricing

## Overview

This project implements a complete **pricing pipeline for Constant Maturity Swap (CMS) certificates** in Python involving *SABR* and *Additive Bacheliers* models for the volatility.
  
It includes:

- **Yield curve bootstrap** from market instruments (`bootstrap.py`)
- **Volatility model calibration** (Bachelier and SABR) from swaption market data (`Calibration_Bachelier.py`, `Calibration_SABR.py`)
- **Certificate pricing** using the bootstrapped curve and calibrated volatilities (`Certificate_pricing.py`)
- A **Jupyter notebook (`main.ipynb`)** demonstrating the full workflow and results

The goal is to reproduce a transparent and modular pricing framework similar to those used in fixed-income derivatives desks.

---
