import numpy as np, pandas as pd
from src.rdd.rdd_main import fit_local_linear_rdd

def hac_sensitivity(y, x, lags=[1,2,3,5,10]):
    return pd.DataFrame([{"lag": l, **fit_local_linear_rdd(y, x, se_type="HAC", hac_maxlags=l)} for l in lags])

def placebo_test(y, x, dates, bw=14):
    return pd.DataFrame([{"date": d, **fit_local_linear_rdd(y, x-d, bandwidth=bw)} for d in dates])

def residual_acf(residuals, max_lag=10):
    from statsmodels.tsa.stattools import acf
    return acf(residuals, nlags=max_lag)
