import numpy as np, pickle
from sklearn.mixture import GaussianMixture
from src.config import cfg

def fit_gmm(panic_prob, n_components=None):
    gc = cfg["gmm"]; nc = n_components or gc["n_components"]
    probs = np.array(panic_prob).reshape(-1, 1)
    gmm = GaussianMixture(n_components=nc, n_init=gc["n_init"], max_iter=gc["max_iter"], random_state=gc["random_state"])
    gmm.fit(probs)
    return gmm

def get_posterior_equal_points(gmm):
    from scipy.optimize import brentq
    si = np.argsort(gmm.means_.ravel()); b = []
    for i in range(gmm.means_.shape[0] - 1):
        mu1, mu2 = gmm.means_.ravel()[si[i]], gmm.means_.ravel()[si[i+1]]
        s1 = np.sqrt(gmm.covariances_.ravel()[si[i]]); s2 = np.sqrt(gmm.covariances_.ravel()[si[i+1]])
        w1, w2 = gmm.weights_[si[i]], gmm.weights_[si[i+1]]
        def f(x): return (w1*np.exp(-0.5*((x-mu1)/s1)**2)/s1) - (w2*np.exp(-0.5*((x-mu2)/s2)**2)/s2)
        try: b.append(brentq(f, mu1, mu2))
        except: b.append((mu1+mu2)/2)
    return np.array(b)

def assign_tiers(panic_prob, thresholds):
    prob = np.array(panic_prob); tiers = np.zeros_like(prob, dtype=int)
    for i, t in enumerate(sorted(thresholds)): tiers[prob > t] = i + 1
    return tiers

def bootstrap_boundaries(gmm, probs, n_resamples=100):
    rng = np.random.RandomState(cfg["gmm"]["random_state"]); bl = []
    n = len(probs)
    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        bg = GaussianMixture(**gmm.get_params()); bg.fit(probs[idx].reshape(-1, 1))
        bl.append(get_posterior_equal_points(bg))
    return np.array(bl)
