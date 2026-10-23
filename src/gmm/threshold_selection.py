import numpy as np
from src.gmm.fit_gmm import assign_tiers, get_posterior_equal_points

def grid_search_thresholds(probs, gmm, target=(0.2, 0.6, 0.2), margin=0.05, steps=100):
    tb = get_posterior_equal_points(gmm)
    b1, b2 = tb[1], tb[-2] if len(tb) >= 3 else (tb[0], tb[-1])
    best_s, best_t = float("inf"), (b1, b2)
    for t1 in np.linspace(max(0, b1-margin), min(1, b1+margin), steps):
        for t2 in np.linspace(max(0, b2-margin), min(1, b2+margin), steps):
            if t1 >= t2: continue
            r = np.bincount(assign_tiers(probs, [t1, t2]), minlength=3)[:3] / len(probs)
            s = sum((ri - ti)**2 for ri, ti in zip(r, target))
            if s < best_s: best_s, best_t = s, (t1, t2)
    return best_t, best_s

def honest_report():
    from src.config import cfg
    return {"claimed": "GMM posterior equal-probability points",
            "actual": "Grid search to hit 20-60-20 ratio",
            "true_points": cfg["gmm"]["true_posterior_equal_points"],
            "final": cfg["gmm"]["final_thresholds"]}
