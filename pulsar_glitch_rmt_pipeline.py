#!/usr/bin/env python3
"""
Pulsar glitch RMT pipeline — per-pulsar memory spectrum.
Superfluid angular-momentum charge-and-release: accumulation -> glitch -> re-accumulation.
CRITICAL: analyze each pulsar SEPARATELY, never pool (superposition collapses to Poisson).
Handles non-stationary glitch rate (unfolding) and observation gaps (J0537 2264-d gap).
Data: Jodrell Bank Observatory glitch catalogue (Basu+2022), per-pulsar glitch epochs (MJD).
"""
import json, numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from math import gamma as _gamma
import warnings; warnings.filterwarnings('ignore')

R_POI, R_GOE, R_GUE = 0.3863, 0.5307, 0.6027

def compute_r(sp):
    if len(sp) < 2: return np.nan, np.nan
    r = np.minimum(sp[:-1], sp[1:]) / np.maximum(sp[:-1], sp[1:])
    return float(np.mean(r)), float(np.std(r)/np.sqrt(len(r)))

def cv(sp): return float(np.std(sp)/np.mean(sp))

def brody(s):
    from scipy.optimize import minimize_scalar
    def nll(b):
        a = (_gamma((b+2)/(b+1)))**(b+1)
        return -np.sum(np.log(b+1)+np.log(a)+b*np.log(s+1e-12)-a*s**(b+1))
    return float(minimize_scalar(nll, bounds=(0.01,3.0), method='bounded').x)

def unfold(ep):
    """Local unfolding: removes non-stationary glitch rate (e.g. Crab 1995-2006 increase)."""
    ep = np.sort(ep); N = len(ep); c = np.arange(1, N+1)
    spl = UnivariateSpline(ep, c, s=N/2.0); u = spl(ep)
    d = np.diff(u); d = d[d > 0]
    return d/np.mean(d)

def boot(s, nb=5000, seed=11):
    rng = np.random.default_rng(seed); n = len(s); o = []
    for _ in range(nb):
        ss = s[rng.integers(0, n, n)]
        if len(ss) > 2:
            o.append(np.mean(np.minimum(ss[:-1],ss[1:])/np.maximum(ss[:-1],ss[1:])))
    return [float(x) for x in np.percentile(o, [2.5, 97.5])]

def signull(rval, n, kind, span, nsim=2000, seed=3):
    rng = np.random.default_rng(seed); rs = []
    for _ in range(nsim):
        if kind == 'poisson':
            ev = np.sort(rng.uniform(0, span, n))
        else:
            H = rng.normal(size=(n,n)); H = (H+H.T)/2
            e = np.sort(np.linalg.eigvalsh(H)); sp = np.diff(e); sp = sp/np.mean(sp)
            ev = np.cumsum(sp)*span/np.sum(sp)
        su = unfold(ev); r, _ = compute_r(su); rs.append(r)
    rs = np.array(rs)
    return float((rval-np.mean(rs))/np.std(rs))

def analyze(name, epochs, drop_largest_gap=False):
    ep = np.array(sorted(epochs))
    if drop_largest_gap:  # remove a single observation gap (e.g. J0537 2011-2017)
        ep = ep[:np.argmax(np.diff(ep))+1]
    su = unfold(ep); r, re = compute_r(su); span = ep.max()-ep.min()
    print(f"\n=== {name} ===")
    print(f"  n_intervals={len(ep)-1}  mean_interval={np.mean(np.diff(ep)):.0f} d")
    print(f"  <r>={r:.4f}+/-{re:.4f}  CV={cv(su):.3f}  beta={brody(su):.2f}  95%CI={boot(su)}")
    print(f"  {signull(r,len(ep),'poisson',span):+.1f} sigma vs Poisson, "
          f"{signull(r,len(ep),'goe',span):+.1f} sigma vs GOE")
    cls = ("Poisson" if r < 0.44 else "Poisson-GOE" if r < 0.50 else "GOE" if r < 0.57 else "GUE/rigid")
    print(f"  -> {cls}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/glitch_epochs.json"
    G = json.load(open(path))
    print("="*66)
    print("  PULSAR GLITCH RMT — per-pulsar memory spectrum (NEVER pooled)")
    print("  Poisson=0.386  GOE=0.531  GUE=0.603")
    print("="*66)
    for name in ['B1338-62','J0537-6910','B1758-23','J2229+6114','B1800-21',
                 'J0631+1036','B1046-58','B1737-30','Vela','Crab']:
        if name in G:
            analyze(name, G[name], drop_largest_gap=(name == 'J0537-6910'))
