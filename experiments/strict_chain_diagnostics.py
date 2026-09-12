"""Split rank/fold R-hat and Geyer initial-positive/monotone sequence ESS.

Diagnostics are necessary evidence, not a proof of convergence. Tail ESS uses
the 5% and 95% indicators. The FFT autocovariance has the usual biased N divisor.
"""
import numpy as np
from scipy.stats import norm,rankdata


def split(x):
    n=len(x)//2
    return np.concatenate([x[:n],x[-n:]],axis=1)


def rank_normalize(x):
    shape=x.shape
    flat=x.reshape(-1,shape[-1])
    return norm.ppf((rankdata(flat,axis=0)-.375)/(len(flat)+.25)).reshape(shape)


def rhat_and_ess(x):
    x=split(x)
    n,m,d=x.shape
    within=x.var(axis=0,ddof=1).mean(axis=0)
    between=n*x.mean(axis=0).var(axis=0,ddof=1)
    varplus=(n-1)/n*within+between/n
    rhat=np.sqrt(varplus/np.maximum(within,1e-300))
    centered=x-x.mean(axis=0,keepdims=True)
    fft=np.fft.rfft(centered,n=2*n,axis=0)
    acov=np.fft.irfft(fft*np.conjugate(fft),n=2*n,axis=0)[:n].mean(axis=1)/n
    rho=1-(within-acov)/np.maximum(varplus,1e-300)
    rho[0]=1
    ess=[]
    for k in range(d):
        previous,total=np.inf,0.
        for i in range(0,n-1,2):
            pair=float(rho[i,k]+rho[i+1,k])
            if pair<0:
                break
            previous=min(previous,pair)
            total+=previous
        # Conservative cap avoids claiming >N effective observations.
        ess.append(n*m/max(1.,2*total-1))
    return rhat,np.asarray(ess)


def diagnose(trace):
    x=np.asarray(trace,dtype=np.float64)
    if x.ndim!=3 or len(x)<8 or x.shape[1]<4 or not np.isfinite(x).all():
        return dict(passed=False,reason='Need finite [draw, >=4 chains, observables], >=8 draws')
    if np.any(x.var(axis=(0,1))==0):
        return dict(passed=False,reason='Constant observable: mixing cannot be assessed')
    rank_rhat,bulk=rhat_and_ess(rank_normalize(x))
    folded_rhat,_=rhat_and_ess(rank_normalize(abs(x-np.median(x,axis=(0,1)))))
    _,raw_ess=rhat_and_ess(x)
    tails=[rhat_and_ess((x<=np.quantile(x,q,axis=(0,1))).astype(float))[1] for q in [.05,.95]]
    rhat=np.maximum(rank_rhat,folded_rhat)
    tail=np.minimum(*tails)
    return dict(passed=bool(np.all(rhat<1.01) and min(bulk)>=100 and min(tail)>=100),
        rhat=rhat.tolist(),bulk_ess=bulk.tolist(),tail_ess=tail.tolist(),raw_ess=raw_ess.tolist(),
        mean=x.mean(axis=(0,1)).tolist(),mcse=(x.std(axis=(0,1),ddof=1)/np.sqrt(raw_ess)).tolist(),
        max_rhat=float(max(rhat)),min_bulk_ess=float(min(bulk)),min_tail_ess=float(min(tail)),
        draws_per_chain=len(x),chains=x.shape[1],observables=x.shape[2],
        thresholds=dict(rhat=1.01,bulk_ess=100,tail_ess=100),
        caveat='Finite monitored observables; passing is necessary evidence, not proof of convergence')
