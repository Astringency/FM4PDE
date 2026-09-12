import numpy as np
from experiments.strict_chain_diagnostics import diagnose,rhat_and_ess


def test_iid_chains_pass_and_separated_chains_fail():
    rng=np.random.default_rng(991)
    x=rng.normal(size=(4096,4,8))
    assert diagnose(x)['passed']
    x[:,0]+=2
    assert not diagnose(x)['passed']
    assert diagnose(x)['max_rhat']>1.1


def test_ar1_ess_matches_analytic_autocorrelation_time():
    rng=np.random.default_rng(881)
    x=np.zeros((16384,4,3))
    for i in range(1,len(x)):
        x[i]=.8*x[i-1]+.6*rng.normal(size=(4,3))
    _,ess=rhat_and_ess(x)
    exact=x.shape[0]*x.shape[1]*(1-.8)/(1+.8)
    assert np.all(abs(ess/exact-1)<.2),(ess,exact)


def test_constant_chains_cannot_be_declared_converged():
    assert not diagnose(np.ones((128,4,2)))['passed']
