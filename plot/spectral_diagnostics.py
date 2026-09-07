"""Orthonormal field spectra with exact error-energy decomposition."""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn


def coefficients(field, kind):
    x = np.asarray(field, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all()
    if kind == 'periodic_fft': return np.fft.fft2(x, norm='ortho')
    if kind == 'cosine': return dctn(x, type=2, norm='ortho')
    raise ValueError(kind)


def reconstruct(c, kind):
    return np.fft.ifft2(c, norm='ortho').real if kind == 'periodic_fft' else idctn(c, type=2, norm='ortho')


def radii(shape, kind):
    h,w = shape
    if kind == 'periodic_fft':
        x,y = np.fft.fftfreq(h)*h, np.fft.fftfreq(w)*w
    elif kind == 'cosine':
        x,y = np.arange(h), np.arange(w)
    else: raise ValueError(kind)
    return np.hypot(x[:,None], y[None,:])


def spectral_record(prediction, truth, kind, cutoffs=(8.,32.)):
    p,t = coefficients(prediction,kind), coefficients(truth,kind)
    ep,et,ee = np.abs(p)**2, np.abs(t)**2, np.abs(p-t)**2
    total = float(et.sum())
    assert total > 0
    physical_error = float(np.sum((np.asarray(prediction,dtype=np.float64)-truth)**2))
    assert np.isclose(ee.sum(),physical_error,rtol=2e-12,atol=2e-18)
    radius = radii(t.shape,kind)
    shell = np.floor(radius + 1e-12).astype(int)
    aggregate = lambda x: np.bincount(shell.ravel(),weights=x.ravel(),minlength=int(shell.max())+1)
    result = dict(reference_power=aggregate(et), prediction_power=aggregate(ep),
                  error_power=aggregate(ee), cross_power=aggregate(np.real(p*np.conj(t))),
                  reference_total=total, rel_l2=float(np.sqrt(ee.sum()/total)), bands={})
    masks = {'dc':radius==0, 'low':(radius>0)&(radius<=cutoffs[0]),
             'mid':(radius>cutoffs[0])&(radius<=cutoffs[1]), 'high':radius>cutoffs[1]}
    assert np.all(sum(m.astype(int) for m in masks.values())==1)
    for name,mask in masks.items():
        tr,pr,er = (float(x[mask].sum()) for x in (et,ep,ee))
        result['bands'][name] = dict(reference_energy=tr,prediction_energy=pr,error_energy=er,
            reference_fraction=tr/total,relative_error=float(np.sqrt(er/tr)) if tr>total*1e-14 else None,
            global_error_contribution=er/total,predicted_reference_energy_ratio=pr/tr if tr>total*1e-14 else None,
            mode_count=int(mask.sum()))
    assert np.isclose(sum(b['global_error_contribution'] for b in result['bands'].values()),result['rel_l2']**2,rtol=2e-12,atol=2e-18)
    return result


def filtered_field(field, kind, band, cutoffs=(8.,32.)):
    c = coefficients(field,kind)
    radius = radii(c.shape,kind)
    masks = {'low':radius<=cutoffs[0], 'high':radius>cutoffs[1]}
    return reconstruct(c*masks[band],kind)


def paired_bootstrap(values, seed=20260908, repetitions=4000):
    """Rows are physical examples; trailing dimensions are matched methods."""
    x=np.asarray(values,dtype=np.float64)
    assert len(x)>=2 and np.isfinite(x).all()
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),size=(repetitions,len(x)))].mean(axis=1)
    return np.quantile(means,[.025,.975],axis=0)


def self_check():
    rng=np.random.default_rng(42)
    checks=[]
    for kind in ['periodic_fft','cosine']:
        truth=rng.normal(size=(32,32));pred=truth+.1*rng.normal(size=truth.shape)
        r=spectral_record(pred,truth,kind,cutoffs=(4.,12.))
        assert np.allclose(reconstruct(coefficients(truth,kind),kind),truth,rtol=1e-12,atol=1e-12)
        assert np.isclose(r['rel_l2'],np.linalg.norm(pred-truth)/np.linalg.norm(truth),rtol=2e-12)
        # Pure mode: identify the exact frequency support, not just Parseval.
        c=np.zeros((32,32),dtype=complex if kind=='periodic_fft' else float)
        if kind=='periodic_fft': c[3,0]=c[-3,0]=1
        else: c[3,0]=1
        field=reconstruct(c,kind);rr=spectral_record(field,field,kind,cutoffs=(4.,12.))
        assert np.isclose(rr['bands']['low']['reference_fraction'],1.)
        assert np.linalg.norm(filtered_field(field,kind,'high',cutoffs=(4.,12.)))<1e-12
        checks.append(dict(transform=kind,parseval=True,inverse=True,pure_mode=True))
    return checks


if __name__=='__main__':
    import json
    print(json.dumps(self_check(),indent=2))
