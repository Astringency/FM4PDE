"""Independent sparse solves of the three static data-generator equations."""
from functools import lru_cache
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve, MatrixRankWarning
from sampling.pde_residuals import _darcy_spline_matrices_numpy


@lru_cache(maxsize=8)
def fixed_solver(pde, n, k=1):
    h = 1/(n-1)
    if pde == 'poisson':
        m = n-2
        line = sparse.diags([np.ones(m-1), -2*np.ones(m), np.ones(m-1)], [-1,0,1],format='csc')/h**2
    elif pde == 'helmholtz':
        m = n
        line = sparse.diags([np.ones(m-1), -2*np.ones(m), np.ones(m-1)], [-1,0,1],format='lil')/h**2
        line[0,:] = 0; line[0,0] = 1
        line[-1,:] = 0; line[-1,-1] = 1
        line = line.tocsc()
    else:
        raise ValueError(pde)
    matrix = sparse.kronsum(line,line,format='csc')
    if pde == 'helmholtz':
        matrix += k**2*sparse.eye(m*m,format='csc')
    return splu(matrix)


@lru_cache(maxsize=8)
def darcy_grids(n):
    to_nodes, from_cells_inverse = _darcy_spline_matrices_numpy(n)
    return to_nodes, np.linalg.inv(from_cells_inverse)


def solve_static(pde, a, *, k=1, binary=False):
    """Return physical u with the original fixed boundary equations.

    Do not clip raw coefficients. Non-elliptic inputs remain flagged even when
    their discrete linear system can be solved.
    """
    a = np.asarray(a,dtype=np.float64)
    assert a.ndim==2 and a.shape[0]==a.shape[1]
    n = len(a)
    if pde in ('poisson','helmholtz'):
        rhs = a.copy()
        rhs[0,:]=0; rhs[-1,:]=0; rhs[:,0]=0; rhs[:,-1]=0
        solver = fixed_solver(pde,n,k)
        if pde=='poisson':
            u = np.zeros_like(a)
            u[1:-1,1:-1] = solver.solve(rhs[1:-1,1:-1].reshape(-1)).reshape(n-2,n-2)
        else:
            u = solver.solve(rhs.reshape(-1)).reshape(n,n)
        return u, dict(nonpositive_fraction=0., minimum_edge_coefficient=None)
    assert pde=='darcy'
    if binary:
        a = np.where(a>=8.,12.,4.)
    to_nodes,to_cells = darcy_grids(n)
    coef = to_nodes @ a @ to_nodes.T
    center = coef[1:-1,1:-1]
    weights = [(.5*(center+neighbor)).reshape(-1) for neighbor in
               (coef[:-2,1:-1],coef[2:,1:-1],coef[1:-1,:-2],coef[1:-1,2:])]
    m=n-2; ids=np.arange(m*m).reshape(m,m)
    row=[ids.reshape(-1)]; col=[ids.reshape(-1)]; values=[sum(weights)]
    for w,dr,dc in zip(weights,(-1,1,0,0),(0,0,-1,1)):
        rr,cc=np.indices((m,m)); nr,nc=rr+dr,cc+dc
        valid=(nr>=0)&(nr<m)&(nc>=0)&(nc<m)
        row.append(ids[valid]); col.append(ids[nr[valid],nc[valid]])
        values.append(-w.reshape(m,m)[valid])
    matrix=sparse.coo_matrix((np.concatenate(values),(np.concatenate(row),np.concatenate(col))),shape=(m*m,m*m)).tocsc()*(n-1)**2
    with warnings.catch_warnings():
        warnings.simplefilter('error',MatrixRankWarning)
        interior=spsolve(matrix,np.ones(m*m))
    if not np.isfinite(interior).all():
        raise ValueError('Nonfinite Darcy solve')
    nodal=np.zeros_like(a); nodal[1:-1,1:-1]=interior.reshape(m,m)
    return to_cells@nodal@to_cells.T, dict(nonpositive_fraction=float(np.mean(a<=0)),
        minimum_edge_coefficient=float(min(w.min() for w in weights)))
