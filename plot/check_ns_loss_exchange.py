"""Independent stencil, Fourier-mode, gradient and extraction checks."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from sampling.config import load_config
from sampling.pde_residuals import compute_pde_residual
from ns_loss_exchange import diffusion_spatial_residual, diffusion_pde_loss, fm_pde_loss, build_native_diffusion


def main():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    cfg = load_config(ROOT/'configs/main/both/nsnonbounded.yaml')
    n = 16
    x = torch.arange(n,dtype=torch.float64)/n
    wave = torch.sin(2*torch.pi*3*x).view(1,1,n,1).expand(1,1,n,n).clone()
    a, u = .7*wave, .8*wave
    # Construct the sine factor directly in double precision.
    expected_diff = .8*torch.sin(torch.tensor(2*torch.pi*3/n,dtype=torch.float64))*torch.cos(2*torch.pi*3*x).view(1,1,n,1).expand_as(u).clone()
    expected_diff[..., (0,-1),:] = 0
    expected_diff[..., :,(0,-1)] = 0
    d = diffusion_spatial_residual(a,u)
    assert torch.allclose(d,expected_diff,atol=2e-15,rtol=2e-14)
    params = dict(nu=.001,T=1.,forcing=0.,enforce_boundary_conditions=False)
    r = compute_pde_residual('nsnonbounded',a,u,pde_params=params,residual_mode='endpoint_secant')
    expected_fm = (.1 + .001*(2*torch.pi*3)**2*.75)*wave
    assert torch.allclose(r.residual,expected_fm,atol=2e-14,rtol=2e-13)
    loss = fm_pde_loss(a,u,cfg,params)
    assert torch.allclose(loss,expected_fm.square().mean(),atol=2e-14,rtol=2e-13)
    aa = torch.randn(1,1,n,n,dtype=torch.float64,requires_grad=True)*.1
    uu = torch.randn_like(aa,requires_grad=True)*.1
    direction_a, direction_u = torch.randn_like(aa), torch.randn_like(uu)
    checks = {}
    for name, func in [('diffusion',diffusion_pde_loss),('fm',lambda a,u:fm_pde_loss(a,u,cfg,params))]:
        val=func(aa,uu)
        grads=torch.autograd.grad(val,(aa,uu),allow_unused=True)
        exact=sum((g*v).sum() for g,v in zip(grads,(direction_a,direction_u)) if g is not None)
        eps=1e-6
        numerical=(func(aa+eps*direction_a,uu+eps*direction_u)-func(aa-eps*direction_a,uu-eps*direction_u))/(2*eps)
        assert torch.allclose(exact,numerical,atol=1e-9,rtol=2e-7),(name,exact,numerical)
        checks[name+'_directional_gradient_abs_error']=float((exact-numerical).abs().detach())
    diffusion_root=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT.parent/'DiffusionPDE'
    native,source=build_native_diffusion(diffusion_root)
    reference,_=build_native_diffusion(diffusion_root,diagnostics=True)
    swapped,_=build_native_diffusion(diffusion_root,exchange=True)
    helper=native.__globals__['get_ns_nonbounded_loss']
    native_r,_,_=helper(aa,uu,aa.detach(),uu.detach(),torch.ones(n,n),torch.ones(n,n),n,device='cpu')
    native_loss=native_r.norm()/(n*n)
    assert torch.equal(native_loss,diffusion_pde_loss(aa,uu))
    class TinyNet:
        img_channels=2
        img_resolution=n
        label_dim=0
        sigma_min=.002
        sigma_max=80.
        def round_sigma(self,sigma):return torch.as_tensor(sigma)
        def __call__(self,x,sigma,class_labels=None):return .2*x/(sigma+1)
    conf={'data':{'obs_size':24},'test':{'iterations':10},'generate':dict(device='cpu',seed=4,batch_size=1,sigma_min=.01,sigma_max=1.,rho=7,guide=True,zeta_obs_a=.001,zeta_obs_u=.001,zeta_pde=.0001)}
    mask=torch.zeros(n,n,dtype=torch.float64);mask.flatten()[:24]=1
    args=(conf,TinyNet(),aa.detach(),uu.detach(),mask,mask,lambda a,u:fm_pde_loss(a,u,cfg,params))
    base=native(*args,[]);ref=reference(*args,[])
    assert all(torch.equal(a,b) for a,b in zip(base,ref))
    conf['generate']['zeta_pde']=0.
    base=native(*args,[]);sw=swapped(*args,[])
    assert all(torch.equal(a,b) for a,b in zip(base,sw))
    assert all(a.dtype==torch.float64 for a in base)
    checks.update(native_scalar_exact=True,analytic_stencil_pass=True,analytic_ns_mode_pass=True,
                  native_reference_prediction_exact=True,zero_weight_exchange_exact=True,native_float64_preserved=True)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
