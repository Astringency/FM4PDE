from tqdm import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
# import dnnlib
import torch.nn.functional as F
# from torch_utils import distributed as dist
import scipy.io

# from flowmatching.solver import Solver, ODESolver
# from flowmatching.utils import ModelWrapper
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from models.model_configs import instantiate_model

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return  self.model(x, t, **extras)

def load_fm4pde(pde_path, pde_type, device, wrap=True):
    pde_models = torch.load(pde_path, weights_only=False, map_location=device)
    fm4pde = instantiate_model(architechture=pde_type, is_discrete=False, use_ema=False)
    fm4pde.load_state_dict(pde_models['model'])
    if wrap:
        return WrappedModel(fm4pde).to(device)
    else:
        return fm4pde.to(device)

def getscheduler(t, scheduler = "CondOT", n = 2, beta_min = 1.0, beta_max = 2.0):
    if scheduler == "CondOT":
        alpha_t = t
        sigma_t = 1 - t
        d_alpha_t = torch.ones_like(t)
        d_sigma_t = -torch.ones_like(t)
    elif scheduler == "PolynomialConvex":
        alpha_t = t**n
        sigma_t = 1 - t**n
        d_alpha_t = n * (t ** (n - 1))
        d_sigma_t = -n * (t ** (n - 1))
    elif scheduler == "VP":
        b = beta_min
        B = beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = -(1 - t) * (B - b) - b
        alpha_t = torch.exp(-0.5 * T),
        sigma_t = torch.sqrt(1 - torch.exp(-T)),
        d_alpha_t = -0.5 * dT * torch.exp(-0.5 * T),
        d_sigma_t = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T)),
    elif scheduler == "LVP":
        alpha_t = t
        sigma_t = (1 - t**2) ** 0.5
        d_alpha_t = torch.ones_like(t)
        d_sigma_t = -t / (1 - t**2) ** 0.5
    elif scheduler == "Cosine":
        alpha_t = torch.sin(torch.pi / 2 * t)
        sigma_t = torch.cos(torch.pi / 2 * t)
        d_alpha_t = torch.pi / 2 * torch.cos(torch.pi / 2 * t)
        d_sigma_t = -torch.pi / 2 * torch.sin(torch.pi / 2 * t)
    else:
        alpha_t = 0
        sigma_t = 0
        d_alpha_t = 0
        d_sigma_t = 0
        
    return alpha_t, sigma_t, d_alpha_t, d_sigma_t

def getaffine(alpha_t, sigma_t, d_alpha_t, d_sigma_t, training = "velocity"):
    if training == "velocity":
        a_t = d_alpha_t / alpha_t
        b_t = -(d_sigma_t * sigma_t * alpha_t - d_alpha_t * (sigma_t**2)) / alpha_t
    elif training == "x1":
        a_t = 1 / alpha_t
        b_t = (sigma_t**2) / alpha_t
    elif training == "x0":
        a_t = 0
        b_t = -sigma_t
    else:
        a_t = 0
        b_t = 0
        
    return a_t, b_t

def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size] grid.'''
    np.random.seed(seed)
    indices = np.random.choice(grid_size**2, k, replace=False)
    indices_2d = np.unravel_index(indices, (grid_size, grid_size))
    indices_list = list(zip(indices_2d[0], indices_2d[1]))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask

def get_poisson_loss(a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
    """Return the loss of the Poisson equation and the observation loss."""
    S = u.size(2)
    h = 1 / (S - 1)
    a = a.view(1, 1, S, S)
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
    d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
           u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / h**2
    pde_loss = d2u - a
    pde_loss = pde_loss.squeeze()
    pde_loss[0, :] = 0
    pde_loss[-1, :] = 0
    pde_loss[:, 0] = 0
    pde_loss[:, -1] = 0
    
    a_GT = a_GT.view(1, 1, S, S)
    u_GT = u_GT.view(1, 1, S, S)
    observation_loss_a = (a - a_GT).squeeze()
    observation_loss_a = observation_loss_a * a_mask  
    observation_loss_u = (u - u_GT).squeeze()
    observation_loss_u = observation_loss_u * u_mask
    
    return pde_loss, observation_loss_a, observation_loss_u

def generate_poisson(config):
    """Generate Poisson equation."""
    ############################ Load data and network ############################
    obs_size = config['data']['obs_size']
    pde_type = config['data']['name']
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    data = scipy.io.loadmat(datapath)
    img_resolution = config['data']['img_resolution']
    img_channels = config['data']['img_channels']
    
    a_GT = data['f_data'][offset, :, :]
    a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
    u_GT = data['phi_data'][offset, :, :]
    u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)
    data_GT = torch.stack((a_GT, u_GT), dim=0)
    
    batch_size = config['generate']['batch_size']
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    
    net = load_fm4pde(config['test']['pre-trained'], pde_type, device)
    
    ############################ Sample the data ############################
    print(f'Generating {batch_size} samples...')
    
    step_size = config['generate']['step_size']
    num_steps = config['test']['num_steps']
    method = config['generate']['method']
    T = torch.linspace(0, 1, num_steps).to(device=device)
    x_init = torch.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)

    solver = ODESolver(velocity_model=net)
    
    loss = {'pde': [], 'obs_a': [], 'obs_u': [], 'global_a': [], 'global_u': []}
    
    x_init.requires_grad = True
    
    sol = []
        
    if config['generate']['full']:
        known_index_a = torch.ones((img_resolution, img_resolution), dtype=torch.float32, device=device)
        known_index_u = torch.ones((img_resolution, img_resolution), dtype=torch.float32, device=device)
    else:
        known_index_a = random_index(obs_size, img_resolution, seed=1, device=device)
        known_index_u = random_index(obs_size, img_resolution, seed=0, device=device)
    
    x_next = x_init
    
    with tqdm(total=len(T), desc=f"Sampling", ncols=200, dynamic_ncols=True, leave=True) as pbar:
        for i in range(1, len(T)):
            x_cur = x_next.detach().clone()
            x_cur.requires_grad = True
            
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = getscheduler(T[i], scheduler=config['generate']['scheduler'])
            a_t, b_t = getaffine(alpha_t, sigma_t, d_alpha_t, d_sigma_t, training=config['generate']['training'])
            
            if config['generate']['samplemode'] == 'deterministic':
                t = torch.tensor([T[i-1], T[i]])
                if config['generate']['process']:
                    x_N = solver.sample(time_grid=t, x_init=x_cur, method=method, step_size=step_size, enable_grad = True, return_intermediates=True)
                    
                    x_next = (x_N[-1] + 1.0) / 2.0
                else:
                    x_N = solver.sample(time_grid=t, x_init=x_cur, method=method, step_size=step_size, enable_grad = True)
                    
                    x_next = (x_N + 1.0) / 2.0
                
                a_N = x_next[:,0,:,:].unsqueeze(0)
                u_N = x_next[:,1,:,:].unsqueeze(0)
                a_N = (a_N*2.5).to(torch.float64)
                u_N = (u_N/36.5).to(torch.float64)
            elif config['generate']['samplemode'] == 'stochastic' and config['generate']['stochastic_t'] == 1:
                x_N = x_cur + (1 - T[i-1]) * net(x_cur, T[i-1])

                x_next = (1 - T[i]) * torch.randn([batch_size, img_channels, img_resolution, img_resolution], device=device) + T[i] * x_N
                x_next = x_next.unsqueeze(0)
                x_next = (x_next[-1] + 1.0) / 2.0

                x_N = x_N.unsqueeze(0)
                x_N = (x_N[-1] + 1.0) / 2.0
                
                a_N = x_N[:,0,:,:].unsqueeze(0)
                u_N = x_N[:,1,:,:].unsqueeze(0)
                a_N = (a_N*2.5).to(torch.float64)
                u_N = (u_N/36.5).to(torch.float64)
            else:
                if i <= config['generate']['stochastic_t'] * num_steps:
                    x_N = x_cur + (1 - T[i-1]) * net(x_cur, T[i-1])

                    x_next = (1 - T[i]) * torch.randn([batch_size, img_channels, img_resolution, img_resolution], device=device) + T[i] * x_N
                    x_next = x_next.unsqueeze(0)
                    x_next = (x_next[-1] + 1.0) / 2.0

                    x_N = x_N.unsqueeze(0)
                    x_N = (x_N[-1] + 1.0) / 2.0
                    
                    a_N = x_N[:,0,:,:].unsqueeze(0)
                    u_N = x_N[:,1,:,:].unsqueeze(0)
                    a_N = (a_N*2.5).to(torch.float64)
                    u_N = (u_N/36.5).to(torch.float64)
                else:
                    t = torch.tensor([T[i-1], T[i]])
                    if config['generate']['process']:
                        x_N = solver.sample(time_grid=t, x_init=x_cur, method=method, step_size=step_size, enable_grad = True, return_intermediates=True)
                        
                        x_next = (x_N[-1] + 1.0) / 2.0
                    else:
                        x_N = solver.sample(time_grid=t, x_init=x_cur, method=method, step_size=step_size, enable_grad = True)
                        
                        x_next = (x_N + 1.0) / 2.0
                    
                    a_N = x_next[:,0,:,:].unsqueeze(0)
                    u_N = x_next[:,1,:,:].unsqueeze(0)
                    a_N = (a_N*2.5).to(torch.float64)
                    u_N = (u_N/36.5).to(torch.float64)
            
            if config['generate']['guide']:
                # Compute the loss
                pde_loss, observation_loss_a, observation_loss_u = get_poisson_loss(a_N, u_N, a_GT, u_GT, known_index_a, known_index_u, device=device)
                L_pde = torch.norm(pde_loss, 2)/(img_resolution*img_resolution)
                L_obs_a = torch.norm(observation_loss_a, 2)/obs_size
                L_obs_u = torch.norm(observation_loss_u, 2)/obs_size
                grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
                grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
                grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
                zeta_obs_a = config['generate']['zeta_obs_a']
                zeta_obs_u = config['generate']['zeta_obs_u']
                zeta_pde = config['generate']['zeta_pde']

                if config['generate']['samplemode'] == "deterministic":
                    if i <= 0.8 * num_steps:
                        x_next = x_next - b_t * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) * (T[i] - T[i-1])
                    else:
                        x_next = x_next - b_t * (0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) + zeta_pde * grad_x_cur_pde) * (T[i] - T[i-1])

                elif config['generate']['samplemode'] == "stochastic":
                    if i <= config['generate']['stochastic_t'] * num_steps:
                        if i <= 0.8 * num_steps:
                            x_next = x_next - (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u)
                        else:
                            x_next = x_next - 0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde
                        
                    else:
                        if i <= 0.8 * num_steps:
                            x_next = x_next - b_t * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) * (T[i] - T[i-1])
                        else:
                            x_next = x_next - b_t * (0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) + zeta_pde * grad_x_cur_pde) * (T[i] - T[i-1])

                    if config['generate']['process']:
                        sol.append(x_next)

                else:
                    print("Without Guidance")


                a_eval = x_next[:,0,:,:].unsqueeze(0)
                u_eval = x_next[:,1,:,:].unsqueeze(0)
                a_eval = (a_eval*2.5).to(torch.float64)
                u_eval = (u_eval/36.5).to(torch.float64)
                re_a_eval = torch.norm(a_eval - a_GT, 2) / torch.norm(a_GT, 2)
                re_u_eval = torch.norm(u_eval - u_GT, 2) / torch.norm(u_GT, 2)
                
                loss['pde'].append(L_pde.item())
                loss['obs_a'].append(L_obs_a.item())
                loss['obs_u'].append(L_obs_u.item())
                loss['global_a'].append(re_a_eval.item())
                loss['global_u'].append(re_u_eval.item())
            
                x_next = x_next * 2.0 - 1.0
            
                postfix = {
                    "Step": i + 1,
                    "LossPDE": round(L_pde.item(), 4),
                    "LossObsA": round(L_obs_a.item(), 4),
                    "LossObsU": round(L_obs_u.item(), 4)
                }
                
            else:
                x_next = x_next * 2.0 - 1.0
                
                postfix = {
                    "Step": i + 1
                }
                
            pbar.set_postfix(postfix)
            pbar.update(1)
        
    ############################ Save the data ############################
    x_final = (x_next + 1.0) / 2.0
    a_final = x_final[:,0,:,:].unsqueeze(0)
    u_final = x_final[:,1,:,:].unsqueeze(0)
    a_final = (a_final*2.5).to(torch.float64)
    u_final = (u_final/36.5).to(torch.float64)
    relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
    relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
    print(f'Relative error of a: {relative_error_a}')
    print(f'Relative error of u: {relative_error_u}')
    a_final = a_final.detach().cpu().numpy()
    u_final = u_final.detach().cpu().numpy()
    scipy.io.savemat(f'{config['output']['output_path']}/{pde_type}_results.mat', {'f': a_final, 'phi': u_final})
    if config['generate']['process']:
        torch.save(sol, f"{config['output']['output_path']}/{pde_type}_sample_path.pth")
    
    if config['output']['return']:
        print('Done.')
        if config['generate']['process']:
            return {"sol": sol, "f_final": a_final, "phi_final": u_final, "loss": loss, "known_index_a": known_index_a, "known_index_u": known_index_a}
        else:
            return {"f_final": a_final, "phi_final": u_final, "loss": loss, "known_index_a": known_index_a, "known_index_u": known_index_a}
    else:
        print('Done.')
