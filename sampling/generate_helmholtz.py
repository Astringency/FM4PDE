from tqdm import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
# import dnnlib
import torch.nn.functional as F
# from torch_utils import distributed as dist
import scipy.io

from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from models.model_configs import instantiate_model

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

def load_fm4pde(pde_path, pde_type, device):
    pde_models = torch.load(pde_path, weights_only=False, map_location=device)
    fm4pde = instantiate_model(architechture=pde_type, is_discrete=False, use_ema=False)
    fm4pde.load_state_dict(pde_models['model'])
    return WrappedModel(fm4pde).to(device)

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

def get_helmholtz_loss(a, u, a_GT, u_GT, a_mask, u_mask, device=torch.device('cuda')):
    """Return the loss of the Helmholtz equation and the observation loss."""
    S = u.size(2)
    h = 1 / (S - 1)
    a = a.view(1, 1, S, S)
    u_padded = torch.nn.functional.pad(u, (1, 1, 1, 1), 'constant', 0)
    d2u = (u_padded[:, :, :-2, 1:-1] + u_padded[:, :, 2:, 1:-1] +
           u_padded[:, :, 1:-1, :-2] + u_padded[:, :, 1:-1, 2:] - 4 * u[:, :, :, :]) / h**2
    pde_loss = d2u + u - a
    pde_loss = pde_loss.squeeze()
    
    a_GT = a_GT.view(1, 1, S, S)
    u_GT = u_GT.view(1, 1, S, S)
    observation_loss_a = (a - a_GT).squeeze()
    observation_loss_a = observation_loss_a * a_mask  
    observation_loss_u = (u - u_GT).squeeze()
    observation_loss_u = observation_loss_u * u_mask
    
    return pde_loss.to(device), observation_loss_a.to(device), observation_loss_u.to(device)
    

def generate_helmholtz(config):
    """Generate Helmholtz equation."""
    ############################ Load data and network ############################
    pde_type = config['data']['name']
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    data = scipy.io.loadmat(datapath)
    img_resolution = config['data']['img_resolution']
    img_channels = config['data']['img_channels']
    
    a_GT = data['f_data'][offset, :, :]
    a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
    u_GT = data['psi_data'][offset, :, :]
    u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)
    
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
    
    x_init.requires_grad = True
    
    if config['generate']['process']:
        sol = []
    
    if config['generate']['guide']:
        known_index_a = random_index(500, 128, seed=1, device=device)
        known_index_u = random_index(500, 128, seed=0, device=device)
        x_next = x_init
        with tqdm(total=len(T), desc=f"Sampling", ncols=100) as pbar:
            for i in range(1, len(T)):
                t = torch.tensor([T[i-1], T[i]])
                x_cur = x_next.detach().clone()
                x_cur.requires_grad = True
                
                if config['generate']['process']:
                    x_N = solver.sample(time_grid=t, x_init=x_cur, method=method, 
                                        step_size=step_size, enable_grad = True, 
                                        return_intermediates=True)
                    
                    x_next = (x_N[-1] + 1.0) / 2.0
                else:
                    x_N = solver.sample(time_grid=t, x_init=x_cur, method=method, 
                                        step_size=step_size, enable_grad = True)
                    
                    x_next = (x_N + 1.0) / 2.0
                    
                a_N = x_next[:,0,:,:].unsqueeze(0)
                u_N = x_next[:,1,:,:].unsqueeze(0)
                a_N = (a_N*2.15).to(torch.float64)
                u_N = (u_N*0.028).to(torch.float64)
                
                # Compute the loss
                pde_loss, observation_loss_a, observation_loss_u = get_helmholtz_loss(a_N, u_N, a_GT, u_GT, known_index_a, known_index_u, device=device)
                L_pde = torch.norm(pde_loss, 2)/(128*128)
                L_obs_a = torch.norm(observation_loss_a, 2)/500
                L_obs_u = torch.norm(observation_loss_u, 2)/500
                grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
                grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
                grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
                zeta_obs_a = config['generate']['zeta_obs_a']
                zeta_obs_u = config['generate']['zeta_obs_u']
                zeta_pde = config['generate']['zeta_pde']
                
                if i <= 0.8 * num_steps:
                    x_next = x_next - zeta_obs_a * grad_x_cur_obs_a - zeta_obs_u * grad_x_cur_obs_u
                else:
                    x_next = x_next - 0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde
                
                if config['generate']['process']:
                    sol.append(x_next)
                
                x_next = x_next * 2.0 - 1.0
                
                pbar.set_postfix(Step=i+1, LossPDE=L_pde, LossObs=(L_obs_a.item(), L_obs_u.item()))
                pbar.update(1)
    else:
        if config['generate']['process']:
            sol = solver.sample(time_grid=T, x_init=x_init, method=method, step_size=step_size, return_intermediates=True)
            x_next = sol[-1]
        else:
            sol = solver.sample(time_grid=T, x_init=x_init, method=method, step_size=step_size)
            x_next = sol
        
    ############################ Save the data ############################
    x_final = (x_next + 1.0) / 2.0
    a_final = x_final[:,0,:,:].unsqueeze(0)
    u_final = x_final[:,1,:,:].unsqueeze(0)
    a_final = (a_final*2.15).to(torch.float64)
    u_final = (u_final*0.028).to(torch.float64)
    relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
    relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
    print(f'Relative error of a: {relative_error_a}')
    print(f'Relative error of u: {relative_error_u}')
    a_final = a_final.detach().cpu().numpy()
    u_final = u_final.detach().cpu().numpy()
    scipy.io.savemat(f'{config['output']['output_path']}/{pde_type}_results.mat', {'f': a_final, 'psi': u_final})
    if config['generate']['process']:
        torch.save(sol, f"{config['output']['output_path']}/{pde_type}_sample_path.pth")
    
    if config['output']['return']:
        print('Done.')
        if config['generate']['process']:
            return {"sol": sol, "f_final": a_final, "psi_final": u_final}
        else:
            return {"f_final": a_final, "psi_final": u_final}
    else:
        print('Done.')