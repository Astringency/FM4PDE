import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
# import dnnlib
import torch.nn.functional as F
# from torch_utils import distributed as dist
import scipy.io

from flowmatching.solver import Solver, ODESolver
from flowmatching.utils import ModelWrapper

from models.model_configs import instantiate_model

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

def load_fm4pde(pde_path, pde_type, device):
    pde_models = torch.load(pde_path, weights_only=False, map_location=device)
    fm4pde = instantiate_model(architechture=pde_type, is_discrete=False, use_ema=False)
    fm4pde.load_state_dict(pde_models['model'])
    return WrappedModel(fm4pde).to(device)

def random_sensor(k, grid_size, seed=0, device=torch.device('cuda')):
    """Return a index list with k sensors randomly placed in a grid of size [grid_size, grid_size]."""
    torch.manual_seed(seed)
    index = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)
    known_index = torch.randperm(grid_size, device=device)[:k]
    for i in known_index:
        index[:, i]=1
    return index

def get_burger_loss(u, u_GT, mask, device=torch.device('cuda')):
    """Return the loss of the Burgers' equation and the observation loss."""
    u = u.view(1, 1, 128, 128)
    u_GT = u_GT.view(1, 1, 128, 128)
    deriv_t = torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2 
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2 
    u_t = F.conv2d(u, deriv_t, padding=(1, 0)) 
    u_x = F.conv2d(u, deriv_x, padding=(0, 1)) 
    u_xx = F.conv2d(u_x, deriv_x, padding=(0, 1))

    pde_loss = u_t + u * u_x - 0.01 * u_xx
    pde_loss = pde_loss.squeeze()
    observation_loss = u - u_GT
    observation_loss = observation_loss.squeeze()
    observation_loss = observation_loss * mask
    return pde_loss, observation_loss

def generate_burgers(config):
    """Generate Burgers' equation."""
    ############################ Load data and network ############################
    pde_type = config['data']['name']
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    data = scipy.io.loadmat(datapath)
    init_state = data['input']
    init_state = torch.tensor(init_state, dtype=torch.float64, device=device)
    ground_truth = data['output'][offset, :, :]
    ground_truth = torch.tensor(ground_truth, dtype=torch.float64, device=device)
    
    img_resolution = config['data']['img_resolution']
    img_channels = config['data']['img_channels']
    
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
        selected_index = random_sensor(5, 128)
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
                    
                x_N = (x_next * 1.415).to(torch.float64)
                
                # Compute the loss
                pde_loss, observation_loss = get_burger_loss(x_N, ground_truth, selected_index, device)
                L_pde = torch.norm(pde_loss, 2)/(128*128)
                L_obs = torch.norm(observation_loss, 2)/(128*5)
                grad_x_cur_obs = torch.autograd.grad(outputs=L_obs, inputs=x_cur, retain_graph=True)[0]
                grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
                zeta_obs = config['generate']['zeta_obs']
                zeta_pde = config['generate']['zeta_pde']
                if i <= 0.8 * num_steps:
                    x_next = x_next - zeta_obs * grad_x_cur_obs
                else:
                    x_next = x_next - zeta_obs / 10 * grad_x_cur_obs - zeta_pde * grad_x_cur_pde
                
                if config['generate']['process']:
                    sol.append(x_next)
                
                x_next = x_next * 2.0 - 1.0
                
                pbar.set_postfix(Step=i+1, LossPDE=L_pde, LossObs=L_obs)
                pbar.update(1)
    else:
        if config['generate']['process']:
            sol = solver.sample(time_grid=T, x_init=x_init, method=method, step_size=step_size, return_intermediates=True)
            x_next = sol[-1]
        else:
            sol = solver.sample(time_grid=T, x_init=x_init, method=method, step_size=step_size)
            x_next = sol
        
    ############################ Save the data ############################
    x_final = (((x_next + 1.0) / 2.0) * 1.415).to(torch.float64)
    relative_error = torch.norm(x_final - ground_truth, 2)/torch.norm(ground_truth, 2)
    print(f'Relative error: {relative_error}')
    x_final = x_final.to('cpu').detach().numpy()
    scipy.io.savemat(f'{config['output']['output_path']}/{pde_type}_results.mat', {'x': x_final})
    if config['generate']['process']:
        torch.save(sol, f"{config['output']['output_path']}/{pde_type}_sample_path.pth")
    
    if config['output']['return']:
        print('Done.')
        if config['generate']['process']:
            return {"sol": sol, "x_final": x_final}
        else:
            return {"x_final": x_final}
    else:
        print('Done.')