# Misc tools
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import torch
import scipy
import h5py
import time
from datetime import datetime
from datetime import timedelta

# Process bar
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Flow matching & model wapper
from flow_matching.utils import ModelWrapper
from models.model_configs import instantiate_model

# Data load & transform
from data.transform_old import PDEtransform as oldPDEtransform
from data.transform import PDEtransform

# Sampler
from sampling import generate_pde
from sampling import sampler

# Evaluation tools
from plot.plot import *

parser = argparse.ArgumentParser(description="FM4PDE sample parameters.")

parser.add_argument("--pdetype", type=str, default="poisson", help="Which PDE to slove.")
parser.add_argument("--problem", type=str, default="both", help="Forward or inverse problem to solve. If both, then reconstruct the pair of global truth from the sparsity observation.")
parser.add_argument("--mode", type=str, default="sparse", help="Full or sparse observations guide mainly for forward and inverse problems.")
parser.add_argument("--dt_sampler", type=str, default="old", help="For the deterministic sampler, use new or old.")
parser.add_argument("--config", type=str, default=".", help="Config used to solve the specified PDE.")
parser.add_argument("--pdeguide", type=bool, default=True, help="Whether to use pde guide.")
parser.add_argument("--lr_decay", type=bool, default=True, help="Whether to decay the learning rate.")
parser.add_argument("--freq_decay", type=int, default=100, help="Learning rate decay item.")
parser.add_argument("--hybrid", type=bool, default=False, help="Whether to use Hybrid sampler.")
parser.add_argument("--remark", type=str, default="", help="Sample setup remark.")

# Adjust Configs
parser.add_argument("--guide", type=str, default=None, help="Whether generate with guidence.")
parser.add_argument("--num_steps", type=int, default=0, help="Sample steps.")
parser.add_argument("--batch", type=int, default=1, help="Sample item.")
parser.add_argument("--sampler", type=str, default=None, help="Stochastic or Deterministic sampler.")


def get_config(config_path: str) -> dict:
    """
    Get configs.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: 'config.yaml' file not found. Please make sure the file exists in the correct location.")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML: {exc}")
        return {}
    except Exception as e:
        print(f"An unknown error occurred: {e}")
        return {}

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

def progress_metrics(iters, coef_loss, sol_loss, pde_loss):
    table = Table.grid()
    table.add_row(f"[cyan]Epoch:[/] {iters}")
    table.add_row(f"[green]Loss(coef):[/] {coef_loss}")
    table.add_row(f"[green]Loss(sol):[/] {sol_loss}")
    table.add_row(f"[green]Loss(pde):[/] {pde_loss}")

    return Panel(table, title="Sampling Eval Metrics", border_style="blue")


def main(args, batch):
    # >>>> Misc <<<< #
    print("Setup")
    # Get generate configs
    config = get_config(args.config)

    # Adjust configs
    ## step size
    if args.num_steps == 0:
        config['generate']['num_steps'] = config['generate']['num_steps'] 
        config['generate']['step_size'] = config['generate']['step_size'] 
    else:
        config['generate']['num_steps'] = args.num_steps
        config['generate']['step_size'] = 1 / args.num_steps

    ## guidance
    if args.guide:
        config['generate']['guide'] = args.guide
    else:
        config['generate']['guide'] = config['generate']['guide']

    ## problem
    if args.mode == "sparse":
        mode = False
    elif args.mode == "full":
        mode = True
    else:
        print(f">>> Unknown {args.mode} <<<")
        sys.exit()

    if args.problem == 'forward':
        config['generate']['zeta_obs_u'] = 0
        config['generate']['full'] = mode
    else:
        config['generate']['zeta_obs_u'] = config['generate']['zeta_obs_u'] 

    if args.problem == 'inverse':
        config['generate']['zeta_obs_a'] = 0
        config['generate']['full'] = mode
    else:
        config['generate']['zeta_obs_a'] = config['generate']['zeta_obs_a']

    if args.problem == "both":
        config['generate']['zeta_obs_a'] = config['generate']['zeta_obs_a']
        config['generate']['zeta_obs_u'] = config['generate']['zeta_obs_u']

    ## sampler
    if args.sampler:
        config['generate']['samplemode'] = args.sampler
    else:
        config['generate']['samplemode'] = config['generate']['samplemode']

    if args.hybrid == False and config['generate']['samplemode'] == 'deterministic':
        config['generate']['stochastic_t'] = 0
    else:
        config['generate']['stochastic_t'] = config['generate']['stochastic_t']

    # Setup file name
    if config['generate']['guide']:
        if_guide = "wG"
    else:
        if_guide = "woG"

    setup_file_name = f"_{config['data']['offset']}_obs({config['data']['obs_size']})_zeta({config['generate']['zeta_obs_a']},{config['generate']['zeta_obs_u']},{config['generate']['zeta_pde']})_step({config['generate']['num_steps']},{config['generate']['step_size']})_{config['generate']['method']}_{if_guide}_{config['generate']['samplemode']}_t({config['generate']['stochastic_t']},{config['generate']['obsguide_t']})_decay({config['generate']['obsguide_decay']})_{config['data']['remark']}"

    # print(setup_file_name)
    # sys.exit()

    # Get functions for bounded Navier-Stoker equations
    random_index_and_cylinder = generate_pde.random_index_and_cylinder
    cylinder_index = generate_pde.cylinder_index

    # Get functions for all pdes
    random_index = generate_pde.random_index
    random_sensor = generate_pde.random_sensor

    get_pde_loss = generate_pde.getPDEloss(args.pdetype).func

    # Get coef and solution variable name
    coef_name = config['data']['coef']
    sol_name = config['data']['solution']

    # >>>> Load data, network and sampler <<<< #
    obs_size = config['data']['obs_size']
    pde_type = config['data']['name']
    
    img_resolution = config['data']['img_resolution']
    img_channels = config['data']['img_channels']
    datapath = config['data']['datapath']
    offset = config['data']['offset'] + batch
    device = config['generate']['device']
    batch_size = config['generate']['batch_size']
    seed = config['generate']['seed']
    
    if config['data']['loadby'] == "scipy":
        data = {}
        data = scipy.io.loadmat(datapath)
    elif config['data']['loadby'] == "h5py":
        data = {}
        with h5py.File(datapath, 'r') as file:
            for key in [coef_name, sol_name]:
                data[key] = file[key][:] # type: ignore
    elif config['data']['loadby'] == "numpy":
        data = {}
        data = np.load(datapath)
    elif config['data']['loadby'] == "swe":
        data = {}
        with h5py.File(datapath, 'r') as file:
            h0 = np.expand_dims(file[list(file.keys())[offset]]['data']['h'][0, :, :, 0], axis = 0) # type: ignore
            h = np.expand_dims(file[list(file.keys())[offset]]['data']['h'][-1, :, :, 0], axis = 0) # type: ignore
            hu0 = np.expand_dims(file[list(file.keys())[offset]]['data']['hu'][0, :, :, 0], axis = 0) # type: ignore
            hu = np.expand_dims(file[list(file.keys())[offset]]['data']['hu'][-1, :, :, 0], axis = 0) # type: ignore
            hv0 = np.expand_dims(file[list(file.keys())[offset]]['data']['hv'][0, :, :, 0], axis = 0) # type: ignore
            hv = np.expand_dims(file[list(file.keys())[offset]]['data']['hv'][-1, :, :, 0], axis = 0) # type: ignore

            data[coef_name] = np.stack([h0, hu0, hv0], axis = 1)
            data[sol_name] = np.stack([h, hu, hv], axis = 1)
    elif config['data']['loadby'] == "rd":
        data = {}
        with h5py.File(datapath, 'r') as file:
            data[coef_name] = file[list(file.keys())[offset]]['data'][:] #type: ignore
            data[sol_name] = file[list(file.keys())[offset]]['data'][:] #type: ignore
    else:
        data = {}
        print("Error to load the dataset.")
        sys.exit()

    # print(data[coef_name].shape, data[sol_name].shape)
    # sys.exit()

    if pde_type == "darcy":
        a_GT = data[coef_name][:, :, offset]
        a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
        u_GT = data[sol_name][:, :, offset]
        u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)
    elif pde_type in ['nsbounded', 'nsnonbounded']:
        a_GT = data[coef_name][offset, :, :]
        a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
        u_GT = data[sol_name][offset, :, :, -1]
        u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)
    elif pde_type in ["shallow_water"]:
        a_GT = data[coef_name][0, :, :, :]
        a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
        u_GT = data[sol_name][0, :, :, :]
        u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)
    elif pde_type == "reaction_diffusion":
        a_GT = data[coef_name][50, :, :, :]
        a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device).permute(2, 0, 1)
        u_GT = data[sol_name][-1, :, :, :]
        u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device).permute(2, 0, 1)
    else:
        a_GT = data[coef_name][offset, :, :]
        a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)
        u_GT = data[sol_name][offset, :, :]
        u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)

    if len(a_GT.shape) == 2:
        data_GT = torch.stack((a_GT, u_GT), dim=0)
    else:
        data_GT = torch.concatenate((a_GT, u_GT), dim=0)

    # print(a_GT.shape, u_GT.shape, data_GT.shape)
    # sys.exit()

    # Get inverse transform functions
    if pde_type in ['shallow_water', 'reaction_diffusion', 'helmholtz']:
    # if pde_type in ['shallow_water', 'reaction_diffusion']:
        PDEtransformer = PDEtransform(data = data_GT, mode = "sample")
        pde_inverse_transform = PDEtransformer.inverse_transform_sample
    else:
        PDEtransformer = oldPDEtransform(pde_type)
        pde_inverse_transform = PDEtransformer.inverse_transform

    if args.pdetype == "nsbounded":
        c_x = config['data']['c_x']
        c_y = config['data']['c_y']
        center = (c_x, c_y) # center of the 2D cylinder
        radius = config['data']['radius'] # radius of the 2D cylinder
    else:
        c_x = c_y = center = radius = None
    
    torch.manual_seed(seed)
    
    net = load_fm4pde(config['model']['pre-trained'], pde_type, device)

    # Get deterministic and stochastic sampler
    if args.dt_sampler == "old":
        deter_sampler = sampler.pdeSampler(net = net, mode = "deterministic", useNODE=False).sampler
    else:
        deter_sampler = sampler.pdeSampler(net = net, mode = "deterministic_new", useNODE=False).sampler

    stoc_sampler = sampler.pdeSampler(net = net, mode = "stochastic", useNODE=False).sampler
    
    # >>>>  Sample the data <<<< #
    print(f'Generating {batch_size} samples for {pde_type} ...')

    # Misc setup
    step_size = config['generate']['step_size']
    num_steps = config['generate']['num_steps']

    if step_size * num_steps != 1:
        step_size = 1 / num_steps
    else:
        step_size = step_size

    method = config['generate']['method']
    T = torch.linspace(0, 1, num_steps).to(device=device)
    stochastic_t = config['generate']['stochastic_t']
    deterministic_t = 1 - stochastic_t
    obsguide_t = config['generate']['obsguide_t']
    
    # Sample initial N(0, 1) noise data
    x_init = torch.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)
    if config['generate']['guide'] == True:
        x_init.requires_grad = True
    else:
        x_init = x_init
    x_next = x_init

    # Initial loss save list and intermediate results save list
    loss = {'pde': [], 'obs_a': [], 'obs_u': [], 'global_a': [], 'global_u': []}
    sol = []
    
    # Sample the sparse observations
    if args.pdetype == "nsbounded":
        img_resolution_x = img_resolution
        img_resolution_y = img_resolution
        if args.problem == 'both':
            known_index_a, a_count = random_index_and_cylinder(center, radius, 0.01, img_resolution, seed=1, device=device)
            known_index_u, u_count = random_index_and_cylinder(center, radius, 0.01, img_resolution, seed=0, device=device)
        elif args.problem == 'forward':
            known_index_a, a_count = random_index_and_cylinder(center, radius, 0.01, img_resolution, seed=1, device=device)
            known_index_u, u_count = cylinder_index(center, radius, img_resolution, device=device)
        elif args.problem == 'inverse':
            known_index_a, a_count = cylinder_index(center, radius, img_resolution, device=device)
            known_index_u, u_count = random_index_and_cylinder(center, radius, 0.01, img_resolution, seed=0, device=device)
        else:
            known_index_a = known_index_u = None
            a_count = u_count = 1
            print("Unknown Problem.")
            sys.exit()
    else:
        a_count = u_count = None
        if img_channels == 1:
            if config['generate']['sensor']:
                img_resolution_x = config['data']['sensor_size']
                img_resolution_y = img_resolution
                known_index_a = random_sensor(config['data']['sensor_size'], img_resolution, seed=1, device=device)
                known_index_u = random_sensor(config['data']['sensor_size'], img_resolution, seed=0, device=device)
            else:
                img_resolution_x = img_resolution
                img_resolution_y = img_resolution
                known_index_a = random_index(obs_size, img_resolution, seed=1, device=device)
                known_index_u = random_index(obs_size, img_resolution, seed=0, device=device)
        else:
            img_resolution_x = img_resolution
            img_resolution_y = img_resolution
            if config['generate']['full']:
                known_index_a = torch.ones((img_resolution, img_resolution), dtype=torch.float32, device=device)
                known_index_u = torch.ones((img_resolution, img_resolution), dtype=torch.float32, device=device)
            else:
                known_index_a = random_index(obs_size, img_resolution, seed=1, device=device)
                known_index_u = random_index(obs_size, img_resolution, seed=0, device=device)

    if a_GT.dim() == 3:
        known_index_a = known_index_a.unsqueeze(0)
        known_index_u = known_index_u.unsqueeze(0)
    else:
        known_index_a = known_index_a
        known_index_u = known_index_u

    a_obs = a_GT * known_index_a
    u_obs = u_GT * known_index_u

    if len(a_GT.shape) == 2:
        data_obs = torch.stack((a_obs, u_obs), dim=0)
    else:
        data_obs = torch.concatenate((a_obs, u_obs), dim=0)

    # print(known_index_a, known_index_u)
    # print(a_obs.shape, u_obs.shape, data_obs.shape)
    # sys.exit()

    # Setup progress bar
    progress = Progress(
        "[progress.description]{task.description}",  # description
        BarColumn(),                                 # process bar
        TaskProgressColumn(),                        # percent
        TimeElapsedColumn(),                         # time elapsed
        TimeRemainingColumn(),                       # time remaining
    )

    main_task = progress.add_task(f"Sampling {batch_size} {pde_type}", total=num_steps)

    time_start = time.time()

    with Live(refresh_per_second=10) as live:
        # Loop through all time steps
        for i in range(1, len(T)):
            # Setup guide strength
            zeta_obs_a = config['generate']['zeta_obs_a']
            zeta_obs_u = config['generate']['zeta_obs_u']
            zeta_pde = config['generate']['zeta_pde']

            # Observation learning rate adjust
            if args.lr_decay:
                if i % args.freq_decay == 0:
                    zeta_obs_a = zeta_obs_a * config['generate']['obsguide_decay']
                    zeta_obs_u = zeta_obs_u * config['generate']['obsguide_decay']
                else:
                    zeta_obs_a = zeta_obs_a * config['generate']['obsguide_decay']
                    zeta_obs_u = zeta_obs_u * config['generate']['obsguide_decay']
            else:
                if i <= obsguide_t * num_steps:
                    zeta_obs_a = zeta_obs_a
                    zeta_obs_u = zeta_obs_u
                else:
                    zeta_obs_a = zeta_obs_a * config['generate']['obsguide_decay']
                    zeta_obs_u = zeta_obs_u * config['generate']['obsguide_decay']

            # Get Flow Matching scheduler
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = getscheduler(T[i], scheduler=config['generate']['scheduler'])
            a_t, b_t = getaffine(alpha_t, sigma_t, d_alpha_t, d_sigma_t, training=config['generate']['training'])
            
            # Sample w/o guidence
            x_cur = x_next.detach().clone()

            if config['generate']['guide'] == True:
                x_cur.requires_grad = True
            else:
                x_cur = x_cur
            
            if config['generate']['samplemode'] == 'deterministic':
                if i <= deterministic_t * num_steps:
                    t = torch.tensor([T[i-1], T[i]])
                    x_1, x_N = deter_sampler(x_cur, t, step_size, method, device)
                    x_tilde = (x_1 + 1.0) / 2.0
                    x_next = (x_N + 1.0) / 2.0
                else:
                    t = T[i-1]
                    x_1, x_N = stoc_sampler(x_cur, t, step_size, method, device)
                    x_1 = x_1.unsqueeze(0)
                    x_N = x_N.unsqueeze(0)
                    x_tilde = (x_1[-1] + 1.0) / 2.0
                    x_next = (x_N[-1] + 1.0) / 2.0
            elif config['generate']['samplemode'] == 'stochastic':
                if i <= stochastic_t * num_steps:
                    t = T[i-1]
                    x_1, x_N = stoc_sampler(x_cur, t, step_size, method, device)
                    x_1 = x_1.unsqueeze(0)
                    x_N = x_N.unsqueeze(0)
                    x_tilde = (x_1[-1] + 1.0) / 2.0
                    x_next = (x_N[-1] + 1.0) / 2.0
                else:
                    t = torch.tensor([T[i-1], T[i]])
                    x_1, x_N = deter_sampler(x_cur, t, step_size, method, device)
                    x_tilde = (x_1 + 1.0) / 2.0
                    x_next = (x_N + 1.0) / 2.0
            else:
                print(f">>> Unknown sampler {config['generate']['samplemode']} <<<")
                sys.exit()

            # data transform
            if img_channels == 1:
                a_N = x_tilde[:,0,:,:].unsqueeze(0)
                u_N = x_tilde[:,0,:,:].unsqueeze(0)
            elif img_channels % 2 == 0:
                a_N = x_tilde[:,:int(img_channels / 2),:,:]
                u_N = x_tilde[:,int(img_channels / 2):,:,:]
            else:
                print(">>> Unknown channels <<<")
                sys.exit()
            
            a_N, u_N = pde_inverse_transform(a_N, u_N)
            a_N = a_N.to(torch.float64)
            u_N = u_N.to(torch.float64)

            # print(x_tilde.shape, a_N.shape, u_N.shape)
            # sys.exit()

            # Compute the observation loss
            pde_loss, observation_loss_a, observation_loss_u = get_pde_loss(a_N, u_N, a_GT, u_GT, known_index_a, known_index_u, device=device)

            # print("funs loss", pde_loss.item(), observation_loss_a.mean().item(), observation_loss_u.mean().item())

            if pde_type == "nsbounded":
                L_obs_a = torch.norm(observation_loss_a, 2)/a_count # type: ignore
                L_obs_u = torch.norm(observation_loss_u, 2)/u_count # type: ignore
            else:
                L_obs_a = torch.norm(observation_loss_a, 2)/obs_size
                L_obs_u = torch.norm(observation_loss_u, 2)/obs_size

            if args.pdeguide == False:
                L_pde = torch.tensor(0)
            else:
                L_pde = torch.norm(pde_loss, 2)/(img_resolution_x*img_resolution_y)

            # print("norm loss", L_pde.item(), L_obs_a.mean().item(), L_obs_u.mean().item())

            # Sample with guidence
            if config['generate']['guide']:
                # Evaluate the gradient
                grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
                grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]

                if args.pdeguide == False:
                    grad_x_cur_pde = 0
                else:
                    grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]

                # Different guide rule for deterministic and stochastic
                if config['generate']['samplemode'] == "deterministic":
                    if i <= deterministic_t * num_steps:
                        x_next = x_next - b_t * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u + zeta_pde * grad_x_cur_pde) * (T[i] - T[i-1])
                    else:
                        x_next = x_next - (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde
                elif config['generate']['samplemode'] == "stochastic":
                    if i <= stochastic_t * num_steps:
                        x_next = x_next - (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde
                    else:
                        x_next = x_next - b_t * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u + zeta_pde * grad_x_cur_pde) * (T[i] - T[i-1])
                else:
                    print("Without Guidance")

            else:
                x_next = x_next
            
            # Save the intermediate results
            if config['generate']['process']:
                sol.append(x_next.detach().cpu())
            else:
                sol = sol

            # Evaluate sparse observation loss, pde loss and global generate loss
            if img_channels == 1:
                a_eval = x_next[:,0,:,:].unsqueeze(0)
                u_eval = x_next[:,0,:,:].unsqueeze(0)
            elif img_channels % 2 == 0:
                a_eval = x_next[:,:int(img_channels / 2),:,:]
                u_eval = x_next[:,int(img_channels / 2):,:,:]
            else:
                print(">>> Unknown channels <<<")
                sys.exit()

            a_eval, u_eval = pde_inverse_transform(a_eval, u_eval)
            a_eval = a_eval.to(torch.float64)
            u_eval = u_eval.to(torch.float64)

            re_a_eval = torch.norm(a_eval.squeeze() - a_GT.squeeze(), 2) / torch.norm(a_GT.squeeze(), 2)
            re_u_eval = torch.norm(u_eval.squeeze() - u_GT.squeeze(), 2) / torch.norm(u_GT.squeeze(), 2)
            
            loss['pde'].append(L_pde.item())
            loss['obs_a'].append(L_obs_a.item())
            loss['obs_u'].append(L_obs_u.item())
            loss['global_a'].append(re_a_eval.item())
            loss['global_u'].append(re_u_eval.item())
        
            x_next = x_next * 2.0 - 1.0

            # Push forward the progress bar
            progress.update(main_task, advance=1)
            live.update(
                    Group(
                        progress.get_renderable(),
                        progress_metrics(i, re_a_eval.item(), re_u_eval.item(), L_pde.item())
                        )
                    )

            if config['generate']['guide']:
                if x_cur.grad is not None:
                    x_cur.grad.detach_()
                    x_cur.grad.zero_()

            torch.cuda.empty_cache()
    time_end = time.time()

    # >>>> Final results evaluate, return and save <<<< #
    time_eval = str(timedelta(seconds=int(time_end - time_start)))

    x_final = (x_next + 1.0) / 2.0
    if img_channels == 1:
        a_final = x_final[:,0,:,:].unsqueeze(0)
        u_final = x_final[:,0,:,:].unsqueeze(0)
    elif img_channels % 2 == 0:
        a_final = x_final[:,:int(img_channels / 2),:,:]
        u_final = x_final[:,int(img_channels / 2):,:,:]
    else:
        print(">>> Unknown channels <<<")
        sys.exit()

    a_final, u_final = pde_inverse_transform(a_final, u_final)
    a_final = a_final.to(torch.float64)
    u_final = u_final.to(torch.float64)

    relative_error_a = torch.norm(a_final.squeeze() - a_GT.squeeze(), 2) / torch.norm(a_GT.squeeze(), 2)
    relative_error_u = torch.norm(u_final.squeeze() - u_GT.squeeze(), 2) / torch.norm(u_GT.squeeze(), 2)

    print(f'Relative error of a: {relative_error_a}')
    print(f'Relative error of u: {relative_error_u}')

    data_final = torch.cat([a_final, u_final], dim=1)

    a_final = a_final.squeeze()
    u_final = u_final.squeeze()
    data_final = data_final.squeeze()

    cur_time = datetime.now().strftime("%Y%m%d")

    # Plot learning curve
    if config['output']['plot'] and config['output']['save']:
        # savedir = f'{config['output']['savepath']}{args.problem}/figs/{cur_time}-FM4{pde_type}-obs{obs_size}-step({num_steps},{step_size})-zeta({config['generate']['zeta_obs_a']},{config['generate']['zeta_obs_u']},{config['generate']['zeta_pde']})-{method}-{config['generate']['samplemode']}-thre_t({stochastic_t},{obsguide_t})/{args.mode}'
        savedir = f'{config['output']['savepath']}{args.problem}/figs/{cur_time}-FM4{pde_type}/{args.mode}'

        os.makedirs(savedir, exist_ok=True)

        plt.figure(figsize=(8, 5))

        for key, values in loss.items():
            plt.plot(values, label=key)

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.suptitle('Training Metrics')
        plt.legend()
        plt.title(f'Relative error coef: {np.round(relative_error_a.item(), 4)}, Relative error sol:{np.round(relative_error_u.item(), 4)}')
        plt.figtext(0.0, -0.05, f'Elapsed Time: {time_eval}')

        plt.savefig(f'{savedir}/{config['data']['name']}{setup_file_name}{args.remark}_metrics.pdf')
        plt.savefig(f'{savedir}/{config['data']['name']}{setup_file_name}{args.remark}_metrics.eps')
        plt.close()

        # Plot global true, sparse observations and generated results
        data_GT = data_GT.detach().cpu().numpy()
        data_obs = data_obs.detach().cpu().numpy()
        data_final = data_final.detach().cpu().numpy()

        plot_eval_pde(data_GT, data_obs, data_final)
        plt.savefig(f'{savedir}/{config['data']['name']}{setup_file_name}{args.remark}_viz.pdf')
        plt.savefig(f'{savedir}/{config['data']['name']}{setup_file_name}{args.remark}_viz.eps')
        plt.close()
    else:
        print("User declare not plot.")

    # Save and return the results
    if config['output']['save']:
        # Save results
        with open(f'{config['output']['savepath']}{args.problem}/{config['data']['name']}{setup_file_name}{args.remark}_results.pkl', 'wb') as f:
            pickle.dump({
                'obs_index': {'known_index_a': known_index_a, 'known_index_u': known_index_u},
                'coef_final': a_final,
                'sol_final': u_final,
                'loss': loss,
                'intermediate': sol,
                'time': time_eval,
                'config': config
                }, f)
    else:
        print("User declare no save.")

    if config['output']['return']:
        print("Done.")
        return {
                'obs_index': {'known_index_a': known_index_a, 'known_index_u': known_index_u},
                'coef_final': a_final,
                'sol_final': u_final,
                'loss': loss,
                'intermediate': sol,
                'time': time_eval
            }
    else:
        print("Done.")



if __name__ == "__main__":
    args = parser.parse_args()
    batch = args.batch
    for i in range(batch):
        main(args, i)
