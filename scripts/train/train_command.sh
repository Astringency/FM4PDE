# W/O SCALAR CONDITIONING
PDE=poisson MASTER_PORT_BASE=29500 bash scripts/train/run_train.sh
PDE=helmholtz MASTER_PORT_BASE=29510 bash scripts/train/run_train.sh
PDE=darcy MASTER_PORT_BASE=29520 bash scripts/train/run_train.sh
PDE=steady_heat_conduction MASTER_PORT_BASE=29600 bash scripts/train/run_train.sh
PDE=burger MASTER_PORT_BASE=29540 bash scripts/train/run_train.sh
PDE=nsnonbounded MASTER_PORT_BASE=29530 bash scripts/train/run_train.sh
PDE=heat MASTER_PORT_BASE=29570 bash scripts/train/run_train.sh
PDE=advection_diffusion MASTER_PORT_BASE=29590 bash scripts/train/run_train.sh
PDE=wave MASTER_PORT_BASE=29580 bash scripts/train/run_train.sh
PDE=reaction_diffusion MASTER_PORT_BASE=29550 bash scripts/train/run_train.sh
PDE=shallow_water MASTER_PORT_BASE=29560 bash scripts/train/run_train.sh

# W SCALAR CONDITIONING
PDE=heat MASTER_PORT_BASE=29570 SCALAR_CONDITIONING_PARAMS="alpha T" bash scripts/train/run_train.sh
PDE=wave MASTER_PORT_BASE=29580 SCALAR_CONDITIONING_PARAMS="c T" bash scripts/train/run_train.sh
PDE=advection_diffusion MASTER_PORT_BASE=29590 SCALAR_CONDITIONING_PARAMS="b_x b_y kappa T" bash scripts/train/run_train.sh
PDE=reaction_diffusion MASTER_PORT_BASE=29550 SCALAR_CONDITIONING_PARAMS="T D_u D_v k" bash scripts/train/run_train.sh
PDE=shallow_water MASTER_PORT_BASE=29560 SCALAR_CONDITIONING_PARAMS="g T" bash scripts/train/run_train.sh
PDE=nsnonbounded MASTER_PORT_BASE=29530 SCALAR_CONDITIONING_PARAMS="nu T" bash scripts/train/run_train.sh
PDE=burger MASTER_PORT_BASE=29540 SCALAR_CONDITIONING_PARAMS="nu" bash scripts/train/run_train.sh
PDE=steady_heat_conduction MASTER_PORT_BASE=29600 SCALAR_CONDITIONING_PARAMS="u_D" bash scripts/train/run_train.sh
