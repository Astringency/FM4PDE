#!/bin/bash

pde_vals=("darcy")
# pde_vals=("darcy" "poisson" "helmholtz" "nsnonbounded")
# pde_vals=("nsnonbounded" "shallow_water" "reaction_diffusion")

# for pde in "${pde_vals[@]}"; do
#     python -u sample.py --pdetype="$pde" --problem="both" --config="./configs/${pde}.yaml"
# done

# mode_vals=("sparse" "full")
mode_vals=("sparse")
# problem_vals=("both" "forward" "inverse")
problem_vals=("forward" "inverse")
steps_vals=(100 200 500 1000 2000)

for pde in "${pde_vals[@]}"; do
    for problem in "${problem_vals[@]}"; do
        for steps in "${steps_vals[@]}"; do
            if [ "$problem" == "both" ]; then
                echo ">>> Flow Matching for ${pde} on ${problem} problem from sparse observations. <<<"
                python -u sample.py --pdetype="$pde" --problem="$problem" --config="./configs/${pde}.yaml" --num_steps=$steps
            else
                for mode in "${mode_vals[@]}"; do
                    echo ">>> Flow Matching for ${pde} on ${problem} problem from ${mode} observations. <<<"
                    python -u sample.py --pdetype="$pde" --problem="$problem" --mode="$mode" --config="./configs/${pde}.yaml" --num_steps=$steps
                done
            fi
        done
    done
done

wait

echo ">>> Done. <<<"

# python -u sample.py --pdetype "burger" --mode "sparse" --config "./configs/burger.yaml" #--lr_decay True --freq_decay 200
# python -u sample.py --pdetype "darcy" --mode "sparse" --config "./configs/darcy.yaml"

# python -u sample.py --pdetype "poisson" --mode "sparse" --config "./configs/poisson.yaml"
# python -u sample.py --pdetype "poisson" --mode "sparse" --config "./configs/poisson.yaml" --problem "forward"
# python -u sample.py --pdetype "poisson" --mode "sparse" --config "./configs/poisson.yaml" --problem "inverse"

# python -u sample.py --pdetype "helmholtz" --mode "sparse" --config "./configs/helmholtz.yaml"
# python -u sample.py --pdetype "helmholtz" --mode "sparse" --config "./configs/helmholtz.yaml" --problem "forward" --remark "k5"
# python -u sample.py --pdetype "helmholtz" --mode "sparse" --config "./configs/helmholtz.yaml" --problem "inverse" --remark "k5"

# python -u sample.py --pdetype "helmholtz" --mode "sparse" --config "./configs/helmholtz.yaml" --remark "k1"
# python -u sample.py --pdetype "helmholtz" --mode "full" --config "./configs/helmholtz.yaml" --problem "forward" --remark "k1" --plot True
# python -u sample.py --pdetype "helmholtz" --mode "full" --config "./configs/helmholtz.yaml" --problem "inverse" --remark "k1" --plot True
