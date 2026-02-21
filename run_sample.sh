#!/bin/bash

# === CONFIGS === #
pde_vals=("darcy" "poisson" "helmholtz" "nsnonbounded" "shallow_water" "reaction_diffusion")
sampler_vals=("stochastic" "deterministic")
problem_vals=("both" "forward" "inverse")
mode_vals=("sparse" "full")
steps_vals=(100 200 500 1000 2000)
obs_vals=(100 500 1000 2000 5000)

# === MAIN GENERATE === #
for pde in "${pde_vals[@]}"; do
    for problem in "${problem_vals[@]}"; do
        for steps in "${steps_vals[@]}"; do
            if [ "$problem" == "both" ]; then
                echo ">>> Flow Matching for ${pde} on ${problem} problem from sparse observations. <<<"
                python -u sample.py --pdetype="$pde" --problem="$problem" --config="./configs/${pde}.yaml" --num_steps=$steps --batch 20
            else
                for mode in "${mode_vals[@]}"; do
                    echo ">>> Flow Matching for ${pde} on ${problem} problem from ${mode} observations. <<<"
                    python -u sample.py --pdetype="$pde" --problem="$problem" --mode="$mode" --config="./configs/${pde}.yaml" --num_steps=$steps --batch 20
                done
            fi
        done
    done
done

# === TEST SAMPLER === #
for pde in "${pde_vals[@]}"; do
    for problem in "${problem_vals[@]}"; do
        for sampler in "${sampler_vals[@]}"; do
            if [ "$problem" == "both" ]; then
                echo ">>> Flow Matching for ${pde} on ${problem} problem from sparse observations. <<<"
                python -u sample.py --pdetype "$pde" --problem "$problem" --config "./configs/${pde}.yaml" --sampler "$sampler" --hybrid true
            else
                for mode in "${mode_vals[@]}"; do
                    echo ">>> Flow Matching for ${pde} on ${problem} problem from ${mode} observations. <<<"
                    python -u sample.py --pdetype "$pde" --problem "$problem" --mode "$mode" --config "./configs/${pde}.yaml" --sampler "$sampler" --hybrid true
                done
            fi
        done
    done
done

# === TEST OBSERVATIONS NUMBER ===
for obs in "${obs_vals[@]}"; do
    for pde in "${pde_vals[@]}"; do
        for problem in "${problem_vals[@]}"; do
            echo ">>> Flow Matching for ${pde} on ${problem} problem from Different Sparse observations. <<<"
            python -u sample.py --pdetype="$pde" --problem="$problem" --mode="sparse" --config="./configs/${pde}.yaml" --num_obs=$obs
        done
    done
done

# === TEST TIME & STEP SIZE === #
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


# === MISC TASK (Example) === #

# python -u sample.py --pdetype "burger" --mode "sparse" --config "./configs/burger.yaml"
# python -u sample.py --pdetype "darcy" --mode "sparse" --config "./configs/darcy.yaml"
# python -u sample.py --pdetype "poisson" --mode "sparse" --config "./configs/poisson.yaml"
# python -u sample.py --pdetype "reaction_diffusion" --problem "both" --mode "sparse" --config "./configs/reaction_diffusion.yaml" --sampler "stochastic"
# python -u sample.py --pdetype "shallow_water" --problem "both" --mode "sparse" --config "./configs/shallow_water.yaml" --sampler "stochastic"


wait

echo ">>> Done. <<<"


