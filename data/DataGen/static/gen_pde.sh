#!/bin/bash

echo ">> Generating 2D Darcy Flow. <<"
matlab -batch generate_darcy

echo ">> Generating 2D Helmholtz Equation. <<"
matlab -batch generate_inhom_helmholtz

echo ">> Generating 2D Poisson Equation. <<"
matlab -batch generate_poisson

echo ">> Generating 1D Burgers Equation. <<"
matlab -batch gen_burgers1

echo ">> All Done. <<"
