# HiPoNet Population Generation

The current generation pipeline reconstructs a cell population from a real
HiPoNet population latent. Population size is supplied explicitly; the model
does not predict cell count.

`train_population_reflow.py` implements the complete generation experiment:

1. A soft point-cloud MLP maps the HiPoNet latent `z`, canonical slot features,
   and residual noise to an initial population.
2. Meta's `flow_matching` package constructs the linear conditional probability
   path between independently paired source and target cells within each
   population.
3. `PopulationVelocityField` learns `v(x_t, t)` using only flow-matching MSE.
   The velocity model does not receive the HiPoNet latent.
4. Heun integration transports the soft population from `t=0` to `t=1`.
5. Held-out evaluation reports mean and sample standard deviation for Chamfer,
   exact uniform-mass EMD, PCC, and SCC.

The implementation is intentionally small:

- `train_population_reflow.py`: training and held-out evaluation
- `models/population_flow.py`: unconditioned velocity model
- `utils/population_generation.py`: generation losses, integration, and metrics
- `run_pdo_population_reflow.sbatch`: Bouchet PDO entry point

The batch script expects the PDO population cache and HiPoNet latent
representations to exist on Bouchet. Large data and checkpoint artifacts are
not stored in this repository.
