# HiPoNet Population Generation

The current generation pipeline reconstructs a cell population from a real
HiPoNet population latent. Population size is supplied explicitly; the model
does not predict cell count.

`train_population_reflow.py` implements the complete generation experiment:

1. `SoftMLPDecoder` concatenates the HiPoNet population latent `z` with
   per-cell Gaussian noise and passes each `[z, epsilon_i]` vector through one
   sequential MLP. The resulting rows form the initial `N x g` soft point
   cloud. There are no Fourier slots, context aggregation, or coupling layers.
2. Meta's `flow_matching` package constructs the linear conditional probability
   path between independently paired source and target cells within each
   population.
3. `PopulationVelocityField` learns `v(x_t, t)` using only flow-matching MSE.
   The velocity model does not receive the HiPoNet latent.
4. Heun integration transports the soft population from `t=0` to `t=1`.
5. Held-out evaluation reports mean and sample standard deviation for Chamfer,
   exact uniform-mass EMD, PCC, and SCC.

Decoder and flow checkpoint selection evaluate every cell in each validation
population. Variable-sized populations are processed one at a time.

Training also uses every cell in the selected population. Each optimization
step processes one variable-sized population and computes dense Sinkhorn,
sliced Wasserstein, moment, and diversity losses on its complete cloud.

The implementation is intentionally small:

- `train_population_reflow.py`: training and held-out evaluation
- `models/population_flow.py`: unconditioned velocity model
- `utils/population_generation.py`: generation losses, integration, and metrics
- `run_pdo_population_reflow.sbatch`: Bouchet PDO entry point

The batch script expects the PDO population cache and HiPoNet latent
representations to exist on Bouchet. Large data and checkpoint artifacts are
not stored in this repository.

For the CAF-inclusive experiment, `prepare_pdo_caf_sampled512.sbatch` builds
3,347 populations containing both PDO cells and cancer-associated fibroblasts
across all 45 markers. `run_pdo_caf_population_reflow.sbatch` trains the same
decoder and reflow model with patients 75 and 99 held out. This configuration
uses at most 512 real cells per population because dense Sinkhorn storage is
quadratic in population size; it is distinct from the full-cell PDO run.
