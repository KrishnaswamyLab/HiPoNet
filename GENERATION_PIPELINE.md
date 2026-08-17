# HiPoNet Population Generation

The current generation pipeline reconstructs a cell population from a real
HiPoNet population latent. Population size is supplied explicitly; the model
does not predict cell count.

`train_population_reflow.py` is the current architecture experiment. An MLP
maps the HiPoNet latent, query features, and residual noise to a soft point
cloud. The `--flow_matcher_backend` flag selects either the installed
`torchcfm` package's stochastic exact-OT matcher or the installed Meta
`flow_matching` package's original `CondOTProbPath`. The latter receives
independent within-population source/target samples, as the package leaves the
coupling to the caller. There is no PCA ordering or locally reimplemented
probability path. The pointwise rectifier is conditioned on the current cell,
time, and HiPoNet population latent and uses only velocity MSE. Its Bouchet
entry point is `run_pdo_population_reflow.sbatch`.

## Current pipeline

1. `train_variable_population_generator.py` trains the multiscale stochastic
   soft decoder and OT-conditioned corrective flow.
2. `finetune_latent_population_generator.py` removes the count head from the
   generation path and jointly fine-tunes the decoder and flow against the
   original population associated with each HiPoNet latent.

The corresponding Bouchet entry points are:

- `run_pdo_variable_population_generator.sbatch`
- `run_pdo_joint_latent_population_generator.sbatch`

Shared implementation modules retained at the repository root are:

- `models/population_flow.py`
- `train_soft_pointcloud_corrective_flow.py`
- `train_stochastic_soft_corrective_flow.py`

Held-out Chamfer and exact uniform-mass Earth Mover's Distance are computed by
`evaluate_population_generator_emd.py`. The active reflow evaluation also
reports mean and sample standard deviation for Chamfer, EMD, and Pearson and
Spearman correlations between generated and real 44-marker mean profiles.

The support trainers remain at the root because the current pipeline imports
their flow model, distribution-loss, integration, and evaluation functions.

## Data and checkpoints

The batch scripts expect the PDO population cache, HiPoNet latent
representations, and generated checkpoints to live on Bouchet. These large
artifacts are intentionally not stored in this repository.

## Historical experiments

Older standalone generation approaches are preserved in
`legacy/generation_experiments/`. They are not imported or executed by the current
pipeline.
