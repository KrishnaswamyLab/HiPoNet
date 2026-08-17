#!/usr/bin/env python
"""Jointly fine-tune a count-free HiPoNet-conditioned population generator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.population_flow import ConditionalPopulationFlow
from train_soft_pointcloud_corrective_flow import (
    aggregate,
    distribution_loss,
    evaluation_metrics,
    integrate_corrective_flow,
    sample_cells,
)
from train_stochastic_soft_corrective_flow import sample_soft_coupled_targets
from train_variable_population_generator import (
    VariableSoftPointCloudDecoder,
    full_population_moment_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_cache", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_patient", default="75")
    parser.add_argument("--test_patient", default="99")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--training_points", default="128,256,512")
    parser.add_argument("--cell_budget", type=int, default=1024)
    parser.add_argument("--query_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--initial_scale", type=float, default=0.10)
    parser.add_argument("--maximum_scale", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--source_weight", type=float, default=0.10)
    parser.add_argument("--flow_matching_weight", type=float, default=0.10)
    parser.add_argument("--full_cloud_weight", type=float, default=0.05)
    parser.add_argument("--full_cloud_every", type=int, default=25)
    parser.add_argument("--ot_epsilon", type=float, default=0.10)
    parser.add_argument("--ot_iterations", type=int, default=30)
    parser.add_argument("--train_integration_steps", type=int, default=4)
    parser.add_argument("--integration_steps", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--metric_points", type=int, default=256)
    parser.add_argument("--eval_populations", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1820)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    point_choices = tuple(int(value) for value in args.training_points.split(","))
    if any(value < 2 for value in point_choices):
        raise ValueError("Training point counts must be at least two")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cache = np.load(args.population_cache, allow_pickle=True)
    populations = [np.asarray(value, dtype=np.float32) for value in cache["populations"]]
    names = [str(value) for value in cache["group_names"]]
    latents = np.asarray(np.load(args.latents), dtype=np.float32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_summary = checkpoint["summary"]
    cell_dim = int(source_summary["cell_dim"])
    latent_dim = int(source_summary["latent_dim"])
    if len(populations) != len(latents) or latents.shape[1] != latent_dim:
        raise ValueError("Population and latent arrays do not match the checkpoint")

    patients = np.asarray([name.split("__", 1)[0] for name in names])
    validation_ids = np.flatnonzero(patients == str(args.validation_patient))
    test_ids = np.flatnonzero(patients == str(args.test_patient))
    train_ids = np.flatnonzero(
        (patients != str(args.validation_patient))
        & (patients != str(args.test_patient))
    )
    if min(len(train_ids), len(validation_ids), len(test_ids)) == 0:
        raise ValueError("Train, validation, and test splits must be nonempty")

    latent_mean = np.asarray(checkpoint["latent_mean"], dtype=np.float32)
    latent_std = np.asarray(checkpoint["latent_std"], dtype=np.float32)
    cell_mean = np.asarray(checkpoint["cell_mean"], dtype=np.float32)
    cell_std = np.asarray(checkpoint["cell_std"], dtype=np.float32)
    scaled_latents = ((latents - latent_mean) / latent_std).astype(np.float32)
    latent_tensor = torch.from_numpy(scaled_latents).to(device)

    def to_model(values: np.ndarray) -> np.ndarray:
        return ((values - cell_mean) / cell_std).astype(np.float32)

    def from_model(values: np.ndarray) -> np.ndarray:
        return (values * cell_std + cell_mean).astype(np.float32)

    decoder = VariableSoftPointCloudDecoder(
        latent_dim,
        args.query_dim,
        cell_dim,
        args.hidden_dim,
        args.initial_scale,
        args.maximum_scale,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    flow = ConditionalPopulationFlow(
        cell_dim=cell_dim,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
    ).to(device)
    flow.load_state_dict(checkpoint["flow_state_dict"])

    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(flow.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    projection_rng = np.random.default_rng(args.seed + 1)
    projections = projection_rng.normal(size=(32, cell_dim)).astype(np.float32)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    projection_tensor = torch.from_numpy(projections).to(device)

    validation_rng = np.random.default_rng(args.seed + 2)
    validation_selection = validation_ids[:8]
    validation_latent = latent_tensor[torch.from_numpy(validation_selection).to(device)]
    validation_target = torch.from_numpy(
        np.stack(
            [
                to_model(sample_cells(populations[int(index)], 128, validation_rng))
                for index in validation_selection
            ]
        )
    ).to(device)
    validation_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    validation_queries = torch.randn(
        len(validation_selection), 128, args.query_dim,
        device=device, generator=validation_generator,
    )
    validation_noise = torch.randn(
        len(validation_selection), 128, cell_dim,
        device=device, generator=validation_generator,
    )

    def validation_score() -> float:
        decoder.eval()
        flow.eval()
        with torch.no_grad():
            source, _, _ = decoder(
                validation_latent, validation_queries, validation_noise
            )
            corrected = integrate_corrective_flow(
                flow, source, validation_latent, args.integration_steps
            )
            value, _ = distribution_loss(
                corrected, validation_target, projection_tensor,
                0.25, 0.25, 0.25, 0.20,
            )
        decoder.train()
        flow.train()
        return float(value)

    initial_validation = validation_score()
    best_validation = initial_validation
    best_step = 0
    best_decoder_state = {
        key: value.detach().cpu().clone()
        for key, value in decoder.state_dict().items()
    }
    best_flow_state = {
        key: value.detach().cpu().clone()
        for key, value in flow.state_dict().items()
    }
    history = []

    for step in range(1, args.steps + 1):
        n_points = int(rng.choice(point_choices))
        batch_size = max(1, args.cell_budget // n_points)
        selected = rng.choice(
            train_ids, size=batch_size, replace=len(train_ids) < batch_size
        )
        selected_tensor = torch.from_numpy(selected).to(device)
        latent = latent_tensor[selected_tensor]
        real = torch.from_numpy(
            np.stack(
                [
                    to_model(sample_cells(populations[int(index)], n_points, rng))
                    for index in selected
                ]
            )
        ).to(device)
        queries = torch.randn(batch_size, n_points, args.query_dim, device=device)
        noise = torch.randn_like(real)
        source, _, _ = decoder(latent, queries, noise)
        corrected = integrate_corrective_flow(
            flow, source, latent, args.train_integration_steps
        )
        endpoint_loss, endpoint_components = distribution_loss(
            corrected, real, projection_tensor, 0.25, 0.25, 0.25, 0.20
        )
        source_loss, _ = distribution_loss(
            source, real, projection_tensor, 0.25, 0.25, 0.25, 0.20
        )

        with torch.no_grad():
            coupled_target = sample_soft_coupled_targets(
                source.detach(), real, args.ot_epsilon, args.ot_iterations
            )
        time = torch.rand(batch_size, n_points, 1, device=device)
        path = (1.0 - time) * source.detach() + time * coupled_target
        target_velocity = coupled_target - source.detach()
        condition = latent[:, None].expand(-1, n_points, -1)
        predicted_velocity = flow(
            path.flatten(0, 1), time.flatten(), condition.flatten(0, 1)
        ).view_as(path)
        flow_matching_loss = F.mse_loss(predicted_velocity, target_velocity)

        full_cloud_loss = endpoint_loss.new_zeros(())
        if args.full_cloud_every > 0 and step % args.full_cloud_every == 0:
            full_index = int(rng.choice(selected))
            full_target = torch.from_numpy(to_model(populations[full_index])).to(device)
            full_queries = torch.randn(
                1, len(full_target), args.query_dim, device=device
            )
            full_noise = torch.randn(1, len(full_target), cell_dim, device=device)
            full_source, _, _ = decoder(
                latent_tensor[full_index][None], full_queries, full_noise
            )
            full_corrected = integrate_corrective_flow(
                flow,
                full_source,
                latent_tensor[full_index][None],
                args.train_integration_steps,
            )
            full_cloud_loss = full_population_moment_loss(
                full_corrected[0], full_target
            )

        loss = (
            endpoint_loss
            + args.source_weight * source_loss
            + args.flow_matching_weight * flow_matching_loss
            + args.full_cloud_weight * full_cloud_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(decoder.parameters()) + list(flow.parameters()), 5.0
        )
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            score = validation_score()
            row = {
                "step": step,
                "training_points": n_points,
                "batch_populations": batch_size,
                "training_total": float(loss.detach()),
                "endpoint_distribution": float(endpoint_loss.detach()),
                "source_distribution": float(source_loss.detach()),
                "flow_matching": float(flow_matching_loss.detach()),
                "full_cloud_moments": float(full_cloud_loss.detach()),
                **{
                    f"endpoint_{key}": float(value.detach())
                    for key, value in endpoint_components.items()
                },
                "validation_distribution": score,
            }
            history.append(row)
            print(json.dumps({"stage": "joint", **row}), flush=True)
            if score < best_validation:
                best_validation = score
                best_step = step
                best_decoder_state = {
                    key: value.detach().cpu().clone()
                    for key, value in decoder.state_dict().items()
                }
                best_flow_state = {
                    key: value.detach().cpu().clone()
                    for key, value in flow.state_dict().items()
                }
            if step - best_step >= args.patience:
                break

    decoder.load_state_dict(best_decoder_state)
    flow.load_state_dict(best_flow_state)
    decoder.eval()
    flow.eval()

    metric_projections = projection_rng.normal(size=(64, cell_dim))
    metric_projections /= np.linalg.norm(metric_projections, axis=1, keepdims=True)
    eval_rng = np.random.default_rng(args.seed + 4)
    selected_test = eval_rng.choice(
        test_ids, size=min(args.eval_populations, len(test_ids)), replace=False
    )
    detail_rows = []
    with torch.no_grad():
        for population_id in selected_test:
            target = populations[int(population_id)]
            n_cells = len(target)
            latent = latent_tensor[int(population_id)][None]
            queries = torch.randn(1, n_cells, args.query_dim, device=device)
            noise = torch.randn(1, n_cells, cell_dim, device=device)
            source, _, _ = decoder(latent, queries, noise)
            corrected = integrate_corrective_flow(
                flow, source, latent, args.integration_steps
            )
            generated = {
                "joint_soft_decoder": from_model(source[0].cpu().numpy()),
                "joint_corrected_flow": from_model(corrected[0].cpu().numpy()),
                "real_vs_real": sample_cells(target, n_cells, eval_rng),
            }
            for method, prediction in generated.items():
                metric_count = min(args.metric_points, len(prediction), len(target))
                prediction_sample = sample_cells(prediction, metric_count, eval_rng)
                target_sample = sample_cells(target, metric_count, eval_rng)
                detail_rows.append(
                    {
                        "population_id": int(population_id),
                        "population_name": names[int(population_id)],
                        "true_count": n_cells,
                        "generated_count": len(prediction),
                        "method": method,
                        **evaluation_metrics(
                            prediction_sample, target_sample, metric_projections
                        ),
                    }
                )

    metrics_path = args.output_dir / "test_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    methods = ("joint_soft_decoder", "joint_corrected_flow", "real_vs_real")
    distribution_metrics = {
        method: aggregate([row for row in detail_rows if row["method"] == method])
        for method in methods
    }
    summary = {
        "method": "count-free joint latent-to-original-population reconstruction",
        "cell_dim": cell_dim,
        "latent_dim": latent_dim,
        "generation_count_source": "target_population_count",
        "training_point_choices": list(point_choices),
        "initial_validation": initial_validation,
        "best_step": best_step,
        "best_validation": best_validation,
        "joint_improved": best_step > 0,
        "distribution_metrics": distribution_metrics,
    }
    torch.save(
        {
            "decoder_state_dict": best_decoder_state,
            "flow_state_dict": best_flow_state,
            "summary": summary,
            "cell_mean": cell_mean,
            "cell_std": cell_std,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        },
        args.output_dir / "joint_latent_population_generator.pt",
    )
    (args.output_dir / "joint_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
