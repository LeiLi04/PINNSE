from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PARENTS = CURRENT_DIR.parents
PROJECT_ROOT = PARENTS[2] if len(PARENTS) >= 3 else CURRENT_DIR
SRC_ROOT = PARENTS[1] if len(PARENTS) >= 2 else CURRENT_DIR
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

cudnn.enabled = False

from src.models.GAKF.discriminator import GAKFDiscriminator  # noqa: E402
from src.models.GAKF.generator import GAKFGenerator  # noqa: E402
import src.models.GAKF.losses as losses  # noqa: E402
import src.models.GAKF.utils as utils  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAKF")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = utils.load_config(args.config)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    params = utils.prepare_parameters(config, device)
    H = params["H"]
    R = params["R"]
    data_cfg = config["data"]

    dataset_dir = Path(data_cfg["dataset_path"])
    if "dataset_file" in data_cfg:
        dataset_file = dataset_dir / data_cfg["dataset_file"]
    else:
        dataset_file = utils.parse_dataset_file(
            dataset_dir,
            data_cfg["dataset_type"],
            data_cfg["n_states"],
            data_cfg["n_obs"],
            data_cfg["T"],
        )
    dataset_pairs = utils.load_pickle_dataset(dataset_file)
    train_loader, val_loader = utils.create_dataloaders(
        dataset_pairs=dataset_pairs,
        batch_size=config["train"]["batch_size"],
        N_train=data_cfg["N_train"],
        N_val=data_cfg["N_val"],
    )

    print(f"Using device: {device}")
    print(f"Dataset file: {dataset_file}")
    print(
        f"Train loader batches: {len(train_loader)}, "
        f"Val loader batches: {len(val_loader)}"
    )
    if len(train_loader) == 0:
        print("Train loader is empty. Check dataset_path / N_train / batch_size settings.")
        return
    if len(val_loader) == 0:
        print("Warning: validation loader is empty; only training losses will be tracked.")

    gen_cfg = config["model"]
    generator = GAKFGenerator(
        n_obs=data_cfg["n_obs"],
        n_states=data_cfg["n_states"],
        H=H,
        R=R,
        hidden_dim=gen_cfg["hidden"],
        num_layers=gen_cfg["layers"],
        rnn_type=gen_cfg["rnn"],
        diag_cov=gen_cfg.get("diag_cov", True),
        lowrank=gen_cfg.get("lowrank", False),
        device=device,
    ).to(device)

    discriminator = GAKFDiscriminator(
        n_obs=data_cfg["n_obs"],
        hidden_dim=gen_cfg["hidden"],
        num_layers=gen_cfg["layers"],
    ).to(device)

    train_cfg = config["train"]
    optim_G = optim.Adam(
        generator.parameters(),
        lr=train_cfg["lr_G"],
        betas=tuple(train_cfg["betas"]),
    )
    optim_D = optim.Adam(
        discriminator.parameters(),
        lr=train_cfg["lr_D"],
        betas=tuple(train_cfg["betas"]),
    )

    writer = utils.make_summary_writer(Path(config["log"]["out_dir"]))
    best_metric = math.inf

    # Warm-up phase
    generator.train()
    discriminator.train()
    for epoch in range(train_cfg["warm_epochs"]):
        warm_prog = tqdm(
            train_loader, desc=f"Warm-up {epoch + 1}/{train_cfg['warm_epochs']}"
        )
        for y_batch, _ in warm_prog:
            y_batch = y_batch.to(device)
            optim_G.zero_grad(set_to_none=True)
            outs = generator(y_batch)
            loss_rec = losses.reconstruction_loss(y_batch, outs.y_hat, R.diag())
            loss_innov = losses.innovation_whitening_loss(outs.aux["innovation_white"])
            warm_loss = loss_rec + loss_innov
            warm_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            optim_G.step()
        generator.update_theta_bar(momentum=0.0)

    global_step = 0
    total_steps = train_cfg["steps"]
    progress = tqdm(total=total_steps, desc="Training")

    while global_step < total_steps:
        for y_real, _ in train_loader:
            if global_step >= total_steps:
                break
            y_real = y_real.to(device)

            # Train discriminator
            for _ in range(train_cfg["n_critic"]):
                optim_D.zero_grad(set_to_none=True)
                with torch.no_grad():
                    y_fake = generator(y_real).y_hat
                D_real = discriminator(y_real)
                D_fake = discriminator(y_fake)
                gp = losses.gradient_penalty(discriminator, y_real, y_fake, device)
                loss_D = losses.wgan_gp_discriminator_loss(
                    D_real, D_fake, gp, train_cfg["gp_lambda"]
                )
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                optim_D.step()

            # Train generator
            optim_G.zero_grad(set_to_none=True)
            outs = generator(y_real)
            D_fake = discriminator(outs.y_hat)
            loss_adv = losses.wgan_gp_generator_loss(D_fake)
            loss_rec = losses.reconstruction_loss(y_real, outs.y_hat, R.diag())
            loss_ms = losses.multi_step_loss(
                generator_rollout_fn=lambda prefix, horizon: generator.rollout(
                    prefix, horizon
                ),
                y_seq=y_real,
                horizon=train_cfg["rollout_K"],
            )
            loss_innov = losses.innovation_whitening_loss(outs.aux["innovation_white"])
            loss_kf = losses.riccati_consistency_loss(outs.P_post, outs.aux["P_post_full"])
            theta_vec = torch.cat([p.flatten() for p in generator.parameters()])
            loss_apbm = losses.apbm_regularizer(
                theta_vec, generator.theta_bar, alpha=train_cfg["lambda_apbm"]
            )
            total_loss = (
                loss_adv
                + train_cfg["lambda_rec"] * loss_rec
                + train_cfg["lambda_ms"] * loss_ms
                + train_cfg["lambda_innov"] * loss_innov
                + train_cfg["lambda_kf"] * loss_kf
                + loss_apbm
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            optim_G.step()
            generator.update_theta_bar(momentum=0.995)

            writer.add_scalar("train/loss_adv", loss_adv.item(), global_step)
            writer.add_scalar("train/loss_rec", loss_rec.item(), global_step)
            writer.add_scalar("train/loss_ms", loss_ms.item(), global_step)
            writer.add_scalar("train/loss_innov", loss_innov.item(), global_step)
            writer.add_scalar("train/loss_kf", loss_kf.item(), global_step)
            writer.add_scalar("train/loss_apbm", loss_apbm.item(), global_step)
            writer.add_scalar("train/disc_loss", loss_D.item(), global_step)

            if global_step % config["log"]["eval_every"] == 0 and len(val_loader) > 0:
                val_metric = evaluate(generator, val_loader, device, R)
                writer.add_scalar("val/rmse", val_metric, global_step)
                if val_metric < best_metric:
                    best_metric = val_metric
                    utils.save_checkpoint(
                        Path(config["log"]["out_dir"]) / "best.pth",
                        generator_state=generator.state_dict(),
                        discriminator_state=discriminator.state_dict(),
                        optim_G_state=optim_G.state_dict(),
                        optim_D_state=optim_D.state_dict(),
                        step=global_step,
                        best_metric=best_metric,
                    )
            if global_step % config["log"]["save_every"] == 0:
                utils.save_checkpoint(
                    Path(config["log"]["out_dir"]) / f"ckpt_{global_step}.pth",
                    generator_state=generator.state_dict(),
                    discriminator_state=discriminator.state_dict(),
                    optim_G_state=optim_G.state_dict(),
                    optim_D_state=optim_D.state_dict(),
                    step=global_step,
                    best_metric=best_metric,
                )

            global_step += 1
            progress.update(1)

    progress.close()
    writer.close()


def evaluate(
    generator: GAKFGenerator,
    val_loader,
    device: torch.device,
    R: torch.Tensor,
) -> float:
    if len(val_loader) == 0:
        return float("inf")

    generator.eval()
    rmse_meter = utils.MetricAccumulator()

    with torch.no_grad():
        for y_batch, x_batch in val_loader:
            y_batch = y_batch.to(device)
            x_batch = x_batch.to(device)
            outs = generator(y_batch)
            mse = torch.mean((outs.x_post[:, 1:, :] - x_batch[:, 1:, :]) ** 2)
            rmse = torch.sqrt(mse).item()
            rmse_meter.update(rmse, y_batch.size(0))

    generator.train()
    return rmse_meter.avg


if __name__ == "__main__":
    main()
