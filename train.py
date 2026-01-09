import argparse as ap
import os
import time
from collections import defaultdict

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from source.config import Config
from source.dataset import build_dataloaders
from source.drawing import visualize_epoch_predictions
from source.general import (
    get_cpu_state_dict,
    load_optimizer_state_dict,
    seed_everything,
)
from source.losses import DetectionLoss
from source.metrics import calculate_map_torchmetrics
from source.models import YuNet
from source.postprocessing import postprocess_predictions
from source.schedulers import WarmupMultiStepLR
from source.targets import generate_targets_batch

# Set benchmark to True and deterministic to False
# if you want to speed up training with less level of reproducibility.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Speed up GEMM if GPU allowed to use TF32.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def read_config(path: str) -> Config:
    with open(path, "r") as yaml_file:
        yml = yaml.safe_load(yaml_file)
    return Config.model_validate(yml)


def train_one_epoch(
    model: YuNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupMultiStepLR,
    criterion: torch.nn.Module | torch.nn.modules.loss._Loss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_losses: defaultdict[str, float] = defaultdict(float)
    for batch in dataloader:
        optimizer.zero_grad()
        images = batch["image"].to(device, non_blocking=True)
        boxes = [item.to(device, non_blocking=True) for item in batch["boxes"]]
        kps = [item.to(device, non_blocking=True) for item in batch["key_points"]]
        p8_out, p16_out, p32_out = model(images)
        obj_preds, cls_preds, box_preds, kps_preds, grids = postprocess_predictions(
            (p8_out, p16_out, p32_out), (8, 16, 32)
        )
        (
            foreground_mask,
            target_cls,
            target_obj,
            target_boxes,
            target_kps,
            kps_weights,
        ) = generate_targets_batch(
            obj_preds, cls_preds, box_preds, grids, boxes, kps, device
        )
        targets = (target_obj, target_cls, target_boxes, target_kps, kps_weights)
        inputs = (obj_preds, cls_preds, box_preds, kps_preds)
        loss_dict: dict[str, torch.Tensor] = criterion(
            inputs, targets, foreground_mask, grids
        )
        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()
        scheduler.step_iter()

        for loss_name, loss_tensor in loss_dict.items():
            loss_value = loss_tensor.detach().cpu().item()
            running_losses[f"train_{loss_name}"] += loss_value / len(dataloader)
    return dict(running_losses)


def train(config: Config, dataframe: pd.DataFrame):
    device = torch.device(config.training.device)
    dataloaders = build_dataloaders(config, dataframe)
    model = YuNet(**config.model.model_dump())
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.training.lr,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
    )
    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=[400, 544],
        gamma=0.1,
        warmup_iters=1500,
        warmup_ratio=0.001,
        warmup_by_epoch=False,
    )
    criterion = DetectionLoss(
        obj_weight=1.0, cls_weight=1.0, box_weight=5.0, kps_weight=0.1
    )
    start_epoch = 0
    log_dir = os.path.join(config.path.artifacts_folder, "tb_logs")
    if config.training.resume_ckpt is not None:
        ckpt = torch.load(config.training.resume_ckpt)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model"])
        load_optimizer_state_dict(optimizer, ckpt["optimizer"], device)
        scheduler.load_state_dict(ckpt["scheduler"])
        criterion.load_state_dict(ckpt["criterion"])
        resume_log_dir = ckpt.get("tb_writer_log_dir")
        log_dir = resume_log_dir if resume_log_dir is not None else log_dir
        os.makedirs(log_dir, exist_ok=True)

    model = model.to(device)
    tb_writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, config.training.epochs):
        train_start_time = time.perf_counter()
        train_losses = train_one_epoch(
            model, dataloaders["train"], optimizer, scheduler, criterion, device
        )
        train_end_time = time.perf_counter()
        train_time = train_end_time - train_start_time
        scheduler.step()
        metrics = {}
        if (epoch + 1) % config.training.eval_interval == 0:
            metrics = calculate_map_torchmetrics(model, dataloaders["val"], device)
            visualize_epoch_predictions(
                config.path.artifacts_folder,
                epoch,
                model,
                next(iter(dataloaders["val"])),
                device,
            )

        tb_writer.add_scalars("Losses", train_losses, global_step=epoch)
        for loss_name, loss_value in train_losses.items():
            tb_writer.add_scalar(loss_name, loss_value, epoch)
        tb_writer.add_scalars("Metrics", metrics, global_step=epoch)
        for metric_name, metric_value in metrics.items():
            tb_writer.add_scalar(metric_name, metric_value, epoch)
        tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step=epoch)

        log_str = ", ".join(
            [
                f"{loss_name}={loss_value:.4f}"
                for loss_name, loss_value in train_losses.items()
            ]
        )
        if metrics:
            log_str = (
                log_str
                + " "
                + ", ".join(
                    [
                        f"{metric_name}={metric_value:.4f}"
                        for metric_name, metric_value in metrics.items()
                    ]
                )
            )
        log_str = log_str + f", Time: {train_time:.2f}s"
        print(f"[EPOCH {epoch + 1}/{config.training.epochs}] {log_str}")
        ckpt = {
            "epoch": epoch,
            "model": get_cpu_state_dict(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "criterion": criterion.state_dict(),
            "tb_writer_log_dir": tb_writer.get_logdir(),
        }
        ckpt_path = os.path.join(config.path.artifacts_folder, "checkpoints")
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_path = os.path.join(ckpt_path, f"epoch_{epoch}_ckpt.pt")
        torch.save(ckpt, ckpt_path)


def main(args: ap.Namespace) -> None:
    config = read_config(args.config)
    if config.path.run_name is None:
        msg = "Run name is not provided in the config file. Using default run name."
        raise ValueError(msg)

    config.path.artifacts_folder = os.path.join(
        config.path.artifacts_folder, config.path.run_name
    )
    os.makedirs(config.path.artifacts_folder, exist_ok=True)
    dataframe = pd.read_csv(config.path.csv)

    seed_everything(config)
    train(config, dataframe)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args)
