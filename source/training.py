from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from source.models import YuNet
from source.postprocessing import postprocess_predictions
from source.schedulers import WarmupMultiStepLR
from source.targets import generate_targets_batch


def train_one_epoch(
    model: YuNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupMultiStepLR,
    criterion: torch.nn.Module | torch.nn.modules.loss._Loss,
    device: torch.device,
) -> dict[str, float]:
    """
    Train model for single epoch.

    NOTE: Simple training loop without distributed training, mixed precision training, etc.

    Args:
        model: Model to train.
        dataloader: DataLoader for training.
        optimizer: Optimizer for training.
        scheduler: Scheduler for training.
        criterion: Loss function for training.
        device: Device to train on.

    Returns:
        Dictionary with running losses.
            "train_total_loss": Total loss.
            "train_obj_loss": Objectness loss.
            "train_cls_loss": Classification loss.
            "train_box_loss": Box loss.
            "train_kps_loss": Keypoints loss.
    """
    model.train()
    running_losses: defaultdict[str, float] = defaultdict(float)
    for batch in dataloader:
        optimizer.zero_grad()

        # Put training data on target device.
        images = batch["image"].to(device, non_blocking=True)
        boxes = [item.to(device, non_blocking=True) for item in batch["boxes"]]
        kps = [item.to(device, non_blocking=True) for item in batch["key_points"]]

        # Get raw predictions from model.
        p8_out, p16_out, p32_out = model(images)

        # Postprocess raw predictions to form objectness, class, box and keypoints logits
        obj_preds, cls_preds, box_preds, kps_preds, grids = postprocess_predictions(
            (p8_out, p16_out, p32_out), (8, 16, 32)
        )

        # Generate targets for training.
        # Assignment + matching.
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

        # Compute losses.
        targets = (target_obj, target_cls, target_boxes, target_kps, kps_weights)
        inputs = (obj_preds, cls_preds, box_preds, kps_preds)
        loss_dict: dict[str, torch.Tensor] = criterion(
            inputs, targets, foreground_mask, grids
        )

        # Gradient step.
        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()
        scheduler.step_iter()

        # Log losses.
        for loss_name, loss_tensor in loss_dict.items():
            loss_value = loss_tensor.detach().cpu().item()
            running_losses[f"train_{loss_name}"] += loss_value / len(dataloader)
    return dict(running_losses)
