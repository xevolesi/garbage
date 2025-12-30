import torch
from .simota import simota_assign_per_image


def generate_targets(
    cls_logits: torch.Tensor,
    obj_logits: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    priors: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_kps: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    device = cls_logits.device
    dtype = boxes_xyxy.dtype
    gt_boxes = gt_boxes.to(boxes_xyxy.dtype)
    gt_kps = gt_kps.to(boxes_xyxy.dtype)

    num_gts = gt_labels.size(0)
    num_priors = priors.size(0)
    num_classes = cls_logits.size(1)
    num_kps = 5 * 2

    # No target
    if num_gts == 0:
        foreground_mask = torch.zeros((num_priors,), device=device, dtype=torch.bool)
        target_obj = torch.zeros((num_priors,), device=device, dtype=dtype)
        target_cls = torch.zeros((num_priors, num_classes), device=device, dtype=dtype)
        target_boxes = torch.zeros((num_priors, 4), device=device, dtype=dtype)
        target_kps = torch.full(
            (num_priors, num_kps), -1.0, device=device, dtype=dtype
        )  # ← -1.0
        kps_weights = torch.zeros((num_priors, 1), device=device, dtype=dtype)
        return (
            foreground_mask,
            target_cls,
            target_obj,
            target_boxes,
            target_kps,
            kps_weights,
        )

    offset_priors = torch.cat(
        [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1
    )

    assigned_gt_ids, assigned_labels, pos_ious = simota_assign_per_image(
        cls_logits.sigmoid() * obj_logits.unsqueeze(1).sigmoid(),
        offset_priors,
        boxes_xyxy,
        gt_boxes,
        gt_labels,
    )

    foreground_mask = assigned_gt_ids >= 0
    dtype = boxes_xyxy.dtype

    target_obj = torch.zeros((num_priors,), device=device, dtype=dtype)
    target_cls = torch.zeros((num_priors, num_classes), device=device, dtype=dtype)
    target_boxes = torch.zeros((num_priors, 4), device=device, dtype=dtype)
    target_kps = torch.full(
        (num_priors, num_kps), -1.0, device=device, dtype=dtype
    )  # ← -1.0!
    kps_weights = torch.zeros((num_priors, 1), device=device, dtype=dtype)

    fg_inds = foreground_mask.nonzero(as_tuple=False).squeeze(1)
    fg_gt_inds = assigned_gt_ids[fg_inds]
    fg_gt_labels = assigned_labels[fg_inds]

    target_obj[fg_inds] = 1.0

    target_cls[fg_inds] = torch.nn.functional.one_hot(
        fg_gt_labels.long(), num_classes
    ).to(dtype) * pos_ious.unsqueeze(-1)

    target_boxes[fg_inds] = gt_boxes[fg_gt_inds].to(dtype)

    # Keypoint targets with visibility weights
    if gt_kps.numel() > 0:
        gt_kps_coords = gt_kps[..., :2]
        gt_kps_visibility = gt_kps[..., 2]

        gt_kps_flat = gt_kps_coords.reshape(gt_kps_coords.size(0), -1).to(dtype)

        target_kps[fg_inds] = gt_kps_flat[fg_gt_inds]

        kps_weight_vals = torch.mean(gt_kps_visibility[fg_gt_inds], dim=1, keepdim=True)
        kps_weights[fg_inds] = kps_weight_vals

    return (
        foreground_mask,
        target_cls,
        target_obj,
        target_boxes,
        target_kps,
        kps_weights,
    )


def generate_targets_batch(
    obj_preds: torch.Tensor,
    cls_preds: torch.Tensor,
    box_preds: torch.Tensor,
    grids: torch.Tensor,
    gt_boxes: list[torch.Tensor],
    gt_kps: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """
    Batched generate targets.

    Args:
        obj_preds: (B, N_priors)
        cls_preds: (B, N_priors, 1)
        box_preds: (B, N_priors, 4)
        grids: (B, N_priors, 4)
        gt_boxes: List[Tensor] of shape (M_i, 4)
        gt_kps: List[Tensor] of shape (M_i, 5, 3)
        device: torch.device

    Returns:
        Tuple of 6 tensors:
        - foreground_masks: (B, N_priors)
        - target_cls: (B, N_priors, 1)
        - target_obj: (B, N_priors)
        - target_boxes: (B, N_priors, 4)
        - target_kps: (B, N_priors, 10)
        - kps_weights: (B, N_priors, 1)
    """
    batch_size = len(gt_boxes)
    gt_labels = [
        torch.zeros((len(gt_boxes[i]),), device=device) for i in range(batch_size)
    ]

    per_image_targets = []
    for i in range(batch_size):
        # Targets as tuple:
        # (foreground_mask_i, target_cls_i, target_obj_i, target_boxes_i, target_kps_i, kps_weights_i)
        targets_i = generate_targets(
            cls_preds[i],
            obj_preds[i],
            box_preds[i],
            grids[i],
            gt_boxes[i],
            gt_labels[i],
            gt_kps[i],
        )
        per_image_targets.append([t.unsqueeze(0) for t in targets_i])

    # Transposing: now we have list of 6 lists of particular targets
    per_target_lists = list(zip(*per_image_targets))  # len = 6 (num fields)

    # Concatenate along batch dimension
    batched_targets = tuple(torch.cat(t_list, dim=0) for t_list in per_target_lists)

    return batched_targets
