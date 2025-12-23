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
    # если gt_kps имеет форму [M,5,2]
    if gt_kps.dim() == 3:
        num_kps = gt_kps.size(1) * gt_kps.size(2)
    else:
        # на всякий случай, если уже [M,10]
        num_kps = gt_kps.size(1)

    # No target
    if num_gts == 0:
        foreground_mask = torch.zeros((num_priors,), device=device, dtype=torch.bool)
        target_obj = torch.zeros((num_priors,), device=device, dtype=dtype)
        target_cls = torch.zeros((num_priors, num_classes), device=device, dtype=dtype)
        target_boxes = torch.zeros((num_priors, 4), device=device, dtype=dtype)
        # для kps заполняем -1, чтобы маской потом игнорировать
        target_kps = torch.full((num_priors, num_kps), -1.0, device=device, dtype=dtype)
        return foreground_mask, target_cls, target_obj, target_boxes, target_kps

    # Uses center priors with 0.5 offset to assign targets,
    # but use center priors without offset to regress bboxes.
    offset_priors = torch.cat(
        [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1
    )

    assigned_gt_ids, assigned_labels = simota_assign_per_image(
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
    target_kps = torch.zeros((num_priors, num_kps), device=device, dtype=dtype)

    fg_inds = foreground_mask.nonzero(as_tuple=False).squeeze(1)
    fg_gt_inds = assigned_gt_ids[fg_inds]
    fg_gt_labels = assigned_labels[fg_inds]

    target_obj[fg_inds] = 1.0

    target_cls[fg_inds] = torch.nn.functional.one_hot(
        fg_gt_labels.long(), num_classes
    ).to(dtype)

    target_boxes[fg_inds] = gt_boxes[fg_gt_inds].to(dtype)

    # кейпоинты только если gt_kps не пустой
    if gt_kps.numel() > 0:
        if gt_kps.dim() == 3:
            gt_kps_flat = gt_kps.view(gt_kps.size(0), -1).to(dtype)  # [M, num_kps]
        else:
            gt_kps_flat = gt_kps.to(dtype)  # уже [M, num_kps]
        target_kps[fg_inds] = gt_kps_flat[fg_gt_inds]

    return foreground_mask, target_cls, target_obj, target_boxes, target_kps


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
        obj_preds: Objectness logits with shape
            (B, N_priors_lvl1 + ... + N_priors_lvlM);
        cls_preds: Classification logits with shape
            (B, N_priors_lvl1 + ... + N_priors_lvlM, 1);
        box_preds: Bounding box logits with shape
            (B, N_priors_lvl1 + ... + N_priors_lvlM, 4);
        grids: Priors with shape 
            (B, N_priors_lvl1 + ... + N_priors_lvlM, 4);
        gt_boxes: List of length B where each item is a GT boxes of shape
            (N_boxes, 4) in [top_left_x, top_left_y, bot_right_x, bot_right_y]
            format;
        gt_kps: List of length B where each item is a GT key points of shape
            (N_boxes, 5, 2);
        device: PyTorch device.
    
    Returns:
        Target tensors of shape (B, N_priors_lvl1 + ... + N_priors_lvlM, *)
        where *:
        - Nothing for foreground mask;
        - Nothing for objectness target tensor;
        - 1 for classification target tensor;
        - 4 for boxes target tensor;
        - 10 for key points target tensor
    """
    batch_size = len(gt_boxes)
    gt_labels = [
        torch.zeros((len(gt_boxes[i]),), device=device)
        for i in range(batch_size)
    ]
    per_image_targets = []
    for i in range(batch_size):
        # Targets as tuple:
        # # (foreground_mask_i, target_cls_i, target_obj_i, target_boxes_i, target_kps_i)
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

    # Transposing: now we have list of lists of particular targets.
    per_target_lists = list(zip(*per_image_targets))  # len = num_fields
    # Concatenate along batch dimension.
    batched_targets = tuple(torch.cat(t_list, dim=0) for t_list in per_target_lists)
    return batched_targets