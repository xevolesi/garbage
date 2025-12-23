import torch
from torchvision.ops import box_iou
from torch.nn.functional import one_hot, binary_cross_entropy


def is_in(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Check if points lies insde boxes. Vectorized variant.

    Args:
        points: (X, Y) points with shape (N, 2);
        boxes: Boxes with shape (M, 4) in
            [topl_left_x, top_left_y, bot_right_x, bot_right_y] format.
    
    Returns: (M, N) inclustion matrix C where c_ij is a bool value showing
        that j-th point lies inside the i-th box.
    """
    # [:, None, :2] to abuse broadcasting.
    top_left_points = boxes[:, None, :2]
    bot_right_points = boxes[:, None, 2:]
    min_distances = torch.cat(
        [points - top_left_points, bot_right_points - points], dim=-1
    ).min(dim=-1).values
    # If min distance from all 4 edges of the box to center point is grater then 0.0
    # than center point are strictly inside the box.
    return torch.gt(min_distances, 0.0)


def are_priors_in_gts(
    priors: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Check if priors lies inside the GT boxes.

    Args:
        priors: Grid priors with shape (N, 4) in
            [center_x, center_y, stride_w, stride_h] format;
        gt_boxes: GT boxes with shape (M, 4) in
            [top_left_x, top_left_y, bot_right_x, bot_right_y] format.

    Return:
        Inclusion matrix C (M, N) where c_ij is a bool value showing that
            j-th prior lies inside the i-th GT box.
    """
    return is_in(priors[:, :2], gt_boxes)


def are_priors_in_gts_center(
    priors: torch.Tensor, gt_boxes: torch.Tensor, r: float = 0.5
) -> torch.Tensor:
    """
    Check if priors lies inside the U(b, r * stride), U is defined as L1-ball.

    Args:
        priors: Grid priors with shape (N, 4) in
            [center_x, center_y, stride_w, stride_h] format;
        gt_boxes: GT boxes with shape (M, 4) in
            [top_left_x, top_left_y, bot_right_x, bot_right_y] format;
        r: B1 ball radius coefficient. Actual radius will be
            (r * stride_w, r * stride_h).

    Return:
        Inclusion matrix C with shape (M, N) where c_ij is a bool value showing that
            j-th prior lies inside the i-th GT box center U.
    """
    gt_centers = 0.5 * (gt_boxes[:, 2:] + gt_boxes[:, :2])
    center_rad = r * priors[0, 2:] # This is [stride_w, stride_h] which is the same for all priors.
    center_boxes = torch.cat([gt_centers - center_rad, gt_centers + center_rad], dim=-1)
    return are_priors_in_gts(priors, center_boxes)


def compute_inclusion_masks(
        priors: torch.Tensor, gt_boxes: torch.Tensor, r: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    in_gt = are_priors_in_gts(priors, gt_boxes) # (num_gt, num_priors)
    in_center_boxes = are_priors_in_gts_center(priors, gt_boxes, r=r) # (num_gt, num_priors)
    in_gt_or_in_center_boxes = in_gt.any(dim=0) | in_center_boxes.any(dim=0) # (num_priors,)
    in_gt_and_in_center_boxes = (
        in_gt[:, in_gt_or_in_center_boxes] &
        in_center_boxes[:, in_gt_or_in_center_boxes]
    )# (num_gt, num_valid)
    return in_gt_or_in_center_boxes, in_gt_and_in_center_boxes


def dynamic_k_matching(
    cost_matrix: torch.Tensor, ious: torch.Tensor, topk: int = 10
):
    num_gt, num_valid = cost_matrix.shape
    # Compute dynamic K for each GT box.
    k = min(topk, num_valid)
    top_ious, _ = torch.topk(ious, k, dim=1)

    if num_valid == 0:
        
        print("cost_matrix", cost_matrix.shape)
        print("ious", ious.shape)
        num_gt, num_valid = ious.shape
        print("num_gt", num_gt, "num_valid", num_valid)
        raise RuntimeError("No valid priors in dynamic_k_matching")
    ks = torch.clamp(top_ious.sum(1).int(), min=1).tolist()
    
    # For i-th GT box select the best ks[i] predictions from valid boxes.
    matching_matrix = ious.new_zeros(ious.shape)
    for gt_idx in range(num_gt):
        _, okay_idx = torch.topk(cost_matrix[gt_idx, :], k=ks[gt_idx], largest=False)
        matching_matrix[gt_idx, okay_idx] = 1
    
    # Can i reimplement this loop in a vectorized fashion?
    for j in range(num_valid):
        matched_gt = matching_matrix[:, j]

        # It's okay if we have 1 GT for 1 valid box or 0 gt for 1 valid box.
        if matched_gt.sum() <= 1:
            continue
        
        # Match best GT in terms of cost with j-th valid predicted box.
        gt_ids = torch.nonzero(matched_gt, as_tuple=False).squeeze(dim=1)
        best_gt = gt_ids[torch.argmin(cost_matrix[gt_ids, j])]
        matching_matrix[:, j] = 0
        matching_matrix[best_gt, j] = 1
    fg_mask_valid = matching_matrix.sum(dim=0) > 0
    matched_gt_valid = matching_matrix.argmax(dim=0)
    matched_ious_valid = (matching_matrix * ious).sum(0)[fg_mask_valid]
    return fg_mask_valid, matched_gt_valid, matched_ious_valid


def simota_assign_per_image(
    cls_probas: torch.Tensor,
    priors: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    r: float = 2.5,
    topk: int = 10,
    iou_weight: float = 3.0,
    cls_weight: float = 1.0,
):
    """
    Perform SimOTA algorithm to assign targets to predictions.

    Args:
        cls_probas: Probabilities for predicted boxes with shape (N_box, n_cls);
        priors: Priors for boxes with shape (N_box, 4) in
            [x_center, y_center, stride_w, stride_h] format;
        boxes_xyxy: Predicted boxes with shape (N_box, 4) in format
            [top_left_x, top_left_y, bot_right_x, bot_right_y];
        gt_boxes: GT boxes with shape (M_box, 4) in format
            [top_left_x, top_left_y, bot_right_x, bot_right_y];
        gt_labels: GT labels for GT boxes with shape (M_box,);
    
    Returns:
        num_gt: Количество
    """
    num_gt = gt_boxes.size(0)
    num_boxes = boxes_xyxy.size(0)
    num_classes = cls_probas.size(-1)

    assigned_gt_ids = boxes_xyxy.new_full((num_boxes,), -1, dtype=torch.long)
    assigned_labels = boxes_xyxy.new_full((num_boxes,), -1, dtype=torch.long)
    if num_gt == 0 or num_boxes == 0:
        return assigned_gt_ids, assigned_labels

    # Let K be the number of valid boxes. So,
    # valid_mask has shape (N_box,), in_gt_and_in_center_boxes has shape
    # (M_box, K).
    valid_mask, in_gt_and_in_center_boxes = compute_inclusion_masks(priors, gt_boxes, r)
    valid_decoded_boxes = boxes_xyxy[valid_mask] # (K, 4)
    valid_probas = cls_probas[valid_mask]        # (K, 1)
    num_valid = valid_decoded_boxes.size(0)

    if num_valid == 0:
        return assigned_gt_ids, assigned_labels

    # IoU cost for cost matrix. Shape is (M_box, K).
    ious = box_iou(gt_boxes, valid_decoded_boxes)
    iou_cost = -torch.log(ious + 1e-7)


    # CE cost for cost matrix. Shape is (M_box, K)
    gt_onehot_label = one_hot(gt_labels.long(), num_classes)    # (M_box, n_cls)
    gt_onehot_label = gt_onehot_label.unsqueeze(0).repeat(num_valid, 1, 1).float()
    valid_probas = valid_probas.unsqueeze(1).repeat(1, num_gt, 1)
    cls_cost = binary_cross_entropy(valid_probas.sqrt(), gt_onehot_label, reduction="none")
    cls_cost = cls_cost.sum(-1).T

    cost_matrix = cls_cost * cls_weight + iou_cost * iou_weight + (~in_gt_and_in_center_boxes) * 10_000
    fg_mask_valid, matched_gt_valid, matched_ious_valid = dynamic_k_matching(cost_matrix, ious, topk)

    valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
    fg_valid_j = fg_mask_valid.nonzero(as_tuple=False).squeeze(1)
    fg_valid_idx = valid_idx[fg_valid_j]
    fg_matched_gt = matched_gt_valid[fg_valid_j] 
    assigned_gt_ids[fg_valid_idx] = fg_matched_gt
    assigned_labels[fg_valid_idx] = gt_labels[fg_matched_gt].long()
    return assigned_gt_ids, assigned_labels