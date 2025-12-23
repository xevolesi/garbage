import torch


def permute_reshape_concat(
    tensors: list[torch.Tensor],
    permute_size: tuple[int, ...],
    reshape_size: tuple[int, ...],
    cat_dim: int,
) -> torch.Tensor:
    return torch.cat(
        [t.permute(*permute_size).reshape(*reshape_size) for t in tensors],
        dim=cat_dim,
    )


def generate_priors_for_stride(
    stride: int,
    offset: float,
    map_size: tuple[int, int] | torch.Size,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate priors (simple uniform strided grid) for feature map of size
    `map_size` from stride `stride` output.

    Args:
        stride: Feature pyramid level stride, e.g. 8, 16, 32, 64, ...;
        offset: Optional offset for grid centers;
        map_size: Spatial shape of feature map;
        device: PyTorch device object.
    
    Returns:
        Regular grid for stride output with shape (B, N_priors, 4) in
        [center_x, center_y, stride_w, stride_h].
    """
    fm_h, fm_w = map_size
    stride_h, stride_w = stride, stride

    # Priors are actually a simple regular cartesian grid. But actual
    # coordinates are multiplied by stride values to from coordinates
    # in actual image dimensions. So, priors are just a point (x, y)
    # where x and y coordinates are simply points of a regular grid
    # multiplied by stride value and optional shifted with offset value.
    shift_x = (torch.arange(0, fm_w, device=device) + offset) * stride_w
    shift_y = (torch.arange(0, fm_h, device=device) + offset) * stride_h
    shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x)
    stride_w = torch.full((fm_w * fm_h,), stride_w, device=device, dtype=shift_xx.dtype)
    stride_h = torch.full((fm_h * fm_w,), stride_h, device=device, dtype=shift_yy.dtype)
    return torch.stack(
        [shift_xx.reshape(-1), shift_yy.reshape(-1), stride_w, stride_h],
        dim=-1
    )


def decode_boxes(grids: torch.Tensor, box_logits: torch.Tensor) -> torch.Tensor:
    """
    Decode predicted boxes from [center_x, center_y, with, height] to
    [top_left_x, top_left_y, bot_right_x, bot_right_y] using priors.
    Note that for YuNet priors are actually a simple strided grid.

    Args:
        grids: Grids from all strides with shape (B, N_priors, 4);
        box_logits: Predicted logits for boxes with shape (B, N_priors, 4).
    
    Returns:
        Decoded boxes with shape (B, N_priors, 4).
    """
    xys = (box_logits[..., :2] * grids[..., 2:]) + grids[..., :2]
    whs = box_logits[..., 2:].exp() * grids[..., 2:]

    top_left_x = (xys[..., 0] - whs[..., 0] / 2)
    top_left_y = (xys[..., 1] - whs[..., 1] / 2)
    bottom_right_x = (xys[..., 0] + whs[..., 0] / 2)
    bottom_right_y = (xys[..., 1] + whs[..., 1] / 2)

    return torch.stack(
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y], -1
    )


def postprocess_predictions(
    pyramid_outputs: tuple[torch.Tensor, ...],
    strides: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Postprocess raw models' predictions to appropriate shape.
    
    Args:
        obj_logits_stages: Objectness logits list from all pyramid levels.
            Shapes are (B, 1, image_h / stride, image_w / stride);
        cls_logits_stages: Class logits list from all pyramid levels.
            Shapes are (B, 1, image_h / stride, image_w / stride);
        box_logits_stages: Box logits list from all pyramid levels.
            Shapes are (B, 4, image_h / stride, image_w / stride);
        kps_logits_stages: Keypoints logits list from all pyramid levels.
            Shapes are (B, 10, image_h / stride, image_w / stride).
        strides: Output strides, e.g 2^(pyramid_level_idx) = (8, 16, 32, ...).
    
    Returns:
        Postprocessed objectness-, class-, box- and keypoints-logits and
        generated priors. Note that each postprocess item are single tensor,
        so it contains concatenated logits from all provided pyramid levels.
    """
    # Let's gather all items of one type from all pyramid levels in single
    # list. Each item in relust lists has the following shape:
    # obj_logits: (B, 1, image_h / stride, image_w / stride);
    # cls_logits: (B, 1, image_h / stride, image_w / stride);
    # box_logits: (B, 4, image_h / stride, image_w / stride);
    # kps_logits: (B, 10, image_h / stride, image_w / stride).
    obj_logits_pyramids = [out[0] for out in pyramid_outputs]
    cls_logits_pyramids = [out[1] for out in pyramid_outputs]
    box_logits_pyramids = [out[2] for out in pyramid_outputs]
    kps_logits_pyramids = [out[3] for out in pyramid_outputs]

    device = obj_logits_pyramids[0].device
    n_images = cls_logits_pyramids[0].shape[0]
    fm_sizes = [cls_logits.shape[2:] for cls_logits in cls_logits_pyramids]

    # Generate priors for all pyramid levels. Final shape will be
    # (B, N_priors_level_1 + N_priors_level_2 + ... N_priors_level_m, 4) and
    # it depends on how many pyramid levels we chose for models' outputs.
    # For example, if we take 3 levels of pyramid
    # [(H/8, W/8), (H/16, W/16), (H/32, W/32)] we get [8, 16, 32] strides.
    # So, let's imagine that we have (640, 640) image as input. (H/8, W/8)
    # pyramid for (640, 640) image has (80, 80) spatial size. So, we have
    # 80 * 80 = 1600 priors for this pyramid level. Same stands for other
    # pyramid levels. So, for our example actual priors shape for all
    # pyramid levels will be (B, 80*80 + 40*40 + 20*20, 4) = (B, 8400, 4).
    priors = torch.cat(
        [
            generate_priors_for_stride(stride, 0.0, size, device)
            for stride, size in zip(strides, fm_sizes, strict=True)
        ]
    ).unsqueeze(0).repeat(n_images, 1, 1)

    # Move channels to last dimension, reshape and concat for all pyramid
    # levels. Our goal is to get similar to priors shape, since for each
    # prior we need corresponding predictions. So, expected shapes are:
    # obj_logits: (B, N_priors_level_1 + N_priors_level_2 + ... N_priors_level_m)
    # cls_logits: (B, N_priors_level_1 + N_priors_level_2 + ... N_priors_level_m, 1)
    # box_logits: (B, N_priors_level_1 + N_priors_level_2 + ... N_priors_level_m, 4)
    # kps_logits: (B, N_priors_level_1 + N_priors_level_2 + ... N_priors_level_m, 10) 
    perm_size = (0, 2, 3, 1)
    obj_logits = permute_reshape_concat(
        obj_logits_pyramids, perm_size, (n_images, -1), 1
    )
    cls_logits = permute_reshape_concat(
        cls_logits_pyramids, perm_size, (n_images, -1, 1), 1
    )
    box_logits = permute_reshape_concat(
        box_logits_pyramids, perm_size, (n_images, -1, 4), 1
    )
    kps_logits = permute_reshape_concat(
        kps_logits_pyramids, perm_size, (n_images, -1, 10), 1
    )

    # Translate boxes from prediction format to
    # [top_left_x, top_left_y, bot_right_x, bot_right_y].
    # Prediction format is [x_center, y_center, width, height].
    boxes = decode_boxes(priors, box_logits)
    return obj_logits, cls_logits, boxes, kps_logits, priors