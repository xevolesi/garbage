import random
import numpy as np
from albumentations.core.transforms_interface import DualTransform


class RandomSquareCrop(DualTransform):
    """Random square crop with padding and retry mechanism (YuNet style).
    
    Args:
        crop_ratio_range (tuple): Range for crop scale, e.g. (0.5, 2.0)
        crop_choice (list): List of crop scales, e.g. [0.5, 1.0, 1.5, 2.0]
        bbox_clip_border (bool): Whether to clip bboxes to crop border. Default: True
        fill_value (int): Padding fill value. Default: 128
        always_apply (bool): Whether to always apply transform. Default: False
        p (float): Probability of applying the transform. Default: 1.0
    """
    
    def __init__(self, 
                 crop_ratio_range=None, 
                 crop_choice=None,
                 bbox_clip_border=True,
                 fill_value=128, 
                 always_apply=False, 
                 p=1.0):
        super().__init__(always_apply, p)
        
        assert (crop_ratio_range is None) ^ (crop_choice is None), \
            "Either crop_ratio_range or crop_choice must be specified, but not both"
        
        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = crop_choice
        self.bbox_clip_border = bbox_clip_border
        self.fill_value = fill_value
        
        if self.crop_ratio_range is not None:
            self.crop_ratio_min, self.crop_ratio_max = self.crop_ratio_range
    
    @property
    def targets_as_params(self):
        return ["image", "bboxes"]
    
    def get_params_dependent_on_targets(self, params):
        """Calculate crop parameters with retry mechanism."""
        img = params["image"]
        bboxes = np.array(params.get("bboxes", []))
        h, w = img.shape[:2]
        
        if self.crop_ratio_range is not None:
            max_scale = self.crop_ratio_max
        else:
            max_scale = np.amax(self.crop_choice)
        
        scale_retry = 0
        scale = None
        
        while True:
            scale_retry += 1
            
            # Определяем scale на текущей итерации
            if scale_retry == 1 or max_scale > 1.0:
                if self.crop_ratio_range is not None:
                    scale = np.random.uniform(self.crop_ratio_min, self.crop_ratio_max)
                elif self.crop_choice is not None:
                    scale = np.random.choice(self.crop_choice)
            else:
                # Увеличиваем scale если предыдущая попытка не удалась
                scale = scale * 1.2
            
            # Пытаемся найти подходящий crop с текущим scale
            for i in range(250):
                short_side = min(w, h)
                cw = int(scale * short_side)
                ch = cw
                
                # Вычисляем позицию crop (может быть отрицательной для padding)
                if w == cw:
                    left = 0
                elif w > cw:
                    left = random.randint(0, w - cw)
                else:
                    left = random.randint(w - cw, 0)
                
                if h == ch:
                    top = 0
                elif h > ch:
                    top = random.randint(0, h - ch)
                else:
                    top = random.randint(h - ch, 0)
                
                patch = np.array([int(left), int(top), int(left + cw), int(top + ch)])
                
                # Проверяем что центры bbox находятся внутри crop
                if len(bboxes) > 0:
                    # bbox уже в абсолютных координатах (pascal_voc format)
                    abs_bboxes = np.array(bboxes)
                    
                    center = (abs_bboxes[:, :2] + abs_bboxes[:, 2:]) / 2
                    mask = ((center[:, 0] > patch[0]) &
                            (center[:, 1] > patch[1]) &
                            (center[:, 0] < patch[2]) &
                            (center[:, 1] < patch[3]))
                    
                    if not mask.any():
                        continue
                
                # Успешно нашли подходящий crop
                return {
                    "patch": patch,
                    "crop_height": ch,
                    "crop_width": cw,
                    "orig_height": h,
                    "orig_width": w
                }
            
            # Если не нашли за 250 итераций и retry больше 10 - возвращаем fallback
            if scale_retry > 10:
                # Fallback - crop по центру с текущим scale
                short_side = min(w, h)
                cw = int(scale * short_side)
                ch = cw
                left = (w - cw) // 2
                top = (h - ch) // 2
                patch = np.array([left, top, left + cw, top + ch])
                
                return {
                    "patch": patch,
                    "crop_height": ch,
                    "crop_width": cw,
                    "orig_height": h,
                    "orig_width": w
                }
    
    def apply(self, img, patch, crop_height, crop_width, **params):
        """Apply crop to image with padding."""
        h, w = img.shape[:2]
        
        # Создаём выходное изображение с заливкой
        if isinstance(self.fill_value, (int, float)):
            rimg = np.ones((crop_height, crop_width, img.shape[2]), 
                          dtype=img.dtype) * self.fill_value
        else:
            rimg = np.ones((crop_height, crop_width, img.shape[2]), 
                          dtype=img.dtype)
            rimg[:, :] = self.fill_value
        
        # Вычисляем область откуда копируем (из исходного изображения)
        patch_from = patch.copy()
        patch_from[0] = max(0, patch_from[0])
        patch_from[1] = max(0, patch_from[1])
        patch_from[2] = min(w, patch_from[2])
        patch_from[3] = min(h, patch_from[3])
        
        # Вычисляем область куда копируем (в результирующее изображение)
        patch_to = patch.copy()
        patch_to[0] = max(0, -patch_to[0])
        patch_to[1] = max(0, -patch_to[1])
        patch_to[2] = patch_to[0] + (patch_from[2] - patch_from[0])
        patch_to[3] = patch_to[1] + (patch_from[3] - patch_from[1])
        
        # Копируем данные
        rimg[patch_to[1]:patch_to[3], patch_to[0]:patch_to[2]] = \
            img[patch_from[1]:patch_from[3], patch_from[0]:patch_from[2]]
        
        return rimg
    
    def apply_to_bbox(self, bbox, patch, crop_width, crop_height, 
                      orig_width, orig_height, **params):
        """Apply crop to bounding box (pascal_voc format - absolute coordinates)."""
        # bbox уже в абсолютных координатах [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox
        
        # Проверяем центр bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        if not (patch[0] < center_x < patch[2] and patch[1] < center_y < patch[3]):
            # Центр вне crop - удаляем bbox
            return None
        
        # Clip к границам patch если нужно
        if self.bbox_clip_border:
            x_max = min(x_max, patch[2])
            y_max = min(y_max, patch[3])
            x_min = max(x_min, patch[0])
            y_min = max(y_min, patch[1])
        
        # Смещаем относительно crop
        x_min -= patch[0]
        y_min -= patch[1]
        x_max -= patch[0]
        y_max -= patch[1]
        
        # Возвращаем в абсолютных координатах относительно нового crop
        return [x_min, y_min, x_max, y_max]
    
    def apply_to_keypoint(self, keypoint, patch, crop_width, crop_height,
                         orig_width, orig_height, **params):
        """Apply crop to keypoint (xy format - absolute coordinates)."""
        # keypoint в формате [x, y, angle, scale] - абсолютные координаты
        x, y, angle, scale = keypoint
        
        # Clip к границам patch если нужно
        if self.bbox_clip_border:
            x = np.clip(x, patch[0], patch[2])
            y = np.clip(y, patch[1], patch[3])
        
        # Смещаем относительно crop
        x -= patch[0]
        y -= patch[1]
        
        # Возвращаем в абсолютных координатах относительно нового crop
        return [x, y, angle, scale]
    
    def get_transform_init_args_names(self):
        return ("crop_ratio_range", "crop_choice", "bbox_clip_border", "fill_value")
