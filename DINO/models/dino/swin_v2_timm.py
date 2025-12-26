"""
Swin-V2 backbone через timm для DINO
Поддерживает переменный размер входа для object detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging

from util.misc import NestedTensor

logger = logging.getLogger(__name__)


# Mapping для удобства
SWINV2_MODELS = {
    'swinv2_large_window12_192': {
        'name': 'swinv2_large_window12_192',
        'num_features': [192, 384, 768, 1536],
    },
    'swinv2_large_window12to16_192to256': {
        'name': 'swinv2_large_window12to16_192to256',
        'num_features': [192, 384, 768, 1536],
    },
    'swinv2_large_window12to24_192to384': {
        'name': 'swinv2_large_window12to24_192to384_22kft1k',
        'num_features': [192, 384, 768, 1536],
    },
    'swinv2_base_window8_256': {
        'name': 'swinv2_base_window8_256',
        'num_features': [128, 256, 512, 1024],
    },
    'swinv2_base_window12_192': {
        'name': 'swinv2_base_window12_192',
        'num_features': [128, 256, 512, 1024],
    },
}


class SwinV2Backbone(nn.Module):
    """
    Swin-V2 backbone через timm с динамическим размером входа
    """
    
    def __init__(self, model_name='swinv2_large_window12_192', pretrained=True, 
                 out_indices=(1, 2, 3), use_checkpoint=False):
        super().__init__()
        
        self.out_indices = out_indices
        self.model_name = model_name
        
        logger.info(f"[Swin-V2] Создание backbone: {model_name}")
        logger.info(f"[Swin-V2] Pretrained: {pretrained}, out_indices: {out_indices}")
        
        # Создаём модель с features_only=True
        # Указываем img_size больше, чем по умолчанию, и window_size
        # Это позволит модели работать с разными размерами
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            # Указываем кратный размер для совместимости
            img_size=384,  # Кратно 32 и window_size
        )
        
        # Получаем информацию о каналах
        if hasattr(self.model, 'feature_info'):
            self.num_features = [info['num_chs'] for info in self.model.feature_info]
        else:
            # Default для Swin-V2-L
            all_channels = [192, 384, 768, 1536]
            self.num_features = [all_channels[i] for i in out_indices]
        
        logger.info(f"[Swin-V2] Feature channels: {self.num_features}")
        
        # Подсчёт параметров
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"[Swin-V2] Parameters: {num_params:,}")
    
    def _pad_to_window_size(self, x, window_size=12):
        """Pad input to be divisible by window_size * patch_size"""
        _, _, H, W = x.shape
        patch_size = 4
        divisor = window_size * patch_size
        
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        return x, (H, W)
    
    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor с изображениями и масками
        
        Returns:
            dict[str, NestedTensor]: Multi-scale features с масками
        """
        x = tensor_list.tensors
        m = tensor_list.mask
        
        # Pad to window size
        x_padded, original_size = self._pad_to_window_size(x)
        
        # Получаем features
        features = self.model(x_padded)
        
        out = {}
        for idx, feat in enumerate(features):
            # Интерполяция маски под размер features
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out[str(idx)] = NestedTensor(feat, mask)
        
        return out


def build_swinv2_backbone(model_name, pretrained=True, out_indices=(1, 2, 3), 
                          use_checkpoint=False, **kwargs):
    """
    Создает Swin-V2 backbone
    """
    logger.info(f"[Build Swin-V2] model={model_name}, pretrained={pretrained}")
    
    return SwinV2Backbone(
        model_name=model_name,
        pretrained=pretrained,
        out_indices=out_indices,
        use_checkpoint=use_checkpoint,
    )
