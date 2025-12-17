"""
Image Utilities
Helper functions cho image processing
"""

import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ..config.model_config import ModelConfig


class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def get_image_transform():
        """
        Get image preprocessing transform
        Returns:
            torchvision.transforms.Compose
        """
        return transforms.Compose([
            transforms.Resize(ModelConfig.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ModelConfig.NORMALIZE_MEAN,
                std=ModelConfig.NORMALIZE_STD
            )
        ])
    
    @staticmethod
    def decode_image(image_binary):
        """
        Decode image from binary
        Args:
            image_binary: Binary image data
        Returns:
            PIL Image
        """
        return Image.open(io.BytesIO(image_binary)).convert('RGB')
    
    @staticmethod
    def preprocess_image(image_binary):
        """
        Preprocess image for model
        Args:
            image_binary: Binary image data
        Returns:
            Tensor
        """
        img = ImageUtils.decode_image(image_binary)
        transform = ImageUtils.get_image_transform()
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        return img_tensor
    
    @staticmethod
    def is_valid_image(image_binary):
        """
        Check if image is valid
        Args:
            image_binary: Binary image data
        Returns:
            bool
        """
        try:
            img = ImageUtils.decode_image(image_binary)
            # Check dimensions
            if img.size[0] > 0 and img.size[1] > 0:
                return True
        except Exception:
            pass
        return False
