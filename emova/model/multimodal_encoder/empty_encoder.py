import torch
from transformers import BatchFeature

from emova.model.multimodal_encoder.builder import VISION_TOWER, BaseModalityEncoder
from einops import rearrange

class DummyImageProcessor:
    """
    假的图像处理器，用于满足训练代码的要求。
    对于 BEV 特征，我们不需要真正的图像处理。
    """
    
    def __init__(self):
        self.size = {"height": 180, "width": 180}  # 默认 BEV 特征尺寸
        self.crop_size = {"height": 180, "width": 180}
        self.do_resize = True
        self.size_divisor = 1
        self.do_pad = False
        self.do_normalize = False
        self.image_mean = [0.0, 0.0, 0.0]
        self.image_std = [1.0, 1.0, 1.0]
        
    def __call__(self, images, return_tensors=None, **kwargs):
        """
        假的处理函数，直接返回输入。
        对于 BEV 特征，我们不需要图像处理。
        """
        return BatchFeature(data=dict(pixel_values=images), tensor_type=return_tensors)

    def preprocess(self, images, **kwargs):
        """
        假的预处理函数，直接返回输入。
        """
        return images


@VISION_TOWER.register_module()
class EmptyVisionTower(BaseModalityEncoder):
    """
    Empty vision encoder that directly returns features from dataset.
    This encoder assumes that the input features are already pre-computed
    and just passes them through without any processing.
    Designed to work with ConcatChannelMMProjector for BEV features.
    """
    
    def __init__(self, hidden_size=384, spatial_size=(180, 180), **kwargs):
        super().__init__()
        self._hidden_size = hidden_size
        self._spatial_size = spatial_size
        
        # 添加假的 image_processor 来满足训练代码要求
        self.image_processor = DummyImageProcessor()
        self.is_loaded = True

    def forward(self, features):
        """
        Forward pass that directly returns the input features.
        
        Args:
            features: Pre-computed features from dataset
                     Shape: (batch_size, num_patches, hidden_size) or list of such tensors
                     Expected: (batch_size, H*W, hidden_size) where H*W = spatial_size[0] * spatial_size[1]
        
        Returns:
            The same features without any processing
        """
        return rearrange(features, 'b c h w -> b (h w) c')
    
    @property
    def dummy_feature(self):
        """Return dummy feature for initialization purposes."""
        expected_patches = self._spatial_size[0] * self._spatial_size[1]
        return torch.zeros(1, expected_patches, self._hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        """Return the dtype of the model."""
        # Use float32 as default, can be overridden
        return torch.float32
    
    @property
    def device(self):
        """Return the device of the model."""
        # Use CPU as default, will be moved to appropriate device by the model
        return torch.device('cpu')
    
    @property
    def config(self):
        """Return a dummy config object."""
        class DummyConfig:
            def __init__(self, hidden_size, spatial_size):
                self.hidden_size = hidden_size
                self.image_size = max(spatial_size)  # Use max dimension as image size
                self.patch_size = 1  # No patch size for BEV features
                
        return DummyConfig(self._hidden_size, self._spatial_size)
    
    @property
    def hidden_size(self):
        """Return the hidden size of the features."""
        return self._hidden_size
    
    @property
    def num_patches_per_side(self):
        """Return the number of patches per side."""
        return max(self._spatial_size)
    
    @property
    def num_patches(self):
        """Return the total number of patches."""
        return self._spatial_size[0] * self._spatial_size[1] 