import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf


class ColorJitterRandomizer(nn.Module):
    """
    Randomly apply color jitter (brightness, contrast, saturation, hue) to images.
    This is crucial for real-world deployment as lighting conditions during testing
    rarely match training conditions exactly.
    
    During training: applies random color jitter
    During evaluation: returns images unchanged (identity transform)
    """
    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=1.0,
    ):
        """
        Args:
            brightness (float or tuple): How much to jitter brightness.
                If float, brightness_factor is uniformly chosen from [max(0, 1 - brightness), 1 + brightness].
                If tuple, should be (min, max) for brightness_factor.
                Default: 0.2
            contrast (float or tuple): How much to jitter contrast.
                If float, contrast_factor is uniformly chosen from [max(0, 1 - contrast), 1 + contrast].
                If tuple, should be (min, max) for contrast_factor.
                Default: 0.2
            saturation (float or tuple): How much to jitter saturation.
                If float, saturation_factor is uniformly chosen from [max(0, 1 - saturation), 1 + saturation].
                If tuple, should be (min, max) for saturation_factor.
                Default: 0.2
            hue (float or tuple): How much to jitter hue.
                If float, hue_factor is uniformly chosen from [-hue, hue].
                If tuple, should be (min, max) for hue_factor.
                Default: 0.1
            p (float): Probability of applying the transform. Default: 1.0
        """
        super().__init__()
        
        self.brightness = self._check_range(brightness, 'brightness')
        self.contrast = self._check_range(contrast, 'contrast')
        self.saturation = self._check_range(saturation, 'saturation')
        self.hue = self._check_range(hue, 'hue', center=0.0)
        self.p = p
        
    def _check_range(self, value, name, center=1.0):
        """Convert single float to tuple range if needed."""
        if isinstance(value, (int, float)):
            if name == 'hue':
                # Hue is centered at 0
                return (-value, value)
            else:
                # Brightness, contrast, saturation are centered at 1
                return (max(0, center - value), center + value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            return tuple(value)
        else:
            raise ValueError(
                f"{name} should be a single number or a tuple/list of 2 numbers. "
                f"Got {type(value)} with value {value}"
            )
    
    def forward(self, inputs):
        """
        Apply color jitter to images.
        
        Args:
            inputs (torch.Tensor): Images of shape [B, C, H, W] or [..., C, H, W]
                Expected to be in range [0, 1] (float32)
        
        Returns:
            outputs (torch.Tensor): Color jittered images with same shape as inputs
        """
        if not self.training:
            # During evaluation, return unchanged
            return inputs
        
        # Store original shape
        original_shape = inputs.shape
        is_batched = len(original_shape) == 4
        
        # Ensure inputs are in [B, C, H, W] format
        if not is_batched:
            inputs = inputs.unsqueeze(0)
        
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Apply transform with probability p (per batch)
        if torch.rand(1, device=device).item() > self.p:
            if not is_batched:
                inputs = inputs.squeeze(0)
            return inputs
        
        # Get random parameters for this batch (same for all images in batch)
        brightness_factor = torch.empty(1, device=device).uniform_(
            self.brightness[0], self.brightness[1]).item()
        contrast_factor = torch.empty(1, device=device).uniform_(
            self.contrast[0], self.contrast[1]).item()
        saturation_factor = torch.empty(1, device=device).uniform_(
            self.saturation[0], self.saturation[1]).item()
        hue_factor = torch.empty(1, device=device).uniform_(
            self.hue[0], self.hue[1]).item()
        
        # Apply color jitter to each image in the batch
        outputs = []
        for img in inputs:
            # Apply brightness
            if brightness_factor != 1.0:
                img = ttf.adjust_brightness(img, brightness_factor)
            
            # Apply contrast
            if contrast_factor != 1.0:
                img = ttf.adjust_contrast(img, contrast_factor)
            
            # Apply saturation
            if saturation_factor != 1.0:
                img = ttf.adjust_saturation(img, saturation_factor)
            
            # Apply hue
            if hue_factor != 0.0:
                img = ttf.adjust_hue(img, hue_factor)
            
            outputs.append(img)
        
        outputs = torch.stack(outputs, dim=0)
        
        # Restore original shape if needed
        if not is_batched:
            outputs = outputs.squeeze(0)
        
        return outputs
    
    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(brightness={}, contrast={}, saturation={}, hue={}, p={})".format(
            self.brightness, self.contrast, self.saturation, self.hue, self.p)
        return msg


class ColorJitterRandomizerPerSample(nn.Module):
    """
    Color jitter randomizer that applies different random parameters to each sample
    in the batch. This provides more diversity compared to applying the same transform
    to all samples.
    """
    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=1.0,
    ):
        """
        Args:
            brightness (float or tuple): Brightness jitter range. Default: 0.2
            contrast (float or tuple): Contrast jitter range. Default: 0.2
            saturation (float or tuple): Saturation jitter range. Default: 0.2
            hue (float or tuple): Hue jitter range. Default: 0.1
            p (float): Probability of applying the transform. Default: 1.0
        """
        super().__init__()
        
        self.brightness = self._check_range(brightness, 'brightness')
        self.contrast = self._check_range(contrast, 'contrast')
        self.saturation = self._check_range(saturation, 'saturation')
        self.hue = self._check_range(hue, 'hue', center=0.0)
        self.p = p
        
    def _check_range(self, value, name, center=1.0):
        """Convert single float to tuple range if needed."""
        if isinstance(value, (int, float)):
            if name == 'hue':
                return (-value, value)
            else:
                return (max(0, center - value), center + value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            return tuple(value)
        else:
            raise ValueError(
                f"{name} should be a single number or a tuple/list of 2 numbers. "
                f"Got {type(value)} with value {value}"
            )
    
    def forward(self, inputs):
        """
        Apply color jitter to images with different parameters for each sample.
        
        Args:
            inputs (torch.Tensor): Images of shape [B, C, H, W] or [..., C, H, W]
                Expected to be in range [0, 1] (float32)
        
        Returns:
            outputs (torch.Tensor): Color jittered images with same shape as inputs
        """
        if not self.training:
            # During evaluation, return unchanged
            return inputs
        
        # Store original shape
        original_shape = inputs.shape
        is_batched = len(original_shape) == 4
        
        # Ensure inputs are in [B, C, H, W] format
        if not is_batched:
            inputs = inputs.unsqueeze(0)
        
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Generate random parameters for each sample in the batch
        # Use torch.rand for GPU compatibility
        brightness_factors = torch.empty(batch_size, device=device).uniform_(
            self.brightness[0], self.brightness[1])
        contrast_factors = torch.empty(batch_size, device=device).uniform_(
            self.contrast[0], self.contrast[1])
        saturation_factors = torch.empty(batch_size, device=device).uniform_(
            self.saturation[0], self.saturation[1])
        hue_factors = torch.empty(batch_size, device=device).uniform_(
            self.hue[0], self.hue[1])
        
        # Random mask for probability p
        apply_mask = torch.rand(batch_size, device=device) < self.p
        
        # Apply color jitter to each image
        outputs = []
        for i, img in enumerate(inputs):
            if not apply_mask[i]:
                outputs.append(img)
                continue
            
            # Apply brightness
            if brightness_factors[i] != 1.0:
                img = ttf.adjust_brightness(img, brightness_factors[i].item())
            
            # Apply contrast
            if contrast_factors[i] != 1.0:
                img = ttf.adjust_contrast(img, contrast_factors[i].item())
            
            # Apply saturation
            if saturation_factors[i] != 1.0:
                img = ttf.adjust_saturation(img, saturation_factors[i].item())
            
            # Apply hue
            if hue_factors[i] != 0.0:
                img = ttf.adjust_hue(img, hue_factors[i].item())
            
            outputs.append(img)
        
        outputs = torch.stack(outputs, dim=0)
        
        # Restore original shape if needed
        if not is_batched:
            outputs = outputs.squeeze(0)
        
        return outputs
    
    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(brightness={}, contrast={}, saturation={}, hue={}, p={})".format(
            self.brightness, self.contrast, self.saturation, self.hue, self.p)
        return msg

