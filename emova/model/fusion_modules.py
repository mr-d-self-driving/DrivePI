import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism to fuse text and image features
    Handles features with different sequence lengths
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Image feature projection
        self.img_proj = nn.Linear(hidden_size, hidden_size)
        
        # Text feature projection
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, img_features, text_features):
        """
        Args:
            img_features: [B, N_img, C]
            text_features: [B, N_text, C]
        Returns:
            fused_features: [B, N_img, C]
        """
        batch_size, n_img, _ = img_features.shape
        
        # Project features
        img_proj = self.img_proj(img_features)
        text_proj = self.text_proj(text_features)
        
        # Cross-attention: image features as query, text features as key and value
        attn_output, _ = self.cross_attn(
            query=img_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Residual connection and normalization
        img_features = self.norm1(img_features + attn_output)
        
        # FFN
        ffn_output = self.ffn(img_features)
        
        # Residual connection and normalization
        fused_features = self.norm2(img_features + ffn_output)
        
        return fused_features


class GatedFusion(nn.Module):
    """
    Gated mechanism to fuse text and image features
    Handles features with different sequence lengths
    """
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        self.hidden_size = hidden_size
        
        # Image feature projection
        self.img_proj = nn.Linear(hidden_size, hidden_size)
        
        # Text feature projection
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        
        # Text feature attention
        self.text_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Gate mechanism
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, img_features, text_features):
        """
        Args:
            img_features: [B, N_img, C]
            text_features: [B, N_text, C]
        Returns:
            fused_features: [B, N_img, C]
        """
        batch_size, n_img, _ = img_features.shape
        _, n_text, _ = text_features.shape
        
        # Project features
        img_proj = self.img_proj(img_features)  # [B, N_img, C]
        
        # Aggregate text features using attention
        text_attn_weights = self.text_attention(text_features)  # [B, N_text, 1]
        text_attn_weights = F.softmax(text_attn_weights, dim=1)  # [B, N_text, 1]
        text_context = torch.bmm(text_attn_weights.transpose(1, 2), text_features)  # [B, 1, C]
        
        # Project aggregated text features
        text_proj = self.text_proj(text_context)  # [B, 1, C]
        
        # Expand text features to match image features shape
        text_proj = text_proj.expand(-1, n_img, -1)  # [B, N_img, C]
        
        # Calculate gate values
        gate_input = torch.cat([img_proj, text_proj], dim=-1)  # [B, N_img, 2*C]
        gate_value = torch.sigmoid(self.gate(gate_input))  # [B, N_img, C]
        
        # Gated fusion
        fused_features = gate_value * img_proj + (1 - gate_value) * text_proj
        
        # Output projection
        fused_features = self.out_proj(fused_features)
        
        return fused_features


class FiLMFusion(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) mechanism to fuse text and image features
    Improved version with scaling and shifting operations, handles features with different sequence lengths
    """
    def __init__(self, hidden_size):
        super(FiLMFusion, self).__init__()
        self.hidden_size = hidden_size
        
        # Text feature attention
        self.text_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Text feature processing
        self.text_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Generate scaling and shifting parameters
        self.scale_generator = nn.Linear(hidden_size, hidden_size)
        self.shift_generator = nn.Linear(hidden_size, hidden_size)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
    
    def forward(self, img_features, text_features):
        """
        Args:
            img_features: [B, N_img, C]
            text_features: [B, N_text, C]
        Returns:
            fused_features: [B, N_img, C]
        """
        batch_size, n_img, _ = img_features.shape
        
        # Aggregate text features using attention
        text_attn_weights = self.text_attention(text_features)  # [B, N_text, 1]
        text_attn_weights = F.softmax(text_attn_weights, dim=1)  # [B, N_text, 1]
        text_context = torch.bmm(text_attn_weights.transpose(1, 2), text_features)  # [B, 1, C]
        
        # Encode text features
        encoded_text = self.text_encoder(text_context)  # [B, 1, C]
        
        # Generate scaling and shifting parameters
        scales = self.scale_generator(encoded_text)  # [B, 1, C]
        shifts = self.shift_generator(encoded_text)  # [B, 1, C]
        
        # Expand parameters to match image features shape
        scales = scales.expand(-1, n_img, -1)  # [B, N_img, C]
        shifts = shifts.expand(-1, n_img, -1)  # [B, N_img, C]
        
        # Apply FiLM modulation
        modulated_features = (1.0 + scales) * img_features + shifts
        
        # Output processing
        fused_features = self.output_layer(modulated_features)
        
        return fused_features


class BilinearFusion(nn.Module):
    """
    Bilinear pooling to fuse text and image features
    Handles features with different sequence lengths
    """
    def __init__(self, hidden_size, reduction_factor=4):
        super(BilinearFusion, self).__init__()
        self.hidden_size = hidden_size
        self.reduced_dim = hidden_size // reduction_factor
        
        # Dimension reduction projections
        self.img_proj = nn.Linear(hidden_size, self.reduced_dim)
        self.text_proj = nn.Linear(hidden_size, self.reduced_dim)
        
        # Text feature attention
        self.text_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.reduced_dim * self.reduced_dim, hidden_size)
        
        # Normalization layer
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, img_features, text_features):
        """
        Args:
            img_features: [B, N_img, C]
            text_features: [B, N_text, C]
        Returns:
            fused_features: [B, N_img, C]
        """
        batch_size, seq_len, _ = img_features.shape
        
        # Project image features
        img_proj = self.img_proj(img_features)  # [B, N_img, C/r]
        
        # Aggregate text features using attention
        text_attn_weights = self.text_attention(text_features)  # [B, N_text, 1]
        text_attn_weights = F.softmax(text_attn_weights, dim=1)  # [B, N_text, 1]
        text_context = torch.bmm(text_attn_weights.transpose(1, 2), text_features)  # [B, 1, C]
        
        # Project aggregated text features
        text_proj = self.text_proj(text_context)  # [B, 1, C/r]
        
        # Expand text features to match image features shape
        text_proj = text_proj.expand(-1, seq_len, -1)  # [B, N_img, C/r]
        
        # Bilinear fusion
        bilinear = torch.bmm(img_proj.reshape(batch_size * seq_len, 1, self.reduced_dim),
                            text_proj.reshape(batch_size * seq_len, self.reduced_dim, 1))  # [B*N_img, 1, 1]
        bilinear = bilinear.reshape(batch_size * seq_len, self.reduced_dim * self.reduced_dim)
        
        # Output projection
        fused_features = self.out_proj(bilinear).reshape(batch_size, seq_len, self.hidden_size)
        
        # Residual connection and normalization
        fused_features = self.norm(img_features + fused_features)
        
        return fused_features


class DualPathFusion(nn.Module):
    """
    Dual-path attention mechanism to fuse text and image features
    Especially suitable for handling features with different sequence lengths
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(DualPathFusion, self).__init__()
        self.hidden_size = hidden_size
        
        # Image-to-text attention
        self.img2text_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Text-to-image attention
        self.text2img_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature projections
        self.img_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Output layer
        self.output_layer = nn.LayerNorm(hidden_size)
    
    def forward(self, img_features, text_features):
        """
        Args:
            img_features: [B, N_img, C]
            text_features: [B, N_text, C]
        Returns:
            fused_features: [B, N_img, C]
        """
        batch_size, n_img, _ = img_features.shape
        
        # Project features
        img_proj = self.img_proj(img_features)
        text_proj = self.text_proj(text_features)
        
        # Image-to-text attention
        img2text_output, _ = self.img2text_attn(
            query=img_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Text-to-image attention
        text2img_output, _ = self.text2img_attn(
            query=text_proj,
            key=img_proj,
            value=img_proj
        )
        
        # Aggregate text-to-image attention output
        text2img_context = text2img_output.mean(dim=1, keepdim=True).expand(-1, n_img, -1)
        
        # Fuse outputs from both attention directions
        fusion_input = torch.cat([img2text_output, text2img_context], dim=-1)
        fusion_output = self.fusion_network(fusion_input)
        
        # Residual connection and normalization
        fused_features = self.output_layer(img_features + fusion_output)
        
        return fused_features


class FiLMFusionv2(nn.Module):
    """
    Original scaling and shifting operations to fuse text and image features
    """
    def __init__(self, hidden_size):
        super(FiLMFusionv2, self).__init__()
        self.hidden_size = hidden_size
        
        # Scale and shift parameter generators
        self.scale_c = nn.Linear(hidden_size, hidden_size)
        self.shift_c = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, img_features, text_features):
        """
        Args:
            img_features: [B, N_img, C]
            text_features: [B, N_text, C]
        Returns:
            fused_features: [B, N_img, C]
        """
        batch_size = img_features.shape[0]
        
        # Average pooling on text features
        text_features = text_features.mean(dim=1, keepdim=True)  # [B, 1, C]
        
        # Generate scale and shift parameters
        scale_c = self.scale_c(text_features)  # [B, 1, C]
        shift_c = self.shift_c(text_features)  # [B, 1, C]
        
        # Apply scaling and shifting
        fused_features = (1.0 + scale_c.view(batch_size, 1, self.hidden_size)) * img_features + shift_c
        
        return fused_features
    

class NoneFusion(nn.Module):
    """
    Pass-through module that returns image features unchanged
    """
    def __init__(self, hidden_size):
        super(NoneFusion, self).__init__()
        self.hidden_size = hidden_size
    
    def forward(self, img_features, text_features):
        """
        Args:
            img_features: [B, N_img, C]
            text_features: [B, N_text, C]
        Returns:
            fused_features: [B, N_img, C]
        """
        return img_features