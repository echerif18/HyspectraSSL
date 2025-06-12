import torch
import torch.nn as nn
from typing import Callable, Optional, Union
from enum import Enum
from itertools import repeat
import collections.abc
import numpy as np


# Adapted from: https://github.com/Romain3Ch216/tlse-experiments/tree/main
#############################################
# Enum and Utility Functions for Format Types
#############################################

class Format(str, Enum):
    """
    Enumeration of different tensor format conventions.
    """
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


def _ntuple(n):
    """
    Returns a function that converts its input into an n-tuple.
    
    If the input is already an iterable (and not a string), it is converted to a tuple.
    Otherwise, the value is repeated n times in a tuple.
    
    Args:
        n (int): The number of repetitions.
        
    Returns:
        Function: A function that converts its input to an n-tuple.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

# Utility functions to convert values to tuples of length 2 or 1.
to_2tuple = _ntuple(2)
to_1tuple = _ntuple(1)

# Define a type alias for format strings or Format enum values.
FormatT = Union[str, Format]


def get_spatial_dim(fmt: FormatT):
    """
    Get the spatial dimension indices based on the tensor format.
    
    Args:
        fmt (FormatT): Tensor format.
        
    Returns:
        tuple: A tuple containing the spatial dimension indices.
    """
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim


def get_channel_dim(fmt: FormatT):
    """
    Get the channel dimension index based on the tensor format.
    
    Args:
        fmt (FormatT): Tensor format.
        
    Returns:
        int: The index of the channel dimension.
    """
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


def nchw_to(x: torch.Tensor, fmt: Format):
    """
    Convert a tensor from NCHW format to another format.
    
    Args:
        x (torch.Tensor): Input tensor in NCHW format.
        fmt (Format): Target format.
        
    Returns:
        torch.Tensor: Tensor converted to the target format.
    """
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        # Flatten spatial dimensions and transpose: (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        x = x.flatten(2)
    return x


def nhwc_to(x: torch.Tensor, fmt: Format):
    """
    Convert a tensor from NHWC format to another format.
    
    Args:
        x (torch.Tensor): Input tensor in NHWC format.
        fmt (Format): Target format.
        
    Returns:
        torch.Tensor: Tensor converted to the target format.
    """
    if fmt == Format.NCHW:
        x = x.permute(0, 3, 1, 2)
    elif fmt == Format.NLC:
        # Flatten first two spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        x = x.flatten(1, 2)
    elif fmt == Format.NCL:
        # Flatten first two spatial dimensions then transpose: (B, H, W, C) -> (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(1, 2).transpose(1, 2)
    return x

#############################################
# _assert: Compatibility for PyTorch's _assert
#############################################

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

#############################################
# Sequence Embedding Module
#############################################

class SeqEmbed(nn.Module):
    """
    Converts a spectral vector (batch_size x n_bands) into a sequence embedding
    (batch_size x n_sequences x dim_embedding) by splitting into patches.
    
    Args:
        n_bands (int): Number of spectral bands.
        seq_size (int): Size of each sequence (patch).
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of the output embedding.
        norm_layer (Optional[Callable]): Normalization layer.
        flatten (bool): Whether to flatten the output.
        output_fmt (Optional[str]): Desired output format (unused in this code).
        bias (bool): Whether to use bias in convolution.
        strict_sp_size (bool): Enforce strict input spectral size.
    """
    def __init__(
            self,
            n_bands: int = 1720,
            seq_size: int = 20,
            in_chans: int = 1,
            embed_dim: int = 128,
            norm_layer: Optional[Callable] = None,
            flatten: bool = False,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_sp_size: bool = True,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.n_bands = n_bands
        # Compute the number of sequences (patches) from the input bands.
        self.num_sequences = self.n_bands // self.seq_size

        self.flatten = flatten
        self.strict_sp_size = strict_sp_size

        # Convolution to project sequence patches into embedding space.
        self.proj = nn.Conv1d(seq_size, embed_dim, kernel_size=5, stride=1, padding=2, bias=bias)
        # Apply normalization if a norm_layer is provided.
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        # Freeze the parameters of the convolution layer.
        for param in self.proj.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass for sequence embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, n_bands).
            
        Returns:
            torch.Tensor: Embedded sequence tensor.
        """
        B, n_bands = x.shape
        
        # Validate spectral dimension.
        if self.strict_sp_size:
            _assert(n_bands == self.n_bands, f"Spectral dimension ({n_bands}) doesn't match model ({self.n_bands}).")
        else:
            _assert(
                n_bands % self.seq_size == 0,
                f"Spectral dimension ({n_bands}) should be divisible by sequence size ({self.seq_size})."
            )
        # Reshape input to patches: (B, num_sequences, seq_size) and then transpose for Conv1d.
        x = x.view(B, self.num_sequences, self.seq_size).transpose(1, 2)
        
        # Apply the projection and then transpose back.
        x = self.proj(x).transpose(1, 2)
        x = self.norm(x)
        return x

#############################################
# Attention Module
#############################################

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        dim (int): Input and output dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add bias to QKV projections.
        qk_norm (bool): If True, apply normalization to Q and K.
        qk_scale (Optional[float]): Scale factor for QK (unused here).
        attn_drop (float): Dropout probability for attention weights.
        proj_drop (float): Dropout probability for the output projection.
        norm_layer (Callable): Normalization layer.
    """
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # Ensure the dimension is divisible by the number of heads.
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Scaling factor for query vectors.
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False  # Flag to use fused attention (if available).

        # Linear layer to compute queries, keys, and values.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Optional normalization for queries and keys.
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        # Output projection layers.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Forward pass for the attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            
        Returns:
            torch.Tensor: Output tensor after attention.
        """
        B, N, C = x.shape
        # Compute Q, K, V and reshape for multi-head attention.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Apply optional normalization.
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            # Use PyTorch's fused attention if available.
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            # Scale queries and compute attention scores.
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            # Compute weighted sum of values.
            x = attn @ v

        # Reshape output and apply final projection.
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#############################################
# MLP Module
#############################################

class Mlp(nn.Module):
    """
    Multi-layer Perceptron (MLP) module as used in Vision Transformers and MLP-Mixer.
    
    Args:
        in_features (int): Number of input features.
        hidden_features (Optional[int]): Number of hidden features.
        out_features (Optional[int]): Number of output features.
        act_layer (Callable): Activation function.
        norm_layer (Optional[Callable]): Normalization layer.
        bias (bool): If True, include bias terms.
        drop (float): Dropout probability.
        use_conv (bool): If True, use a convolutional layer instead of linear.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Convert bias and dropout parameters to 2-tuples.
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        # Choose between Conv2d or Linear layer based on use_conv flag.
        linear_layer = nn.Conv2d if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """
        Forward pass for the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after MLP.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

#############################################
# Drop Path (Stochastic Depth) Utilities
#############################################

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Args:
        x (torch.Tensor): Input tensor.
        drop_prob (float): Probability of dropping paths.
        training (bool): If True, perform dropout.
        scale_by_keep (bool): If True, scale output by 1/keep_prob.
        
    Returns:
        torch.Tensor: Tensor after applying drop path.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Generate random tensor for dropping paths.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Module that implements Drop Path (Stochastic Depth).
    
    Args:
        drop_prob (float): Probability of dropping a path.
        scale_by_keep (bool): If True, scale the output by 1/keep_prob.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


#############################################
# LayerScale Module
#############################################

class LayerScale(nn.Module):
    """
    Layer scaling module to scale features by a learnable parameter.
    
    Args:
        dim (int): Dimension of the scaling factor.
        init_values (float): Initial value for the scaling parameter.
        inplace (bool): If True, perform in-place multiplication.
    """
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # In-place multiplication if specified, otherwise regular multiplication.
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

#############################################
# Transformer Block Module
#############################################

class Block(nn.Module):
    """
    Transformer block that combines attention, MLP, normalization, and residual connections.
    
    Args:
        dim (int): Dimension of the input and output.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio for hidden dimension in MLP.
        qkv_bias (bool): If True, add bias to QKV projections.
        qk_norm (bool): If True, normalize queries and keys.
        qk_scale (Optional[float]): Scaling factor for QK (unused).
        proj_drop (float): Dropout probability for projections.
        attn_drop (float): Dropout probability for attention.
        init_values (Optional[float]): Initial value for layer scaling.
        drop_path (float): Drop path probability.
        act_layer (Callable): Activation function.
        norm_layer (Callable): Normalization layer.
        mlp_layer (Callable): MLP module.
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            qk_scale=None,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        # First normalization and attention sub-layer.
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        # Optional layer scaling and drop path for the attention branch.
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Second normalization and MLP sub-layer.
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        # Optional layer scaling and drop path for the MLP branch.
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after processing through the block.
        """
        # Attention branch with residual connection.
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # MLP branch with residual connection.
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

#############################################
# 1D Sin-Cos Positional Embedding
#############################################

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    """
    Generate 1D sin-cos positional embeddings.
    
    Args:
        embed_dim (int): Output dimension for each position (must be even).
        grid_size (int): Number of positions.
        cls_token (bool): If True, prepend a zero vector for a classification token.
        
    Returns:
        np.ndarray: Positional embeddings of shape (grid_size, embed_dim) or
                    (grid_size+1, embed_dim) if cls_token is True.
    """
    assert embed_dim % 2 == 0, "Embed dimension must be even."
    grid = np.arange(grid_size, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # Compute the frequency terms.

    # Outer product to generate angle rates.
    out = np.einsum('m,d->md', grid, omega)

    # Compute sin and cos embeddings.
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    # Concatenate sin and cos parts.
    emb = np.concatenate([emb_sin, emb_cos], axis=1)

    # Optionally prepend a zero vector for the class token.
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb
