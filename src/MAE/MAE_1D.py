"""
1D Masked autoenecoder Trainer.
This is a modified version from the MAE method adopted from : https://github.com/Romain3Ch216/tlse-experiments/
"""

import sys
import os

# Set up the parent directory in the system path
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import utility modules from local packages
from MAE.utils_mae import *
from utils_all import *

import torch
import torch.nn as nn
import numpy as np


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone.

    This class implements both the encoder and decoder parts for a masked autoencoder.
    It supports positional embeddings, masking strategies, and weight initialization.
    """
    def __init__(self, n_bands=1720, seq_size=20, in_chans=1,
                 embed_dim=128, depth=6, num_heads=4,
                 decoder_embed_dim=128, decoder_depth=6, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, cls_token=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE Encoder Specifics
        # Create a sequential embedder to split spectra into patches
        self.seq_embed = SeqEmbed(n_bands, seq_size, in_chans, embed_dim)
        num_sequences = self.seq_embed.num_sequences

        # Initialize a classification token if required
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.is_cls_token = True
        else:
            self.is_cls_token = False

        # Fixed sin-cos positional embedding (not learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_sequences + np.sum(self.is_cls_token), embed_dim), 
            requires_grad=False
        )

        # Create Transformer blocks for the encoder
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # Calculate and print total number of parameters for the encoder
        n_params = 0
        for param in self.parameters():
            n_params += param.shape.numel()
        print(f'Encoder has {n_params} parameters.')
        # --------------------------------------------------------------------------
        # MAE Decoder Specifics
        # Linear layer to project encoder outputs to decoder embedding dimensions
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask token for the masked patches in the decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sin-cos positional embedding for the decoder (non-learnable)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_sequences + np.sum(self.is_cls_token), decoder_embed_dim),
            requires_grad=False
        )

        # Create Transformer blocks for the decoder
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Final projection layer to predict the original sequence patches
        self.decoder_pred = nn.Linear(decoder_embed_dim, seq_size * in_chans, bias=True)
        # --------------------------------------------------------------------------

        # Initialize weights for positional embeddings, tokens, and network layers
        self.initialize_weights()

        # Calculate and print total number of parameters for the decoder
        n_params = -n_params  # subtract encoder parameters to isolate decoder ones
        for param in self.parameters():
            n_params += param.shape.numel()
        print(f'Decoder has {n_params} parameters.')

    def initialize_weights(self):
        """Initialize weights for the positional embeddings, projection layers, and tokens."""
        # Initialize and freeze positional embeddings using fixed sin-cos values
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.seq_embed.num_sequences, cls_token=self.is_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.seq_embed.num_sequences, cls_token=self.is_cls_token)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize the projection weights in the sequential embedder using Xavier uniform initialization
        w = self.seq_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls_token (if used) and mask_token with a normal distribution
        if self.is_cls_token:
            torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Apply weight initialization to all Linear and LayerNorm layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for Linear and LayerNorm modules."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def sequencify(self, spectra):
        """
        Reshape a flat spectra tensor into sequential patches.

        Args:
            spectra: Tensor of shape (batch_size, n_bands).

        Returns:
            x: Tensor reshaped to (batch_size, num_sequences, seq_size).
        """
        seq_size = self.seq_embed.seq_size
        # Ensure the number of bands can be divided evenly by the sequence size
        assert spectra.shape[1] % seq_size == 0

        num_sequences = spectra.shape[1] // seq_size
        x = spectra.reshape(shape=(spectra.shape[0], num_sequences, seq_size))
        return x

    def unsequencify(self, x):
        """
        Reshape sequential patches back into the original flat spectra.

        Args:
            x: Tensor of shape (batch_size, num_sequences, seq_size).

        Returns:
            spectra: Tensor reshaped to (batch_size, n_bands).
        """
        spectra = x.reshape(shape=(x.shape[0], -1))
        return spectra

    def half_masking(self, x, last_token=None):
        """
        Perform per-sample half masking.

        The first half of the patches is masked (True) and the second half is unmasked (False).

        Args:
            x: Tensor of shape [N, L, D] where:
                N = batch size,
                L = sequence length (number of patches),
                D = embedding dimension.
            last_token: Optional integer to override half-length calculation.

        Returns:
            x_masked: Tensor with only the unmasked patches.
            mask: Binary mask tensor of shape [N, L] (0 = keep, 1 = mask).
            ids_restore: Tensor to restore the original sequence order.
        """
        N, L, D = x.shape
        
        # Determine the number of patches to keep
        if last_token is None:
            first_half_length = L // 2
        else:
            first_half_length = last_token

        # Generate indices for shuffling and restoration
        ids = torch.arange(L, device=x.device).unsqueeze(0).repeat(N, 1)
        ids_keep = ids[:, :first_half_length]  # Indices for the unmasked (kept) patches
        ids_mask = ids[:, first_half_length:]  # Indices for the masked patches

        # Form the restore order by concatenating kept and masked indices
        ids_restore = torch.cat([ids_keep, ids_mask], dim=1)

        # Extract the unmasked patches
        x_masked = x[:, :first_half_length, :]

        # Create the binary mask: 0 for unmasked, 1 for masked
        mask = torch.ones((N, L), device=x.device)
        mask[:, :first_half_length] = 0  # Unmask the first half
        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking via shuffling.

        A random noise is generated for each patch to determine which patches to keep.

        Args:
            x: Tensor of shape [N, L, D] (sequence of patches).
            mask_ratio: Fraction of patches to mask.

        Returns:
            x_masked: Masked tensor with only the unmasked patches.
            mask: Binary mask tensor of shape [N, L] (0 = keep, 1 = remove).
            ids_restore: Indices to restore the original ordering.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # Generate random noise for each patch
        noise = torch.rand(N, L, device=x.device)

        # Shuffle indices based on the noise values
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Select the first subset of indices to keep
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Create a binary mask: 0 for kept patches, 1 for removed patches
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle the binary mask to match original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through the encoder.

        Args:
            x: Input spectra tensor.
            mask_ratio: Ratio of patches to mask.

        Returns:
            x: Encoder output.
            mask: Binary mask tensor.
            ids_restore: Indices to restore the original sequence order.
        """
        # Embed the patches from the input spectra
        x = self.seq_embed(x)
        # Add positional embeddings (skip cls token if present)
        x = x + self.pos_embed[:, np.sum(self.is_cls_token):, :]

        # Apply random masking to the embedded patches
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # If a classification token is used, prepend it to the sequence
        if self.is_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Pass through Transformer encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through the decoder.

        Args:
            x: Latent representation from the encoder.
            ids_restore: Indices for restoring the original sequence order.

        Returns:
            x: Reconstructed patches.
        """
        # Project latent features to the decoder embedding space
        x = self.decoder_embed(x)

        # Generate mask tokens for the positions that were masked out
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        # Concatenate unmasked tokens with mask tokens (excluding cls token)
        x_ = torch.cat([x[:, np.sum(self.is_cls_token):, :], mask_tokens], dim=1)
        # Unshuffle tokens to original ordering
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # Reattach the cls token if used
        if self.is_cls_token:
            x = torch.cat([x[:, :1, :], x_], dim=1)
        else:
            x = x_

        # Add decoder positional embeddings
        x = x + self.decoder_pos_embed

        # Pass through Transformer decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Final prediction projection to reconstruct the patches
        x = self.decoder_pred(x)

        # Remove the cls token from the output if present
        if self.is_cls_token:
            x = x[:, 1:, :]

        return x

    def forward_loss(self, spectra, pred, mask, w_loss= 0.):
        """
        Compute the reconstruction loss between predicted and target patches.

        Args:
            spectra: Original input spectra tensor of shape [batch_size, n_bands].
            pred: Decoder predictions with shape [batch_size, num_sequences, seq_size].
            mask: Binary mask indicating which patches were removed.

        Returns:
            loss: Weighted reconstruction loss computed on the masked patches.
        """
        # Convert the input spectra to its sequential (patch) representation
        target = self.sequencify(spectra)
        # Compute mean squared error loss per patch
        loss_mse = torch.abs(pred - target) ** 2
        loss_mse = loss_mse.mean(dim=-1)  # [N, L]

        # Compute cosine similarity loss between prediction and target
        cosine_loss = CosineSimilarityLoss()
        # cosine_loss_ = 1 - cosine_loss(pred, target)
        cosine_loss_ = cosine_loss(pred, target)

        # Weighted sum of cosine and MSE losses (currently using only MSE loss)
        loss = w_loss * cosine_loss_ + loss_mse

        # Compute loss only on the masked patches if any are masked
        if mask.sum() > 0:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def latent(self, spectra):
        """
        Extract the latent representation from the encoder.

        Args:
            spectra: Input spectra tensor.

        Returns:
            latent: Latent feature vector.
        """
        latent, _, _ = self.forward_encoder(spectra, mask_ratio=0)
        if self.is_cls_token:
            latent = latent[:, 0, :]
        else:
            latent = torch.mean(latent[:, 1:, :], dim=1)
        return latent

    def forward(self, spectra, mask_ratio=0.75, w_loss= 0.):
        """
        Perform the forward pass through the autoencoder.

        Args:
            spectra: Input spectra tensor.
            mask_ratio: Ratio of patches to mask (default is 0.75).

        Returns:
            loss: Reconstruction loss.
            pred: Decoder predictions.
            mask: Binary mask used during encoding.
            latent: Latent representation from the encoder.
        """
        # Run the encoder and obtain latent representation and mask details
        latent, mask, ids_restore = self.forward_encoder(spectra, mask_ratio)
        # Reconstruct the input from the latent representation using the decoder
        pred = self.forward_decoder(latent, ids_restore)
        # Compute the reconstruction loss
        loss = self.forward_loss(spectra, pred, mask, w_loss)
        return loss, pred, mask, latent
