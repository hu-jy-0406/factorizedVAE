
import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder_Simple
from quantizer import VectorQuantizer
from decoder import Decoder_Simple


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder_Simple(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder_Simple(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        ## x.shape = (batch_size, channels, height, width)
        ## x.shape = (32, 3, 32, 32)
        z_e = self.encoder(x)
        ## z_e.shape = (batch_size, h_dim, height, width)
        ## z_e.shape = (32, 128, 8, 8)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        ## z_q.shape = (batch_size, embedding_dim, height, width)
        ## z_q.shape = (32, 64, 8, 8)
        x_hat = self.decoder(z_q)
        ## x_hat.shape = (batch_size, channels, height, width)
        ## x_hat.shape = (32, 3, 32, 32)
        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
