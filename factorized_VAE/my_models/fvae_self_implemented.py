import torch
import torch.nn as nn
from imagefolder_models.vae import AutoencoderKL, Decoder
from factorized_VAE.my_models.adablock import AdaBlock
from factorized_VAE.my_models.vector_quantizer import VectorQuantizer
from tokenizer.tokenizer_image.lpips import LPIPS

class FVAE(nn.Module):
    
    '''
    pretrained vae encoder (get continous latent z from img)
    pretrained vae encoder + AdaBlock = vq-vae encoder (get z' from img)
    vector quantizer (get z_q from z')
    vae decoder (decode z to x_recon)
    vq-vae decoder (decode z_q to x_recon)
    
    '''
    def __init__(self, vae_ckpt_path=None):
        super().__init__()
        self.embed_dim = 16
        self.codebook_size = 8192  # Number of discrete codes in the VQ model
        self.vae = AutoencoderKL(embed_dim=self.embed_dim, ch_mult=(1, 1, 2, 2, 4), 
                                 ckpt_path=vae_ckpt_path).cuda().eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.adaptive_block = AdaBlock(in_channels=self.embed_dim).cuda()  # Ensure PostQuant is on GPU
        self.quantizer = VectorQuantizer(n_e=self.codebook_size, e_dim=self.embed_dim, beta=0.25).cuda()
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.embed_dim, 1).cuda()
        self.decoder = Decoder(ch_mult=(1, 1, 2, 2, 4), z_channels=self.embed_dim).cuda()
        self.ploss = LPIPS().cuda().eval()
        
    def decode_quantized_latent(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
        
    def forward(self, x):
        '''
        encode with VAE, quantize the latent, decode the quantized latent
        '''
        posterior = self.vae.encode(x) #x.shape = (batch_size, 3, 256, 256)
        z = posterior.mean #z.shape = (batch_size, embed_dim=16, 16, 16)
        z = self.adaptive_block(z)#z.shape = (batch_size, embed_dim=16, 16, 16)
        codebook_loss, z_q, perplexity = self.quantizer(z)
        x_recon = self.decode_quantized_latent(z_q)
        recon_loss = nn.MSELoss()(x_recon, x)
        ploss = self.ploss(x_recon, x).mean()
        return codebook_loss, recon_loss, ploss, z_q, x_recon, perplexity        

# fvae = FVAE("/home/renderex/causal_groups/jinyuan.hu/mar/pretrained_models/vae/kl16.ckpt")

# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# dataset = ImageFolder("/home/renderex/causal_groups/jinyuan.hu/CIFAR10", transform=transform)
# print(f"Dataset size: {len(dataset)}")
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# for images, _ in tqdm(dataloader):
#     images = images.cuda()
#     _, z_q, _ = fvae(images)
#     print(f"Encoded shape: {z_q.shape}")
    
        