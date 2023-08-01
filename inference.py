import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision import utils

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    # objective = 'pred_noise',
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
utils.save_image(sampled_images, '/mnt/workspace/workgroup/gavin/denoising-diffusion-pytorch/results/sample.png', nrow = 4)
print(sampled_images.shape) # (4, 3, 128, 128)