import torch
import torchvision.transforms as transforms
from PIL import Image

def linear_interpolation(z1, z2, num_steps=10):
    """
    Creates a linear interpolation between two latent vectors z1 and z2.
    Args:        
        z1: First latent vector (torch.Tensor).
        z2: Second latent vector (torch.Tensor).
        num_steps: Number of interpolation steps (int).
    Returns:        
    A tensor of shape (num_steps, latent_dim) containing the interpolated vectors.
    """
    alphas = torch.linspace(0, 1, steps=num_steps)
    
    z_interp = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
    return z_interp

def prepare_masked_image(image_path, device, mask_size=24, img_size=64):
    """
    Loads an image, resizes it, creates a central mask, and applies it.
    
    Args:
        image_path (str): Path to the real image.
        device (torch.device): CPU or GPU.
        mask_size (int): Size of the square mask (e.g., 24x24).
        img_size (int): Target size of the image (e.g., 64x64).
        
    Returns:
        tuple: (real_img, masked_img, mask) - all as tensors ready for the model.
    """
    # Load and transform the image
    try:
        raw_img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find image at: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # [1, 3, 64, 64]
    real_img = transform(raw_img).unsqueeze(0).to(device)

    # Create the central mask
    # Calculate starting and ending pixels for the mask
    center = img_size // 2
    half_mask = mask_size // 2
    start = center - half_mask
    end = center + half_mask

    mask = torch.zeros_like(real_img)
    mask[:, :, start:end, start:end] = 1.0  # 1.0 means "masked/missing"

    # Apply the mask (zero out the center)
    masked_img = real_img * (1.0 - mask)

    return real_img, masked_img, mask

def calculate_lpips(loss_fn, real_patch, recon_patch):
    """
    Calculates LPIPS between two patches, scaling them to [-1, 1].
    """
    # LPIPS expects inputs in [-1, 1]
    real_patch_scaled = real_patch * 2.0 - 1.0
    recon_patch_scaled = recon_patch * 2.0 - 1.0
    
    with torch.no_grad():
        val = loss_fn(real_patch_scaled, recon_patch_scaled).item()
    return val