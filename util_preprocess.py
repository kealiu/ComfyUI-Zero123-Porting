from PIL import Image
import numpy as np
import torch

def image_to_mask(image, channel='red'):
    channels = ["red", "green", "blue", "alpha"]
    mask = image[:, :, channels.index(channel)]
    return (mask,)

# mask handle
def tensor2PIL(image: torch.Tensor) -> list[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2PIL(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )
    ]

def mask2bbox(mask: torch.Tensor):
    _mask = tensor2PIL(mask)[0]
    alpha_channel = np.array(_mask)

    non_zero_indices = np.nonzero(alpha_channel)

    try:
        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    except:
        return (-1, -1, -1, -1, -1)
 
    h = max_y - min_y
    w = max_x - min_x
    corpx = min_x
    corpy = min_y
    sidelen = h
    if (h > w):
        corpx = corpx - (h - w)//2
    elif (h < w ):
        sidelen = w
        corpy = corpy - (w - h)//2
    return (corpx, corpy, h, w, sidelen)

#
# code borrow from comfyui
#
def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
    return tensor

def composite_image_with_mask(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination

def composite_new_image(destination, source, x, y, resize_source, mask = None):
    destination = destination.clone().movedim(-1, 1)
    output = composite_image_with_mask(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
    return (output,)

def generate_pure_image(width, height, batch_size=1, color=0):
    r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
    g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
    b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
    a = torch.full([batch_size, height, width, 1], 1)
    return (torch.cat((r, g, b, a), dim=-1), )
