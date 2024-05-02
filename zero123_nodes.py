import torch
import folder_paths
import numpy as np
from PIL import Image
from zero123 import init_model, predict_cam

#_GPU_INDEX = 0

g_model = None
g_ckpt = None
g_device = None
g_hf = None
def load_model(checkpoint, hf=True):
    global g_model
    global g_ckpt
    global g_device
    global g_hf
    if (g_ckpt == checkpoint) and (g_hf == hf) and g_model:
        return (g_model, g_device)
    # not init or ckpt changed
    if g_model:
        del g_model # may need reload model
        g_model = None
        torch.cuda.empty_cache()

    if not g_device:
        g_device = 'cpu'
        if torch.cuda.is_available():
            gpu = torch.cuda.current_device()
            if gpu >= 0:
                g_device = f'cuda:{gpu}'
    g_model = init_model(g_device, checkpoint, half_precision=hf)
    g_ckpt = checkpoint
    g_hf = hf
    return (g_model, g_device)

class Zero123:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                    "image": ("IMAGE",), 
                    "polar_angle": ("INT", { "default": 0, "min": -180, "max": 180, "step": 1, "display": "number"}),
                    "azimuth_angle": ("INT", { "default": 0, "min": -180, "max": 180, "step": 1, "display": "number"}),
                    "scale": ("FLOAT", { "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                    "steps": ("INT", { "default": 75, "min": 1, "step": 1, "display": "number"}),
                    "batch_size": ("INT", { "default": 1, "min": 1, "step": 1, "display": "number"}),
                    "fp16": ("BOOLEAN", { "default": True }),
                    "checkpoint": (list(filter(lambda k: 'zero123' in k, folder_paths.get_filename_list("checkpoints"))), ),  
                },
            "optional": {
                "height": (["height=256"],),
                "width": (["width=256"],),
                "sampler": (["ddim"],),
                "scheduler": (["ddim-uniform"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (False, )
    FUNCTION = "moveCam"
    CATEGORY = "image"

    #
    # height, width, sample, scheduler cannot changed currently, just for show information
    #
    def moveCam(self, image, polar_angle, azimuth_angle, scale, steps, batch_size, fp16, checkpoint, *args, **kwargs):
        xs = [polar_angle]*batch_size
        ys = [azimuth_angle]*batch_size
        model, device = load_model(folder_paths.get_full_path("checkpoints", checkpoint), hf=fp16)

        # just for simplify
        input_im = Image.fromarray((255. * image[0]).numpy().astype(np.uint8))
        w, h = input_im.size
        if (w != 256) or (h != 256):
            input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0

        # input_im = Image.open('input.png')
        # input_imnp = np.asarray(input_im, dtype=np.float32) / 255.0
        outputs = predict_cam(model, input_im, xs, ys, scale=scale, device=device, n_samples=batch_size, ddim_steps=steps)

        return outputs
