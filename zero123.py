import os
import sys
import numpy as np
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import CLIPImageProcessor
from torch import autocast
from torchvision import transforms

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    
    sys.path.insert(0, os.path.dirname(__file__))
    model = instantiate_from_config(config.model)
    sys.path.pop(0)

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


def init_model(device, ckpt, half_precision=False):
    config = os.path.join(os.path.dirname(__file__), 'config/zero123.yaml')
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    print('Instantiating LatentDiffusion...')
    if half_precision:
        model = torch.compile(load_model_from_config(config, ckpt, device=device)).half()
    else:
        model = torch.compile(load_model_from_config(config, ckpt, device=device))
    #models['clip_fe'] = CLIPImageProcessor.from_pretrained(clip_vision)

    return model

@torch.no_grad()
def sample_model_batch(model, sampler, input_im, xs, ys, n_samples=4, precision='autocast', ddim_eta=1.0, ddim_steps=75, scale=3.0, h=256, w=256):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = []
            for x, y in zip(xs, ys):
                T.append([np.radians(x), np.sin(np.radians(y)), np.cos(np.radians(y)), 0])
            T = torch.tensor(np.array(T))[:, None, :].float().to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage(input_im).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            # print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            ret_imgs = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            del cond, c, x_samples_ddim, samples_ddim, uc, input_im
            torch.cuda.empty_cache()
            return ret_imgs

@torch.no_grad()
def predict_cam(model, imnp, xs, ys, device="cuda", n_samples=1, ddim_steps=75, scale=3.0):
    # raw_im = raw_im.resize([256, 256], Image.LANCZOS)
    # input_im_init = preprocess_image(models, raw_im, preprocess=False)
    input_im = transforms.ToTensor()(imnp).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1

    sampler = DDIMSampler(model)
    #ksampler("euler", inpaint_options={"random": True})
    #sampler = sampler_object("ddim")(model)
    # sampler.to(device)
    print("input_im", input_im)
    print("xs", xs)
    print("ys", ys)
    print("n_samples", n_samples)
    print("scale", scale)
    sampleimg = sample_model_batch(model, sampler, input_im, xs, ys, n_samples=n_samples, ddim_steps=ddim_steps, scale=scale)

    out_images = []
    for sample in sampleimg:    
        image = torch.from_numpy(rearrange(sample.numpy(), 'c h w -> h w c'))[None,]
        out_images.append(image)
    return out_images
