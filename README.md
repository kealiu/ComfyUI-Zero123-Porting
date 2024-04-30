# ComfyUI Node for Zero-1-to-3: Zero-shot One Image to 3D Object

This is an unofficial porting of [Zero123:https://zero123.cs.columbia.edu/](https://zero123.cs.columbia.edu/) for ComfyUI, Zero123 is a framework for changing the camera viewpoint of an object given just a single RGB image.

This porting enable you generate 3D rotated image in ComfyUI.

![Functions](https://github.com/cvlab-columbia/zero123/blob/main/teaser.png)

# Quickly Start

After install this node, download the [sample workflow](sample/simple_workflow.json) to start trying.

## PREREQUISITES

- INPUT `image` must `square` (width=height), otherwise, this node will automatically trans it forcely
- INPUT `image` should be an `object` with **white background**, which means you need preprocess of image.
- OUTPUT `image` only support `256x256` (fixed) currently, you can upscale it later.

# Explains

## Input and Output

### INPUT

- image: input image, should be an `square` image, and an `object` with `white backgroup`.
- polar_angle: angle of `x` axis, turn up or down
    - `<0.0`: turn up
    - `>0.0`: turn down
- azimuth_angle: angle of `y` axis, turn left or right
    - `<0.0`: turn left
    - `>0.0`: turn right
- scale: `z` axis, `far away` or `near`;  
    - `>1.0` : means bigger, or `near`;
    - `0<1<1.0` : means smaller, or `far away`
    - `1.0` : mean no change
- steps: `75` is the default value by original `zero123` repo, do not smaller then `75`.
- batch_size: how many images you do like to generated. 
- fp16: whether to load model in `fp16`. enable it can speed up and save GPU mem.
- checkpoint: the model you select, `zero123-xl` is the lates one.
- height: output height, fix to 256, information only
- width: output width, fix to 256, information only
- sampler: cannot change, information only
- scheduler: cannot change, information only

### OUTPUT

- images: the output images

## Node and Workflow

![simple workflow](images/Zero123-Simple.png)

## Tips

- for proprecess image, segment out the subject, and remove all backgroup.
- use image corp to focus the main subject, and make a squre image
- try multi images and select the best one
- upscale for final usage.

# Installation

## By ComfyUI Manager

### Customer Nodes 

search `zero123` and select this repo, install it.

### Models

search `zero123` and install the model you like. `zero123-xl.ckpt` is the latest one.

## Manually Installation

### Customer Nodes 

```
cd ComfyUI/custom_nodes
git clone https://github.com/kealiu/ComfyUI-Zero123-Porting.git
cd ComfyUI-Zero123-Porting
pip install -r requirements.txt
```

And then, restart `ComfyUI`, and refresh your browser.

### Models

check out [`model-list.json`](model-list.json) for modules download URL, their should be place under **`ComfyUI/models/checkpoints/zero123/`**

# Thanks to

[Zero-1-to-3: Zero-shot One Image to 3D Object](https://github.com/cvlab-columbia/zero123)

```
@misc{liu2023zero1to3,
      title={Zero-1-to-3: Zero-shot One Image to 3D Object}, 
      author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
      year={2023},
      eprint={2303.11328},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
