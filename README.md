## Froked Neuralangelo Repository for robbin647

### My contribution

1. Added [eval_rgb.py](./eval_rgb.py)  
This is a script that can load a pretrained Neuralangelo model and output pairs of a source RGB image and RGB image rendered from same viewing direction as the source image. It also outputs a camera pose matrix (compressed .npz file) associated with that rendered RGB image.

<b> How to run it</b>  
Rerequisite: `pip install pyiqa`  or remove the `import pyiqa` at line #19. This package is required only if you want to use the `eval_metrics` function (line #73).   
The minimal way to run this script is to pass 3 command line parameters. The `--dump_dir` specifies the path to output each pair of RGB images and the associated camera pose matrix.

```shell
!torchrun eval_rgb.py \
    --dump_dir={path to dump results} \
    --config={path to Neuralangelo configuration file} \
    --checkpoint={the path to pretrained Neuralangelo checkpoint } \
```

At the current stage, it is required that before executing this script, the same set of RGB images & `transforms.json` that was used to train the Neuralangelo model must be present. This can be configured by setting `data.root` in the Neuralangelo configuration file or by `--data.root` command line argument.

```yaml
checkpoint:
  ...
cudnn:
  ...
data:
  root: /path/to/train/data <== Set this
  ...
```

<b>The output format</b>  
By default, the script will assign an interger id (strting from 0) to the output file each time an image is rendered from a viewing direction. So the names of outputs will be:

| Type of file | File name |
| -- | -- |
|Rendered RGB image | rendered_rgb_{id} |
|Training RGB image with the same viewing direction | gt_rgb_{id} |
|Camera pose matrix | pose_{id}.npz |

Remark: you can obtain the camera pose matrix by `numpy.load("pose_{id}.npz")["pose"]`. You can control how many images are rendered from model and the rendered image resolution by modifying the following config values in Neuralangelo configuration file.

```yaml
data:
    val:
        batch_size: 2   
        image_size:
        - 300 <== rendered image height
        - 400 <== rendered image width
        max_viz_samples: 16
        subset: 4 <== number of rendered images you want
```

### Citation

This repo is forked from official implementation of **Neuralangelo: High-Fidelity Neural Surface Reconstruction**.

[Zhaoshuo Li](https://mli0603.github.io/),
[Thomas Müller](https://tom94.net/),
[Alex Evans](https://research.nvidia.com/person/alex-evans),
[Russell H. Taylor](https://www.cs.jhu.edu/~rht/),
[Mathias Unberath](https://mathiasunberath.github.io/),
[Ming-Yu Liu](https://mingyuliu.net/),
[Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/)  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023

