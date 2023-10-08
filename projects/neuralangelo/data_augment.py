import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import os.path as path
import cv2
from pdb import set_trace

ia.seed(1)

def clip(images):
    seq = iaa.Sequential([
        iaa.size.Resize(size={"height": 600, "width": 800})
    ])
    return seq(images=images)

def noise(images):
    """
        images: np.ndarray NxHxWxC where N: number of images, H: height, W: width, C: channel
    """
    seq = iaa.Sequential([
        # For each channel, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # Gaussian Noise with mean=0 and std=12.75
        iaa.AdditiveGaussianNoise(loc=0, scale=50, per_channel=True)
    ], random_order=False)
    return seq(images=images)

def blur(images, type="average"):
    """
    images: np.ndarray NxHxWxC where N: number of images, H: height, W: width, C: channel
    """
    if type == "average":
        seq = iaa.Sequential([
            
            # Kernel size is 6x8 (height, weight)
            iaa.AverageBlur(k = ((6,6), (8,8)))

        ], random_order=False) # apply augmenters in random order
    else: #type == "motion"
        seq = iaa.Sequential([
            
            # Kernel size is 6x8 (height, weight)
            # Fixed angle of motion blur = 45 degree
            # `direction=0.0` gives a uniform blur
            iaa.MotionBlur(k = 8, angle=-30, direction=0.0 )

        ], random_order=False) # apply augmenters in random order
    return seq(images=images)

def fog(images):
    """
    images: np.ndarray NxHxWxC where N: number of images, H: height, W: width, C: channel
    """
    seq = iaa.Sequential([
        
        # TODO: customizable fog parameters
        # Uniform Fog in 3D space 
        iaa.CloudLayer(intensity_mean=190, 
                       intensity_freq_exponent=-2.5, 
                       intensity_coarse_scale=0, 
                       alpha_min=0.1, 
                       alpha_multiplier=1.0,
                       alpha_size_px_max=1.0,
                       alpha_freq_exponent=-1.5,
                       sparsity=0.9,
                       density_multiplier=1.0)

    ], random_order=False) # apply augmenters in random order

    return seq(images=images)


if __name__ == "__main__":
    src_folder = "/research/d1/msc/txyang23/msc_project/dataset/dtu/dtu_scan24/image"
    degradation_type = "uniform_fog"
    dest_folder = path.join(src_folder, f"../img_{degradation_type}")
    if not path.exists(dest_folder):
        os.makedirs(dest_folder)
    # partially load 10 images into memory: 
    image_files = [file for file in os.listdir(src_folder) if file.endswith('.jpg') or file.endswith('.png')]
    image_files = image_files[:10]
    images = []
    for file in image_files:
        image_path = os.path.join(src_folder, file)
        image = cv2.imread(image_path)
        images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = clip(images)

    # call degradation subroutine
    if degradation_type == "noise":
        results = noise(images)
    elif degradation_type == "average_blur":
        results = blur(images)
    elif degradation_type == "motion_blur":
        results = blur(images, type="motion")
    elif degradation_type == "uniform_fog":
        results = fog(images)
    # write results to the destination
    for idx, file in enumerate(image_files):
        image_path = os.path.join(dest_folder, file)
        image = cv2.imwrite(image_path, results[idx])
    print("Done writing results")