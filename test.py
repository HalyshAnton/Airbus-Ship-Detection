import argparse
import numpy as np
from skimage.io import imread
from skimage.util import montage
from PIL import Image


from train import get_model


def make_prediction(image, model, img_size, opt):
    """
    Make semantic segmentation prediction for a given input image using a specified model.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.
    - model: The semantic segmentation model to use for prediction.
    - img_size (tuple): Tuple specifying the dimensions (height, width) of the input image.
    - opt: Command-line arguments and options obtained from `parse_opt()`.

    Returns:
    - numpy.ndarray: predicted binary mask with shape img_size
    """
    
    if image.ndim < 4:
        image = np.expand_dims(image, axis=0)

    masks = []

    for i in range(0, img_size[0], opt.crop):
        for j in range(0, img_size[1], opt.crop):
            patches = image[:, i:i + opt.crop, j:j + opt.crop, :]
            masks += [make_prediction_patch(patches, model)]

    return montage(masks)


def make_prediction_patch(patches, model):
    """
    Make semantic segmentation prediction for a given patch using a specified model.
    Apply horizontal and vertical flips for robust prediction

    Parameters:
    - patches (numpy.ndarray): Input patch as a NumPy array.
    - model: The semantic segmentation model to use for prediction.

    Returns:
    - numpy.ndarray: Predicted segmentation mask for the input patch
    """
    
    if patches.ndim < 4:
        patches = np.expand_dims(patches, axis=0)

    final_mask = np.zeros(patches.shape[1:3], dtype=float)

    for hor in (1, -1):
        for ver in (1, -1):
            img = patches

            # flip image
            img = img[:, ::hor, ::ver, :]

            # prediction
            mask = model(img).numpy()

            # flip to original position
            mask = mask[0, ::hor, ::ver, 0]

            final_mask += mask

    final_mask /= 4
    final_mask[final_mask > 0.5] = 1
    final_mask[final_mask <= 0.5] = 0

    return final_mask.astype(int)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to image")
    parser.add_argument("--crop", type=int, default=256, help="size for cropping")
    parser.add_argument("--weights", type=str, default="weights.h5", help="path to file with model weights")

    return parser.parse_args()


def main():
    opt = parse_opt()

    img_path = opt.image
    mask_path = img_path[:-4] + "_mask.jpg"

    image = imread(img_path)
    img_size = image.shape[:2]

    model = get_model((opt.crop, opt.crop, 3))
    model.load_weights(opt.weights)

    mask = make_prediction(image, model, img_size, opt)

    mask_im = Image.fromarray(mask.astype('uint8') * 255)
    mask_im.save(mask_path)


if __name__ == "__main__":
    main()
