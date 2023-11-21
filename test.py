import sys
from PIL import Image
import numpy as np
from skimage.io import imread
from skimage.util import montage


from train import get_model, IMG_SIZE, CROP_SIZE


def make_prediction(image, model):
    if image.ndim < 4:
        image = np.expand_dims(image, axis=0)

    masks = []

    for i in range(0, IMG_SIZE[0], CROP_SIZE[0]):
        for j in range(0, IMG_SIZE[1], CROP_SIZE[1]):
            patches = image[:, i:i + CROP_SIZE[0], j:j + CROP_SIZE[1], :]
            masks += [make_prediction_patch(patches, model)]

    return montage(masks)


def make_prediction_patch(patches, model):
    if patches.ndim < 4:
        patches = np.expand_dims(patches, axis=0)

    final_mask = np.zeros(CROP_SIZE, dtype=float)

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


def main():
    img_path = sys.argv[1]
    mask_path = img_path[:-4] + "_mask.png"

    image = imread(img_path)

    model = get_model()
    model.load_weights("weights.h5")

    mask = make_prediction(image, model)

    plt.imshow(mask, cmap="gray_r")
    plt.axis("off")
    plt.savefig(mask_path)


if __name__ == "__main__":
    main()
