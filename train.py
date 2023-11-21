import sys
import numpy as np
import pandas as pd
from skimage.io import imread
from tensorflow import keras
from keras import models, layers, backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (768, 768)
CROP_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_EPOCH = 10

DG_ARGS = dict(featurewise_center = False,
               samplewise_center = False,
               rotation_range = 45,
               width_shift_range = 0.1,
               height_shift_range = 0.1,
               shear_range = 0.01,
               zoom_range = [0.9, 1.25],
               horizontal_flip = True,
               vertical_flip = True,
               fill_mode = 'reflect',
               data_format = 'channels_last')


def rle_decoder(mask_rle):
    """
    Decode a Run-Length Encoded (RLE) mask string into a binary mask array.
    If the input is not a string, a zero-filled mask array is returned.

    Parameters:
    - mask_rle (str): The RLE encoded mask string.

    Returns:
    - numpy.ndarray: Binary mask array of size IMG_SIZE
    """

    if not isinstance(mask_rle, str):
        return np.zeros(IMG_SIZE)

    s = mask_rle.split()
    starts, lengths = (np.asarray(x, dtype=int) for x in (s[::2], s[1::2]))

    # Adjust start positions to be zero-based
    starts -= 1

    ends = starts + lengths

    mask = np.zeros(IMG_SIZE[0] * IMG_SIZE[1])
    assert ends[-1] <= len(mask)

    for start, end in zip(starts, ends):
        mask[start:end+1] = 1

    return mask.reshape(IMG_SIZE).T


def rle_masks_to_array(in_mask_list):
    """
    Convert a list of Run-Length Encoded (RLE) masks to a binary array.

    Parameters:
    - in_mask_list (list): A list of strings, each containing an RLE-encoded mask.

    Returns:
    - np.ndarray: A binary array representing the combined masks.

    Note:
    The input RLE masks are decoded using the rle_decoder function, and the resulting
    binary arrays are summed to create a final combined mask.
    """

    mask = np.zeros(IMG_SIZE)

    for rle_mask in  in_mask_list:
        mask += rle_decoder(rle_mask)

    return mask


def crop(image, mask):
    """
    Crop the given image and mask into smaller patches based on the specified
    crop size. Patches with ships are selected based on a mean value threshold of the mask.
    If image doesn't contain ship then return random patch

    Parameters:
    - image (numpy.ndarray): The input image to be cropped.
    - mask (numpy.ndarray): The mask associated with the input image.

    Returns:
    - images (list of numpy.ndarray): List of cropped image patches.
    - masks (list of numpy.ndarray): List of corresponding cropped mask patches.
    """

    mean_num = np.sum(mask) / np.prod(np.array(IMG_SIZE) / np.array(CROP_SIZE))

    # no ship case
    if mean_num == 0:
        i, j = np.array(CROP_SIZE) * np.random.randint(3, size=(2,))

        mask_crop = mask[i:i+CROP_SIZE[0], j:j+CROP_SIZE[1], :]
        image_crop = image[i:i+CROP_SIZE[0], j:j+CROP_SIZE[1], :]

        return [image_crop], [mask_crop]

    images = []
    masks = []

    for i in range(0, IMG_SIZE[0], CROP_SIZE[0]):
        for j in range(0, IMG_SIZE[1], CROP_SIZE[1]):
            mask_crop = mask[i:i+CROP_SIZE[0], j:j+CROP_SIZE[1], :]

            if np.sum(mask_crop) >= mean_num:
                images += [image[i:i+CROP_SIZE[0], j:j+CROP_SIZE[1], :]]
                masks += [mask_crop]

    return images, masks


def make_image_gen(in_df, train_img_dir, batch_size=BATCH_SIZE):
    """
    Generator function to yield batches of images and corresponding masks for training.

    Parameters:
    - in_df (pd.DataFrame): DataFrame containing information about images and their encoded masks.
    - batch_size (int, optional): Number of samples in each batch.

    Yields:
    - tuple: A tuple containing a batch of images and their corresponding masks.

    Note:
    The function utilizes image cropping to generate training samples.
    """

    all_batches = in_df.groupby("ImageId")["EncodedPixels"].apply(tuple)
    out_img = []
    out_mask = []
    while True:
        for img_id, rle_masks in all_batches.sample(batch_size).items():
            image = imread(train_img_dir + "/" + img_id)
            mask = rle_masks_to_array(rle_masks)
            mask = np.expand_dims(mask, axis=-1)

            image_patches, mask_patches = crop(image, mask)

            out_img += image_patches
            out_mask += mask_patches

            if len(out_img) >= BATCH_SIZE:
                yield np.stack(out_img[:BATCH_SIZE]), np.stack(out_mask[:BATCH_SIZE])
                out_img = out_img[BATCH_SIZE:]
                out_mask = out_mask[BATCH_SIZE:]


def create_aug_gen(in_gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    image_gen = ImageDataGenerator(**DG_ARGS)
    label_gen = ImageDataGenerator(**DG_ARGS)

    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x), next(g_y)


def conv_block(filters, activation="relu"):
    """
    Creates a convolutional block for Unet model.

    Parameters:
    - filters (int): number of filters in each convolutional layer.
    - activation (str): activation function for convolutional layers (default is "relu").

    Returns:
    - block (keras.Sequential)
    """

    block = keras.Sequential([
        layers.Conv2D(filters, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation(activation=activation),
        layers.Conv2D(filters, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation(activation=activation)
    ])

    return block


def decoder_block(filters, in1, in2):
    out = layers.UpSampling2D((2, 2))(in1)
    out = layers.concatenate([out, in2])
    out = conv_block(filters)(out)

    return out


def get_model():
    """
    Creates a U-Net convolutional neural network model for image segmentation.

    Parameters:
    - start_filters (int): number of filters in the first convolutional layer.
    - depth (int): number of downsampling and upsampling steps.

    Returns:
    - keras.models.Model

    Note:
    Returned model includes preprocessing
    """

    input_img = layers.Input(CROP_SIZE + (3,))

    # pretrained encoder
    encoder = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=preprocess_input(input_img)
    )
    encoder.trainable = False

    # downsampling blocks
    e1 = input_img
    e2 = encoder.get_layer("block2a_expand_activation").output
    e3 = encoder.get_layer("block3a_expand_activation").output
    e4 = encoder.get_layer("block4a_expand_activation").output

    # bottleneck
    x = encoder.get_layer("block6a_expand_activation").output

    # upsampling blocks
    x = decoder_block(512, x, e4)
    x = decoder_block(256, x, e3)
    x = decoder_block(128, x, e2)
    x = decoder_block(64, x, e1)

    # prediction layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid') (x)

    return models.Model(inputs=[input_img], outputs=[output])


def dice_score(y_true, y_pred):
    """
    Calculates the differentiable Dice score.

    Parameters:
    - y_true (tensor): binary mask.
    - y_pred (tensor): predicted binary mask.

    Returns:
    - dice score (tensor)
    """

    eps = K.epsilon()
    axis = list(range(1, len(y_true.shape)))

    intersection = K.sum(y_true * y_pred, axis=axis)
    total = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    return K.mean((2*intersection + eps) / (total + eps), axis=0)


def bce_dice_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + 1 - dice_score(y_true, y_pred)


def main():
    np.random.seed(42)

    if len(sys.argv) == 2:
        train_img_dir = sys.argv[1]
    else:
        print("You didn't specify directory")
        sys.exit(1)

    # creating data generators
    train_df = pd.read_csv("train_df.csv")
    val_df = pd.read_csv("val_df.csv")

    train_gen = make_image_gen(train_df, train_img_dir, batch_size=BATCH_SIZE)
    val_gen = make_image_gen(val_df, train_img_dir, batch_size=BATCH_SIZE)

    train_gen = create_aug_gen(train_gen)

    # model training
    seg_model = get_model()

    step_count = train_df["ImageId"].nunique() // BATCH_SIZE
    val_step_count = val_df["ImageId"].nunique() // BATCH_SIZE

    callback = ReduceLROnPlateau(
                            monitor="val_loss",
                            factor=0.1,
                            patience=2)

    seg_model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_score])
    loss_history = seg_model.fit(train_gen,
                                 validation_data=val_gen,
                                 steps_per_epoch=step_count,
                                 validation_steps = val_step_count,
                                 epochs=NUM_EPOCH,
                                 callbacks=[callback]
                                )

    # model saving
    seg_model.save_weights("weights.h5")


if __name__ == "__main__":
    main()

