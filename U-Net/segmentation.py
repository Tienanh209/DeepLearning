import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import os

input_dir = "images/"
target_dir = "annotations/trimaps"
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)


# Display input image #2
display(Image(filename=input_img_paths[2]))

# Display auto-contrast version of corresponding target (per-pixel categories)
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[2]))
display(img)

# Building the convolutional block
# def ConvBlock(inputs, filters=64):
#     # Taking first input and implementing the first conv block
#     conv1 = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(inputs)
#     batch_norm1 = layers.BatchNormalization()(conv1)
#     act1 = layers.ReLU()(batch_norm1)
#
#     # Taking first input and implementing the second conv block
#     conv2 = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(act1)
#     batch_norm2 = layers.BatchNormalization()(conv2)
#     act2 = layers.ReLU()(batch_norm2)
#
#     return act2
#
#
# # Building the encoder
# def encoder(inputs, filters=64):
#     # Collect the start and end of each sub-block for normal pass and skip connections
#     enc1 = ConvBlock(inputs, filters)
#     MaxPool1 = layers.MaxPooling2D(strides=(2, 2))(enc1)
#     return enc1, MaxPool1
#
#
# # Building the decoder
# def decoder(inputs, skip, filters=64):
#     # Upsampling and concatenating the essential features
#     Upsample = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
#     Connect_Skip = layers.Concatenate()([Upsample, skip])
#     out = ConvBlock(Connect_Skip, filters)
#     return out
#
#
# def get_U_Net(image_size):
#     inputs = layers.Input(image_size)
#
#     # Construct the encoder blocks and increasing the filters by a factor of 2
#     skip1, encoder_1 = encoder(inputs, 64)
#     skip2, encoder_2 = encoder(encoder_1, 64 * 2)
#     skip3, encoder_3 = encoder(encoder_2, 64 * 4)
#     skip4, encoder_4 = encoder(encoder_3, 64 * 8)
#
#     # Preparing the next block
#     conv_block = ConvBlock(encoder_4, 64 * 16)
#
#     # Construct the decoder blocks and decreasing the filters by a factor of 2
#     decoder_1 = decoder(conv_block, skip4, 64 * 8)
#     decoder_2 = decoder(decoder_1, skip3, 64 * 4)
#     decoder_3 = decoder(decoder_2, skip2, 64 * 2)
#     decoder_4 = decoder(decoder_3, skip1, 64)
#
#     outputs = layers.Conv2D(3, 1, padding="same", activation="sigmoid")(decoder_4)
#
#     model = models.Model(inputs, outputs)
#     return model
#
# input_shape = (160, 160, 3)
# model = get_U_Net(input_shape)
# # model.summary()
#
# model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy")
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
# ]
#
# epochs = 15
#
# model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
#
