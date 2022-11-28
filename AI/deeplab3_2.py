import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 20
DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50

project_images = ['../img/actor.jpg', '../img/longhair.jpg', '../img/hyunbin.png', '../img/jungmin.png', '../img/sangsoo.png', '../img/sukhun.png',  '../img/longhair.jpg']

colormap = loadmat("./human_colormap.mat")["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)

def load_data(image_list, mask_list):
  image = read_image(image_list)
  mask = read_image(mask_list, mask=True)
  return image, mask

def read_image(image_path, mask=False):
  image = tf.io.read_file(image_path)
  if mask:
    image = tf.image.decode_png(image, channels = 1)
    image.set_shape([None,None,1])
    image = tf.image.resize(images=image, size =[IMAGE_SIZE, IMAGE_SIZE])
  else:
    image = tf.image.decode_png(image, channels = 3)
    image.set_shape([None,None,3])
    image = tf.image.resize(images=image, size =[IMAGE_SIZE, IMAGE_SIZE])
    image = image/127.5 -1

  return image

def read_image2(image, mask=False):
    image = np.asarray(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(rgb, dtype=tf.float32)

    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size =[IMAGE_SIZE, IMAGE_SIZE])
    image = image/127.5 -1

    return image

def data_generator(image_list, mask_list):
  dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
  dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
  return dataset

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )

def plot_predictions2(img, colormap, model):
    img_y,img_x,_ = img.shape

    image_tensor = read_image2(img)
    prediction_mask = infer(image_tensor=image_tensor, model=model)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 3)
    overlay = get_overlay(image_tensor, prediction_colormap)

    prediction_colormap = changeBGR(prediction_colormap, (img_x,img_y))
    overlay = changeBGR(overlay, (img_x, img_y))

    print(prediction_colormap)

    cv2.imshow("img",img)
    cv2.imshow("prediction_colormap", prediction_colormap)
    cv2.imshow("overlay",overlay)

def changeBGR(image, size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size))
    return image

model = tf.keras.models.load_model('./model2.h5')

for i in project_images:
    i = cv2.imread(i)
    plot_predictions2(i, colormap, model)
    cv2.waitKey(0)