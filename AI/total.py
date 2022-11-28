import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms, models
import cv2
import os
from glob import glob
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras

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

labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class removeBack:
    def __init__(self):
        self.model_1 = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.cmap = plt.cm.get_cmap('tab20c')
        self.colors = (self.cmap(np.arange(self.cmap.N)) * 255).astype(np.int)[:, :3].tolist()
        np.random.seed(2020)
        np.random.shuffle(self.colors)
        self.colors.insert(0, [0, 0, 0]) # background color must be black
        self.colors = np.array(self.colors, dtype=np.uint8)

        self.palette_map = np.empty((10, 0, 3), dtype=np.uint8)
        self.legend = []

        for i in range(21):
            self.legend.append(mpatches.Patch(color=np.array(self.colors[i]) / 255., label='%d: %s' % (i, labels[i])))
            self.c = np.full((10, 10, 3), self.colors[i], dtype=np.uint8)
            self.palette_map = np.concatenate([self.palette_map, self.c], axis=1)

    def segment(self, net, img):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model_1.to('cuda')

        output = self.model_1(input_batch)['out'][0] # (21, height, width)

        output_predictions = output.argmax(0).byte().cpu().numpy() # (height, width)

        r = Image.fromarray(output_predictions).resize((img.shape[1], img.shape[0]))
        r.putpalette(self.colors)

        return r, output_predictions

    def changeBGR(self, image, size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size))
        return image

    def getremovemask(self,file_name):
        img = np.array(Image.open(file_name))
        fg_h, fg_w, _ = img.shape
        segment_map, pred = self.segment(self.model_1, img)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        numpy_image=np.array(segment_map)
        opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        _, mask = cv2.threshold(opencv_image,10,255, cv2.THRESH_BINARY)

        img = self.changeBGR(img, (fg_w, fg_h))
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        return mask
class hairsegement:
    def __init__(self):
        self.model2 = tf.keras.models.load_model('./model2.h5')

    def read_image2(self,image, mask=False):
        image = np.asarray(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(rgb, dtype=tf.float32)

        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1

        return image

    def infer(self,model, image_tensor):
        predictions = model.predict(np.expand_dims((image_tensor), axis=0))
        predictions = np.squeeze(predictions)
        predictions = np.argmax(predictions, axis=2)
        return predictions

    def decode_segmentation_masks(self,mask, colormap, n_classes):
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

    def get_overlay(self,image, colored_mask):
        image = tf.keras.preprocessing.image.array_to_img(image)
        image = np.array(image).astype(np.uint8)
        overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
        return overlay

    def plot_predictions2(self,img, colormap, model):
        img_y, img_x, _ = img.shape

        image_tensor = self.read_image2(img)
        prediction_mask = self.infer(image_tensor=image_tensor, model=model)
        prediction_colormap = self.decode_segmentation_masks(prediction_mask, colormap, 3)
        overlay = self.get_overlay(image_tensor, prediction_colormap)

        prediction_colormap = self.changeBGR(prediction_colormap, (img_x, img_y))
        overlay = self.changeBGR(overlay, (img_x, img_y))

        return overlay, prediction_colormap

    def changeBGR(self,image, size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size))
        return image

    def gethairsegement(self, file_name):
        img = cv2.imread(file_name)
        overlay, prediction_colormap = self.plot_predictions2(img, colormap, self.model2)
        return overlay, prediction_colormap

def runSegement(file_name):
    mask = removeBack().getremovemask(file_name)
    overlay, prediction_colormap = hairsegement().gethairsegement(file_name)


    overlay = cv2.bitwise_and(overlay,overlay,mask=mask)
    prediction_colormap = cv2.bitwise_and(prediction_colormap,overlay,mask=mask)

    cv2.imshow("prediction_colormap", prediction_colormap)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)

runSegement("../img/sukhun.png")