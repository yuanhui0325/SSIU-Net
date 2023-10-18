import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import random
import glob
import io
import numpy as np
import PIL.Image as pil_image
import torch.utils.data as data

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)


class Dataset(object):
    def __init__(self, images_dir, patch_size, jpeg_quality, use_augmentation=False, use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.jpeg_quality = jpeg_quality
        self.use_augmentation = use_augmentation
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            label = tf.read_file(self.image_files[idx])
            label = tf.image.decode_jpeg(label, channels=3)
            label = pil_image.fromarray(label.numpy())
        else:
            label = pil_image.open(self.image_files[idx]).convert('RGB')

        if self.use_augmentation:
            # randomly rescale image
            if random.random() <= 0.5:
                scale = random.choice([0.9, 0.8, 0.7, 0.6])
                label = label.resize((int(label.width * scale), int(label.height * scale)), resample=pil_image.BICUBIC)

            # randomly rotate image
            if random.random() <= 0.5:
                label = label.rotate(random.choice([90, 180, 270]), expand=True)

        # randomly crop patch from training set
        crop_x = random.randint(0, label.width - self.patch_size)
        crop_y = random.randint(0, label.height - self.patch_size)
        label = label.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        # additive jpeg noise
        buffer = io.BytesIO()
        label.save(buffer, format='jpeg', quality=self.jpeg_quality)
        input = pil_image.open(buffer)

        input = np.array(input).astype(np.float32)
        label = np.array(label).astype(np.float32)
        input = np.transpose(input, axes=[2, 0, 1])
        label = np.transpose(label, axes=[2, 0, 1])

        # normalization
        input /= 255.0
        label /= 255.0

        return input, label

    def __len__(self):
        return len(self.image_files)


def load_h5(h5txt_path):
    with open(h5txt_path, "r") as f:
        txt = f.read()
    data_li = []
    label_li = []
    for filename in txt.split():
        h5 = h5py.File(os.path.join(os.path.dirname(h5txt_path), filename), 'r')
        data = h5['data_hui'][:]
        label = h5['label_hui'][:]
        data_li.append(data)
        label_li.append(label)
    datas = np.concatenate(data_li)
    # datas = datas.transpose(0, 3, 1, 2)
    datas_norm = datas.astype(np.float32)
    labels = np.concatenate(label_li)
    # labels = labels.transpose(0, 3, 1, 2)
    labels_norm = labels.astype(np.float32)
    return datas_norm, labels_norm


def load_h5_manual(h5txt3D_path):         ###it's incomplete
    with open(h5txt3D_path, "r") as f:
        txt = f.read()
    data_snake_li = []
    label_snake_li = []
    data_hui_li=[]
    label_hui_li=[]
    for filename in txt.split():
        print(filename)
        h5 = h5py.File(os.path.join(os.path.dirname(h5txt3D_path), filename), 'r')
        data_snake = h5['data_snake'][:]
        label_snake = h5['label_snake'][:]
        data_snake_li.append(data_snake)
        label_snake_li.append(label_snake)
        data_hui = h5['data_hui'][:]
        label_hui = h5['label_hui'][:]
        data_hui_li.append(data_hui)
        label_hui_li.append(label_hui)
    datas_snake = np.concatenate(data_snake_li)
    datas_hui = np.concatenate(data_hui_li)
    data = np.concatenate((datas_snake, datas_hui), axis=0)
    # datas = datas.transpose(0, 3, 1, 2)
    datas_norm = data.astype(np.float32)
    labels_snake = np.concatenate(label_snake_li)
    labels_hui = np.concatenate(label_hui_li)
    label = np.concatenate((labels_snake, labels_hui), axis=0)
    # labels = labels.transpose(0, 3, 1, 2)
    labels_norm = label.astype(np.float32)
   # k = random.randrange(1, 4)
   # k = 1
   # datas_aug = np.rot90(datas_norm, k, axes=[2, 3])
   # labels_aug = np.rot90(labels_norm, k, axes=[2, 3])
   # datas_norm = np.concatenate((datas_norm, datas_aug), axis=0)
   # labels_norm = np.concatenate((labels_norm, labels_aug), axis=0)
    return datas_norm, labels_norm


class h5DataSet_more(data.Dataset):
    def __init__(self, h5txt3D_path, img_pth):
        data, label = load_h5_manual(h5txt3D_path)
        h5 = h5py.File(img_pth, 'r')                   # get dataset from 2d img -- DIV2K
        datas = h5['data'][:]
        labels = h5['label'][:]
        state = np.random.get_state()
        np.random.shuffle(datas)
        np.random.set_state(state)
        np.random.shuffle(labels)
        rand_len = int(datas.shape[0] * 0.2)
        data_img = datas[0:rand_len, :, :, :]
        label_img = labels[0:rand_len, :, :, :]
        self.data = np.concatenate((data, data_img), axis=0)
        self.label = np.concatenate((label, label_img), axis=0)
        self.length = self.label.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.label[index, :, :, :]

    def __len__(self):
        return self.length


class h5Dataset(data.Dataset):
    def __init__(self, txtPth):
        data, label = load_h5(txtPth)
       # k = random.randrange(1,4)
       # data_aug = np.rot90(data, k, axes=[2, 3])
       # label_aug = np.rot90(label, k, axes=[2, 3])
        self.data = data
      #  print(self.data.shape)
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, index):
        data = self.data[index, :, :, :]
        label = self.label[index, :, :, :]
       # k = random.randrange(1,5)
       # data = np.rot90(data, k, axes=[1, 2])
       # label = np.rot90(label, k ,axes=[1, 2])
        return data, label

    def __len__(self):
        return self.length
