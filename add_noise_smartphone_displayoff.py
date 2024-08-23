import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from os.path import join
import numpy as np
import scipy.io as sio
import cv2
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import math
import threading
from util.visualizer import Visualizer
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from skimage.feature import blob_dog, blob_log, blob_doh
from ms_ssim import tf_ssim, tf_ms_ssim_resize

from wave_optics import Camera, set_params, capture, capture_alpha, wiener_deconv, get_interfer_img, get_wiener_loss
from loss import Loss
from utils import print_opt, crop_image
from imageStreamingTest import ImageStreamingTest

import torch
from torchvision import models
from torchvision import transforms
from GradCAMutils import GradCAM, show_cam_on_image, center_crop_img

import time
import dropbox



# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'

# import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# device_gradcam = torch.device("cpu")
device_gradcam = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hort, port = "192.168.0.126", 8100
# access_token = 'sl.BXiBXk5eop5TVY2CzkmNhh3I-EfNDndJXh5vnl0iuA3etfAAOCvAKBVdaV3EDsoAOc2GfguqjOrPdn4-UowcWqW6V8k4Mi5kLgaS7FOHRlmRW8Ly62us1uoHFI9iC2bEqCaisOieS2Pq'
# import torch
# from torch.backends import cudnn
# print(torch.cuda.is_available())
# print(cudnn.is_available())
# print(tf.config.list_physical_devices('GPU'))

# print(tf.__version__)
# print(tf.__path__)

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# #allow growth
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# per_process_gpu_memory_fraction
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# session = InteractiveSession(config=config)


# config = tf.compat.v1.ConfigProto()#allow_soft_placement=True
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)
# sess.as_default()
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# physical_gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_gpus[0], True)
# tf.config.experimental.set_memory_growth(physical_gpus[1], True)


# from tensorflow.keras import backend as K
# import tensorflow as tf

# session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
# sess = tf.Session(config=session_config)
# keras.set_session(sess)



# tf dataloader
batch_size = 1  # 12
img_height = 224
img_width = 224
img_number = 60000
# tf.keras.backend.set_floatx('float32')

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def random_size(image, target_size=None):
    #（batch,H,W,C）
    np.random.seed(0)
    _, height, width, _ = tf.shape(image)
    if target_size is None:
        # for test
        # target size is fixed
        target_size = np.random.randint(*c.short_side_scale)
    if height < width:
        size_ratio = target_size / height
    else:
        size_ratio = target_size / width

    # width and height in cv2 are opposite to np.shape()
    if height!=0 and width!=0:
        if height.numpy() * size_ratio.numpy() < target_size or width.numpy() * size_ratio.numpy() < target_size:
            resize_shape = (
            math.ceil(width.numpy() * size_ratio.numpy()), math.ceil(height.numpy() * size_ratio.numpy()))
        else:
            resize_shape = (int(width.numpy() * size_ratio.numpy()), int(height.numpy() * size_ratio.numpy()))
        # resize_shape = (int(height.numpy() * size_ratio.numpy()), int(width.numpy() * size_ratio.numpy()))
        # resize_shape = (int(width.numpy() * size_ratio.numpy()), int(height.numpy() * size_ratio.numpy()))
        image_resized = tf.image.resize(image, resize_shape)
    else:
        image_resized = image
    # resize_shape = (int(height.numpy() * size_ratio.numpy()), int(width.numpy() * size_ratio.numpy())) ## Right one
    return image_resized

def random_size_numpy(image, target_size=None):
    # （H,W,C）
    np.random.seed(0)
    height, width, _ = np.shape(image)
    if target_size is None:
        # for test
        # target size is fixed
        target_size = np.random.randint(*c.short_side_scale)
    if height < width:
        size_ratio = target_size / height
    else:
        size_ratio = target_size / width

    if height * size_ratio < target_size or width * size_ratio < target_size:
        resize_shape = (math.ceil(width * size_ratio), math.ceil(height * size_ratio))
    else:
        resize_shape = (int(width * size_ratio), int(height * size_ratio))
    # resize_shape = (int(width * size_ratio), int(height * size_ratio)) ## Right one/correct one
    # resize_shape = (int(height * size_ratio), int(width * size_ratio))
    ### width and height in cv2 are opposite to np.shape() ###
    return cv2.resize(image, resize_shape, interpolation=cv2.INTER_NEAREST) #INTER_NEAREST INTER_AREA

def center_crop(image):
    # （batch,H,W,C）
    _, height, width, _ = tf.shape(image)
    # input_height, input_width, _ = c.input_shape
    input_height = img_height
    input_width = img_width
    crop_x = (width - input_width) // 2
    crop_y = (height - input_height) // 2
    return image[:, crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

def center_crop_numpy(image):
    # （H,W,C）
    height, width, _ = np.shape(image)
    # input_height, input_width, _ = c.input_shape
    input_height = img_height
    input_width = img_width
    crop_x = (width - input_width) // 2
    crop_y = (height - input_height) // 2
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

def load_data():

    # base_dir = os.path.dirname(os.path.dirname(__file__))
    # # 获取当前文件目录
    # data_path = os.path.abspath(os.path.join(base_dir, 'Train/Poled/HQ/*.png', ""))
    # # 改为绝对路径
    # # 获取文件拼接后的路径
    # D: / Dropbox / TuD work / ScreenAI_Privacy_Underscreen / UPC_ICCP21_Code - main / Train / Poled / HQ / *.png

    # todo: change path to your training image directory
    if opt.images_mode == 'images_test':
        train_ds = tf.data.Dataset.list_files('miniimagenet/images_test/*.JPG', shuffle=False)  # images images_test
    else:
        train_ds = tf.data.Dataset.list_files('miniimagenet/images/*.JPG', shuffle=False)  # images images_test
    # train_ds = tf.data.Dataset.list_files('miniimagenet/images_test/*.jpg', shuffle=False) #images images_test
    # train_ds = tf.data.Dataset.list_files('/data/volume_2/miniimagenet/images/*.jpg', shuffle=False)
    train_ds_name = train_ds
    # train_ds = tf.data.Dataset.list_files('D:/Dropbox/TuD work/ScreenAI_Privacy_Underscreen/UPC_ICCP21_Code-main/Train/Toled/HQ/*.png', shuffle=False)
    # train_ds = tf.data.Dataset.list_files('Train/Poled/HQ/*.png', shuffle=False)
    # train_ds = tf.data.Dataset.list_files(data_path, shuffle=False)
    # train_ds = train_ds.shuffle(240, reshuffle_each_iteration=True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def resize_and_rescale(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3, dct_method='INTEGER_ACCURATE')
        img = tf.cast(img, dtype=tf.float32) / 255


        # data augmentation
        # img = random_size(img, target_size=256)
        # img = center_crop(img)
        # img = tf.cast(img, dtype=tf.float32) / 255

        # img = tf.compat.v1.image.resize_bilinear(img, [img_height, img_width])
        # # img = tf.image.resize(img, [img_height, img_width])
        # img = tf.image.resize_with_crop_or_pad(img, img_height, img_width)
        # img.set_shape([img_height, img_width, 3])
        # img = tf.image.random_crop(img, size=tf.constant([img_height, img_width, 3]))
        # img = tf.image.resize_with_crop_or_pad(img, size=tf.constant([img_height, img_width, 3]))

        # img = cv2.imread(img_path.numpy().decode()).astype(np.float32)
        # img = cv2.imread(img_path)
        # img = random_size(img, target_size=256)
        # img = center_crop(img)
        # img = tf.cast(img, dtype=tf.float32) / 255
        return img

    train_ds = train_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    return train_ds, train_ds_name

def load_pattern():
    # pattern_path = 'D:/Dropbox/TuD work/ScreenAI_Privacy_Underscreen/UPC_ICCP21_Code-main/data/pixelPatterns/POLED_42.png'
    pattern_path = 'data/pixelPatterns/POLED_42.png'
    # pattern_path = 'data/pixelPatterns/POLED_21.png'
    # pattern_path = '/data/volume_2/optimize_display_POLED_400PPI/data/pixelPatterns/POLED_42.png'
    pattern = cv2.imread(pattern_path,0)

    return pattern




def optimize_pattern_with_data(opt):

    tf.keras.backend.set_floatx('float32')

    # visualization and log
    vis = Visualizer(opt)
    log_dir = join('log', opt.display_env)  # directory that saves optimized pixel control and training log
    os.makedirs(log_dir, exist_ok=True)
    logfile = open('%s/log.txt' % log_dir, 'w')
    print_opt(logfile, opt)  # print opt parameters

    # set up pattern
    # pattern = load_pattern()

    # set up camera
    # cameraOpt = set_params()
    # camera = Camera(pattern)


    # set up optimization
    optimizer = keras.optimizers.Adam(learning_rate=opt.lr, beta_1=0.9)#lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    # criterion = Loss(opt, cameraOpt)
    # vars = []               # variables which we track gradient
    train_ds, train_ds_name = load_data()  # load training dataset

    file_name = []
    path_name = []
    for fileindex in train_ds_name:
        if opt.images_mode == 'images_test':
            file_name.append(str(fileindex.numpy())[27:-1])  # python的切片左闭右开
            path_name.append(str(fileindex.numpy().decode('utf-8')))
        else:
            file_name.append(str(fileindex.numpy())[22:-1])
            path_name.append(str(fileindex.numpy().decode('utf-8')))

    web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
    img_dir = os.path.join(web_dir, opt.images_mode)  # opt.images_mode 'images'
    img_temp_dir = os.path.join(img_dir, opt.save_temp_dir)
    mkdirs([img_dir, img_temp_dir])


    txt_path = os.path.join(img_temp_dir, 'orders.txt')  # C:\Users\yhtxy\Desktop\android_test.txt
    cap_path = os.path.join(img_temp_dir, 'capture.jpg')
    img_cap_dir = os.path.join(img_dir, opt.save_cap_dir)
    mkdirs([img_dir, img_cap_dir])

    # dropbox_path = '/input.txt'
    # # //Dropbox/Test/android_test.txt
    # client = dropbox.Dropbox(access_token)

    colorstatusbar_flag = opt.statusbarcolor_flag  # 0 red, 1 green, 2 blue, 3 white...
    ssim_min = 1
    max_sum_power = opt.maxscreen_brightness

    # _, _, _, u1_RGB_real_mask = camera(None, DoR_option='small', mode_option = 'preprocessing')
    # u1_RGB_real_mask_temp = np.array(u1_RGB_real_mask)
    #
    # DOR_height, DOR_width = np.shape(u1_RGB_real_mask_temp)

    # all_interfer_pixel_index = np.unique(u1_RGB_real_mask_temp)
    # all_interfer_pixel_index = all_interfer_pixel_index[all_interfer_pixel_index != 0]
    # all_interfer_pixel_index = all_interfer_pixel_index.astype(int)
    # all_interfer_pixel_trans_index = np.zeros_like(all_interfer_pixel_index)
    #
    # all_interfer_pixel_red_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
    # all_interfer_pixel_green_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
    # all_interfer_pixel_blue_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)

    # for ii in range(len(all_interfer_pixel_index)):
    #     if all_interfer_pixel_index[ii] % 3 == 2:
    #         all_interfer_pixel_trans_index[ii] = 2  # Red
    #         all_interfer_pixel_red_index[ii] = max_sum_power  # power for red pixel
    #     elif all_interfer_pixel_index[ii] % 3 == 0:
    #         all_interfer_pixel_trans_index[ii] = 3  # Green
    #         all_interfer_pixel_green_index[ii] = max_sum_power  # power for red pixel
    #     elif all_interfer_pixel_index[ii] % 3 == 1:
    #         all_interfer_pixel_trans_index[ii] = 4  # Blue
    #         all_interfer_pixel_blue_index[ii] = max_sum_power  # power for red pixel
    #
    # all_red_pixel_index = np.where(all_interfer_pixel_trans_index == 2)
    # all_green_pixel_index = np.where(all_interfer_pixel_trans_index == 3)
    # all_blue_pixel_index = np.where(all_interfer_pixel_trans_index == 4)

    # all_Red_mask = np.zeros_like(u1_RGB_real_mask_temp)
    # all_Green_mask = np.zeros_like(u1_RGB_real_mask_temp)
    # all_Blue_mask = np.zeros_like(u1_RGB_real_mask_temp)
    #
    # for iii in range(len(all_red_pixel_index[0])):
    #     all_Red_mask = np.where(u1_RGB_real_mask_temp == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
    #                             all_interfer_pixel_index[all_red_pixel_index[0][iii]], all_Red_mask)
    #
    # for jjj in range(len(all_green_pixel_index[0])):
    #     all_Green_mask = np.where(u1_RGB_real_mask_temp == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
    #                               all_interfer_pixel_index[all_green_pixel_index[0][jjj]], all_Green_mask)
    #
    # for kkk in range(len(all_blue_pixel_index[0])):
    #     all_Blue_mask = np.where(u1_RGB_real_mask_temp == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
    #                              all_interfer_pixel_index[all_blue_pixel_index[0][kkk]], all_Blue_mask)


    # all_display_red_mask_ori = all_Red_mask  # tf.cast(Red_mask)
    # all_display_green_mask_ori = all_Green_mask  # tf.cast(Green_mask)
    # all_display_blue_mask_ori = all_Blue_mask  # tf.cast(Blue_mask)
    #
    # all_interfer_pixel = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
    # all_interfer_pixel = tf.constant(all_interfer_pixel)


    # for iii in range(len(all_red_pixel_index[0])):
    #     all_display_red_mask_ori = tf.where(
    #         all_display_red_mask_ori == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
    #         all_interfer_pixel[all_red_pixel_index[0][iii]],
    #         all_display_red_mask_ori)  # mapped_pixel[iii], display_red_mask)
    # for jjj in range(len(all_green_pixel_index[0])):
    #     all_display_green_mask_ori = tf.where(
    #         all_display_green_mask_ori == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
    #         all_interfer_pixel[all_green_pixel_index[0][jjj]],
    #         all_display_green_mask_ori)  # mapped_pixel[len(red_pixel_index[0]) + jjj], display_green_mask)
    # for kkk in range(len(all_blue_pixel_index[0])):
    #     all_display_blue_mask_ori = tf.where(
    #         all_display_blue_mask_ori == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
    #         all_interfer_pixel[all_blue_pixel_index[0][kkk]], all_display_blue_mask_ori)
    #
    # all_display_RGB_mask_ori = np.stack(
    #     [all_display_red_mask_ori, all_display_green_mask_ori, all_display_blue_mask_ori],
    #     axis=2)



    ## CNN model
    if opt.pretrained == 'resnet50':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100

        weights_path_gradcam = "./resNet50_miniimagenet_Nointerfer_full_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet18':
        model_gradcam = models.resnet18(num_classes=100).to(device_gradcam)  # num_classes=100

        weights_path_gradcam = "./resNet18_miniimagenet_Nointerfer_full_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet50_adv':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100

        weights_path_gradcam = "./resNet50_miniimagenet_PGD_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'mobilenet_v3_large':
        model_gradcam = models.mobilenet_v3_large(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./mobilenet_v3_large_miniimagenet_Nointerfer_full_SGD.pth"
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.features[-1]]
    elif opt.pretrained == 'mobilenet_v3_small':
        model_gradcam = models.mobilenet_v3_small(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./mobilenet_v3_small_miniimagenet_Nointerfer_full_SGD.pth"
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.features[-1]]
    elif opt.pretrained == 'shufflenet_v2_x0_5':
        model_gradcam = models.shufflenet_v2_x0_5(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./shufflenetv2_x0_5_miniimagenet_Nointerfer_full_SGD.pth"
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.conv5]
    elif opt.pretrained == 'shufflenet_v2_x1_0':
        model_gradcam = models.shufflenet_v2_x1_0(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./shufflenetv2_x1_miniimagenet_Nointerfer_full_SGD.pth"
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.conv5]
    elif opt.pretrained == 'shufflenet_v2_x1_5':
        model_gradcam = models.shufflenet_v2_x1_5(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./shufflenetv2_x1_5_miniimagenet_Nointerfer_full_SGD.pth"
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.conv5]
    elif opt.pretrained == 'shufflenet_v2_x2_0':
        model_gradcam = models.shufflenet_v2_x2_0(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./shufflenetv2_x2_miniimagenet_Nointerfer_full_SGD.pth"
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.conv5]


    cam = GradCAM(model=model_gradcam, target_layers=target_layers, use_cuda=False)

    # model 2 for predict
    if opt.pretrained == 'resnet50':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100

        weights_path = "./resNet50_miniimagenet_Nointerfer_full_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet18':
        model = models.resnet18(num_classes=100).to(device)  # num_classes=100

        weights_path = "./resNet18_miniimagenet_Nointerfer_full_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet50_adv':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100

        weights_path = "./resNet50_miniimagenet_PGD_SGD.pth"
    elif opt.pretrained == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(num_classes=100).to(device)
        weights_path = "./mobilenet_v3_large_miniimagenet_Nointerfer_full_SGD.pth"
    elif opt.pretrained == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(num_classes=100).to(device)
        weights_path = "./mobilenet_v3_small_miniimagenet_Nointerfer_full_SGD.pth"
    elif opt.pretrained == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x0_5_miniimagenet_Nointerfer_full_SGD.pth"
    elif opt.pretrained == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x1_miniimagenet_Nointerfer_full_SGD.pth"
    elif opt.pretrained == 'shufflenet_v2_x1_5':
        # /home/lingyu/anaconda3/envs/tensorflow22/bin/python /home/lingyu/pycharm-community-2021.3.1/plugins/python-ce/helpers/pydev/pydevconsole.py --mode=client --port=35179
        model = models.shufflenet_v2_x1_5(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x1_5_miniimagenet_Nointerfer_full_SGD.pth"
    elif opt.pretrained == 'shufflenet_v2_x2_0':
        model = models.shufflenet_v2_x2_0(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x2_miniimagenet_Nointerfer_full_SGD.pth"

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))




    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    losses = []         # loss for each batch
    avg_losses = []     # smoothed losses
    best_loss = 1e18
    epochs = 1  # 150
    image_id = 0
    theta_h = 40
    theta_w = 50
    Obs_d = 25  # unit: cm h_i 1cm
    # gamma_DOR = DOR_height / DOR_width
    gamma_o = 9 / 16
    alpha_z = 2
    beta_z = 2
    for epoch_id in range(epochs):

        for batch_id, batch_img in enumerate(train_ds):
            batch_start_time = time.time()
            color_flag = 1
            red_flag = 0
            green_flag = 0
            blue_flag = 0
            location_flag = 1
            intensity_flag = 1
            iter_count = 0
            judge_test = np.zeros(1000, dtype=np.float32)

            _, img_ori_height, img_ori_width, _ = tf.shape(batch_img)

            while (location_flag):
                with tf.GradientTape() as g:





                    ## dynamic mask size
                    # display_interference_pre = crop_image(
                    #     cv2.resize(np.array(u1_RGB_real_mask_temp), None, fx=3, fy=3, interpolation=cv2.INTER_AREA),
                    #     img_ori_height, img_ori_width)  # INTER_NEAREST INTER_LINEAR INTER_AREA
                    #
                    # interfer_pixel_index = np.unique(display_interference_pre)
                    # interfer_pixel_index = interfer_pixel_index[interfer_pixel_index != 0]
                    # interfer_pixel_index = interfer_pixel_index.astype(int)
                    # interfer_pixel_trans_index = np.zeros_like(interfer_pixel_index)
                    #
                    #
                    # interfer_pixel_red_index = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    # interfer_pixel_green_index = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    # interfer_pixel_blue_index = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #
                    # for ii in range(len(interfer_pixel_index)):
                    #     if interfer_pixel_index[ii] % 3 == 2:
                    #         interfer_pixel_trans_index[ii] = 2  # Red
                    #         interfer_pixel_red_index[ii] = 0.66  # power for red pixel
                    #     elif interfer_pixel_index[ii] % 3 == 0:
                    #         interfer_pixel_trans_index[ii] = 3  # Green
                    #         interfer_pixel_green_index[ii] = 0.66  # power for red pixel
                    #     elif interfer_pixel_index[ii] % 3 == 1:
                    #         interfer_pixel_trans_index[ii] = 4  # Blue
                    #         interfer_pixel_blue_index[ii] = 0.66  # power for red pixel
                    #
                    # red_pixel_index = np.where(interfer_pixel_trans_index == 2)
                    # green_pixel_index = np.where(interfer_pixel_trans_index == 3)
                    # blue_pixel_index = np.where(interfer_pixel_trans_index == 4)
                    #
                    # Red_mask = np.zeros_like(u1_RGB_real_mask_temp)
                    # Green_mask = np.zeros_like(u1_RGB_real_mask_temp)
                    # Blue_mask = np.zeros_like(u1_RGB_real_mask_temp)
                    #
                    # for iii in range(len(red_pixel_index[0])):
                    #     Red_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[red_pixel_index[0][iii]],
                    #                         interfer_pixel_index[red_pixel_index[0][iii]], Red_mask)
                    #
                    # for jjj in range(len(green_pixel_index[0])):
                    #     Green_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[green_pixel_index[0][jjj]],
                    #                           interfer_pixel_index[green_pixel_index[0][jjj]], Green_mask)
                    #
                    # for kkk in range(len(blue_pixel_index[0])):
                    #     Blue_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[blue_pixel_index[0][kkk]],
                    #                          interfer_pixel_index[blue_pixel_index[0][kkk]], Blue_mask)

                    # baseline: randome pattern
                    np.random.seed(0)

                    # if (color_flag == 1):
                    #     interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    # alpha = 1
                    # n = len(interfer_pixel)


                    ## smartphone
                    # if (color_flag == 1):
                    #     interfer_pixel_index = np.arange(3240).reshape(1080, 3).flatten()  # np.zeros((1, 30), dtype=int)
                    #     # interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #     interfer_pixel = 50*np.ones(np.shape(interfer_pixel_index), dtype=np.float32)
                    #
                    #
                    # if tf.is_tensor(interfer_pixel):
                    #     interfer_pixel = interfer_pixel.numpy()
                    #
                    # interfer_pixel_orders = interfer_pixel.reshape(-1, 3)
                    # np.savetxt(txt_path, interfer_pixel_orders, fmt="%d", delimiter=",")
                    #
                    # while True:
                    #     if not len(client.files_search_v2(dropbox_path).matches):
                    #         with open(txt_path, 'rb') as f:
                    #             client.files_upload(f.read(), dropbox_path, strict_conflict=False)
                    #         break
                    #
                    #
                    # # else:
                    # #     client.files_delete_v2(dropbox_path)
                    # #     with open(txt_path, 'rb') as f:
                    # #         client.files_upload(f.read(), dropbox_path, strict_conflict=False)
                    #
                    #
                    #
                    #
                    #
                    # alpha = 1
                    # n = len(interfer_pixel)
                    #
                    #
                    #
                    # while True:
                    #     if not len(client.files_search_v2(dropbox_path).matches):
                    #         break





                    # all_interfer_pixel_ssim = all_interfer_pixel
                    # for ii in range(len(interfer_pixel_index)):
                    #     all_interfer_pixel_ssim = np.where(all_interfer_pixel_index == interfer_pixel_index[ii],
                    #                                        interfer_pixel[ii], all_interfer_pixel_ssim)
                    #
                    # all_display_red_mask_ssim = all_Red_mask  # tf.cast(Red_mask)
                    # all_display_green_mask_ssim = all_Green_mask  # tf.cast(Green_mask)
                    # all_display_blue_mask_ssim = all_Blue_mask
                    #
                    # for iii in range(len(all_red_pixel_index[0])):
                    #     all_display_red_mask_ssim = tf.where(
                    #         all_display_red_mask_ssim == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
                    #         all_interfer_pixel_ssim[all_red_pixel_index[0][iii]],
                    #         all_display_red_mask_ssim)  # mapped_pixel[iii], display_red_mask)
                    # for jjj in range(len(all_green_pixel_index[0])):
                    #     all_display_green_mask_ssim = tf.where(
                    #         all_display_green_mask_ssim == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
                    #         all_interfer_pixel_ssim[all_green_pixel_index[0][jjj]],
                    #         all_display_green_mask_ssim)  # mapped_pixel[len(red_pixel_index[0]) + jjj], display_green_mask)
                    # for kkk in range(len(all_blue_pixel_index[0])):
                    #     all_display_blue_mask_ssim = tf.where(
                    #         all_display_blue_mask_ssim == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
                    #         all_interfer_pixel_ssim[all_blue_pixel_index[0][kkk]], all_display_blue_mask_ssim)
                    #
                    # all_display_RGB_mask_ssim = np.stack(
                    #     [all_display_red_mask_ssim, all_display_green_mask_ssim, all_display_blue_mask_ssim],
                    #     axis=2)
                    #
                    # Z_sast = math.pow(math.sqrt(1 / (4 * math.tan(theta_h / 2 * math.pi / 180) * math.tan(
                    #     theta_w / 2 * math.pi / 180)) * math.pow(1 / Obs_d, 2) / gamma_DOR), (
                    #                           1 - math.pow(math.fabs(gamma_DOR - gamma_o),
                    #                                        beta_z) / alpha_z))  # (1-math.pow(math.fabs(gamma_DOR-gamma_o),beta_z)/alpha_z)
                    #
                    # all_display_RGB_mask_ssim_blur = cv2.bilateralFilter(all_display_RGB_mask_ssim, 5, 50, 50)
                    # all_display_RGB_mask_ssim_new = cv2.resize(all_display_RGB_mask_ssim_blur, None, None, fx=Z_sast,
                    #                                            fy=Z_sast,
                    #                                            interpolation=cv2.INTER_LINEAR)
                    #
                    # all_display_RGB_mask_ssim_new = tf.expand_dims(all_display_RGB_mask_ssim_new, axis=0)
                    #
                    # all_display_RGB_mask_ori_old = tf.zeros_like(all_display_RGB_mask_ssim_new)#tf.expand_dims(all_display_RGB_mask_ori_old, axis=0)
                    #
                    #
                    # ssim_value, cs_map_value = tf_ssim(all_display_RGB_mask_ori_old, all_display_RGB_mask_ssim_new,
                    #                                    cs_map=True, mean_metric=False, filter_size=11,
                    #                                    filter_sigma=1.5)
                    #
                    # # plt.imshow(ssim_value[0, :].numpy())
                    # # plt.show()
                    # #
                    # # plt.imshow(cs_map_value[0, :].numpy())
                    # # plt.show()
                    # #
                    # ssim_mean = tf.reduce_mean(ssim_value)
                    # cs_mean = tf.reduce_mean(cs_map_value)
                    #
                    # if (ssim_mean < ssim_min):
                    #     ssim_min = ssim_mean







                    ## control kernel k
                    # all_interfer_pixel = np.ones(np.shape(all_interfer_pixel_index), dtype=np.float32)
                    # all_interfer_pixel = tf.constant(all_interfer_pixel)



                    # interfer_pixel = tf.constant(interfer_pixel, dtype=tf.float32)
                    # all_interfer_pixel = tf.constant(all_interfer_pixel, dtype=tf.float32)

                    # preprocessing ***************************************************************************************************
                    # mapped_pattern = pattern / 255
                    mapped_pixel = 1
                    # mapped_all_pixel = all_interfer_pixel
                    # mapped_u1_RGB_real_mask = u1_RGB_real_mask



                    # for iii in range(len(all_interfer_pixel_index)):
                    #     mapped_u1_RGB_real_mask = tf.where(
                    #         u1_RGB_real_mask == tf.constant(all_interfer_pixel_index[iii], dtype=tf.float32),
                    #         mapped_all_pixel[iii], mapped_u1_RGB_real_mask)
                    #
                    # # plt.imshow(mapped_u1_RGB_real_mask)
                    # # plt.show()
                    # PSFs, PSFs_RGB, interfer_all_pixel, u1_RGB_real = camera(mapped_u1_RGB_real_mask, DoR_option='small',
                    #                                                          mode_option='running')
                    #
                    # interfer_img = get_interfer_img(Red_mask, Green_mask, Blue_mask, red_pixel_index, green_pixel_index,
                    #                                 blue_pixel_index, interfer_pixel_index, mapped_pixel, batch_size,
                    #                                 img_ori_height,
                    #                                 img_ori_width)

                    ## Obtain all interference pixel index
                    # Red_pixel_mask = crop_image(cv2.resize(Red_mask, None, fx=3, fy=3, interpolation=cv2.INTER_AREA),
                    #                             img_ori_height, img_ori_width)
                    # Green_pixel_mask = crop_image(
                    #     cv2.resize(Green_mask, None, fx=3, fy=3, interpolation=cv2.INTER_AREA),
                    #     img_ori_height, img_ori_width)
                    # Blue_pixel_mask = crop_image(cv2.resize(Blue_mask, None, fx=3, fy=3, interpolation=cv2.INTER_AREA),
                    #                              img_ori_height, img_ori_width)
                    # RGB_pixel_mask = np.stack([Red_pixel_mask, Green_pixel_mask, Blue_pixel_mask],
                    #                           axis=2)  ## with pixel index


                    # captured = capture_alpha(batch_img, interfer_img, PSFs, PSFs_RGB, alpha)
                    # deconved = wiener_deconv(captured, PSFs)
                    # captured_temp = captured
                    # deconved_temp = deconved


                    # Camera capture images
                    event = threading.Event()

                    top = tk.Tk()  # 导入tk模块
                    top.attributes("-fullscreen", True)
                    top.attributes("-topmost", 1)
                    width = top.winfo_screenwidth()
                    height = top.winfo_screenheight()
                    print(width, height)
                    image = Image.open(path_name[image_id])
                    image_width = image.width
                    image_height = image.height
                    if image_height < image_width:
                        size_ratio = height / image_height
                        if image_height * size_ratio < height:
                            resize_shape = (
                                math.ceil(image_width * size_ratio),
                                math.ceil(image_height * size_ratio))
                        else:
                            resize_shape = (
                                int(image_width * size_ratio), int(image_height * size_ratio))
                    else:
                        size_ratio = width / image_width
                        if image_width * size_ratio < width:
                            resize_shape = (
                                math.ceil(image_width * size_ratio),
                                math.ceil(image_height * size_ratio))
                        else:
                            resize_shape = (
                                int(image_width * size_ratio), int(image_height * size_ratio))

                    photo = ImageTk.PhotoImage(image.resize(resize_shape))
                    # photo = ImageTk.PhotoImage(image.resize((width, height)))
                    # photo = ImageTk.PhotoImage(image)
                    label = Label(top, bg = 'black')
                    label.pack(expand=YES, fill=BOTH)  # 让图像在中央填充
                    label.configure(image=photo)
                    top.bind("<Escape>", lambda event: top.destroy())


                    img_path = os.path.join(img_cap_dir, file_name[image_id])
                    # top.after(1000, ImageStreamingTest(hort, port, img_path))
                    t = threading.Thread(target=ImageStreamingTest, args=(hort, port, img_path))
                    t.start()
                    top.after(2000, top.destroy)

                    top.mainloop()

                    # top.destroy()
                    location_flag = 0



                    # batch_img_temp = random_size(batch_img, target_size=256)
                    # batch_img_temp = center_crop(batch_img_temp)
                    #
                    # interfer_img_index = random_size_numpy(RGB_pixel_mask, target_size=256)
                    # interfer_img_index = center_crop_numpy(interfer_img_index)




                    # deconved = random_size(deconved, target_size=256)
                    # deconved = center_crop(deconved)





                    # for iiii in range(batch_size):
                    #     captured_img = vis.tensor_to_img_save(captured, iiii)  # captured image in current batch
                    #     deconved_img = vis.tensor_to_img_save(deconved, iiii)






                    # img_path = cap_path ##os.path.join(img_temp_dir, 'temp.jpg')
                    #
                    # # image_pil = None
                    # # if captured_img.shape[2] == 1:
                    # #     captured_img = np.reshape(captured_img, (captured_img.shape[0], captured_img.shape[1]))
                    # #     image_pil = Image.fromarray(captured_img, 'L')
                    # # else:
                    # #     image_pil = Image.fromarray(captured_img)
                    # # image_pil.save(img_path)
                    # img = Image.open(img_path).convert('RGB')
                    # img_np = np.array(img, dtype=np.uint8)
                    # img.close()
                    #
                    # captured = random_size_numpy(img_np, target_size=256)
                    # captured = center_crop_numpy(captured)



                    # [C, H, W]
                    # img_tensor = data_transform(captured)
                    # captured = tf.expand_dims(captured, axis=0)



                    # expand batch dimension
                    # [C, H, W] -> [N, C, H, W]
                    # input_tensor = torch.unsqueeze(img_tensor, dim=0)

                    # model.eval()
                    # with torch.no_grad():
                    #     # predict class
                    #     output = torch.squeeze(model(input_tensor.to(device))).cpu()
                    #     predict = torch.softmax(output, dim=0)
                    #     predict_cla = torch.argmax(predict).numpy()
                    #
                    #     predict_sort, idx_sort = torch.sort(torch.Tensor(predict), dim=0, descending=True)
                    #
                    # if (color_flag == 1):
                    #
                    #     # 0 1 31 4 29 original
                    #     # 31 0 4 29 57
                    #     target_category_ori = idx_sort[0].numpy().tolist()  # tabby, tabby cat
                    #     target_category_ori_all = idx_sort.numpy().tolist()
                    #     target_category_prob = predict[predict_cla].numpy()
                    #     target_category_prob_all = predict.numpy()
                    #
                    #     grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category_ori)
                    #     grayscale_cam = grayscale_cam[0, :]
                    #
                    #     # all_redopt_pixel_index = np.unique(interfer_img_index[:, :, 0])
                    #     # all_redopt_pixel_index = all_redopt_pixel_index[all_redopt_pixel_index != 0]
                    #     # all_redopt_pixel_index = all_redopt_pixel_index.astype(int)
                    #     #
                    #     # all_greenopt_pixel_index = np.unique(interfer_img_index[:, :, 1])
                    #     # all_greenopt_pixel_index = all_greenopt_pixel_index[all_greenopt_pixel_index != 0]
                    #     # all_greenopt_pixel_index = all_greenopt_pixel_index.astype(int)
                    #     #
                    #     # all_blueopt_pixel_index = np.unique(interfer_img_index[:, :, 2])
                    #     # all_blueopt_pixel_index = all_blueopt_pixel_index[all_blueopt_pixel_index != 0]
                    #     # all_blueopt_pixel_index = all_blueopt_pixel_index.astype(int)
                    #
                    #     grayscale_cam_mean = 1 * np.mean(grayscale_cam) # 1.5
                    #     grayscale_cam_mask = np.where(grayscale_cam <= grayscale_cam_mean, 0, 1)
                    #
                    #     # visualization = show_cam_on_image(captured_img.astype(dtype=np.float32) / 255.,
                    #     #                                   grayscale_cam,
                    #     #                                   use_rgb=True)
                    #     #
                    #     # plt.imshow(visualization)
                    #     # plt.show()
                    #     #
                    #     # plt.imshow(grayscale_cam_mask)
                    #     # plt.show()
                    #
                    #     # redopt_pixel_mask = interfer_img_index[:, :, 0] * grayscale_cam_mask
                    #     # greenopt_pixel_mask = interfer_img_index[:, :, 1] * grayscale_cam_mask
                    #     # blueopt_pixel_mask = interfer_img_index[:, :, 2] * grayscale_cam_mask
                    #
                    #     if captured.shape[2] == 1:
                    #         captured_test = np.reshape(captured, (captured.shape[0], captured.shape[1]))
                    #         captured_test = captured_test[0, :, :, :].numpy()
                    #         captured_test = (captured_test - np.amin(captured_test)) / (
                    #                 np.amax(captured_test) - np.amin(captured_test))
                    #     else:
                    #         captured_test = captured
                    #         captured_test = captured_test[0, :, :, :].numpy()
                    #         captured_test = (captured_test - np.amin(captured_test)) / (
                    #                 np.amax(captured_test) - np.amin(captured_test))
                    #
                    #     red_captured_mask = captured_test[:, :, 0] * grayscale_cam_mask
                    #     green_captured_mask = captured_test[:, :, 1] * grayscale_cam_mask
                    #     blue_captured_mask = captured_test[:, :, 2] * grayscale_cam_mask
                    #
                    #     # test = redopt_pixel_mask.reshape(-1)
                    #
                    #     ## get the element with the most occurrences
                    #
                    #     # redopt_pixel_mask_count = np.bincount(interfer_img_index[:, :, 0].reshape(-1).astype(int))
                    #     # greenopt_pixel_mask_count = np.bincount(interfer_img_index[:, :, 1].reshape(-1).astype(int))
                    #     # blueopt_pixel_mask_count = np.bincount(interfer_img_index[:, :, 2].reshape(-1).astype(int))
                    #
                    #     # redopt_pixel_mask_count = np.bincount(redopt_pixel_mask.reshape(-1).astype(int))
                    #     # greenopt_pixel_mask_count = np.bincount(greenopt_pixel_mask.reshape(-1).astype(int))
                    #     # blueopt_pixel_mask_count = np.bincount(blueopt_pixel_mask.reshape(-1).astype(int))
                    #     #
                    #     # redopt_pixel_mask_count[
                    #     #     np.argmax(redopt_pixel_mask_count)] = 0  # np.min(redopt_pixel_mask_count)
                    #     # greenopt_pixel_mask_count[
                    #     #     np.argmax(greenopt_pixel_mask_count)] = 0  # np.min(greenopt_pixel_mask_count)
                    #     # blueopt_pixel_mask_count[
                    #     #     np.argmax(blueopt_pixel_mask_count)] = 0  # np.min(blueopt_pixel_mask_count)
                    #
                    #     # redopt_pixel_maxnumindex = np.argmax(redopt_pixel_mask_count)
                    #     # greenopt_pixel_maxnumindex = np.argmax(greenopt_pixel_mask_count)
                    #     # blueopt_pixel_maxnumindex = np.argmax(blueopt_pixel_mask_count)
                    #
                    #     # redopt_pixel_mask_index = np.unique(redopt_pixel_mask)
                    #     # redopt_pixel_mask_index = redopt_pixel_mask_index[redopt_pixel_mask_index != 0]
                    #     # redopt_pixel_mask_index = redopt_pixel_mask_index.astype(int)
                    #     # redpixels_len = len(redopt_pixel_mask_index)
                    #     # if redpixels_len == 0:
                    #     #     redopt_pixel_mask_index = np.unique(interfer_img_index[:, :, 0])
                    #     #     redopt_pixel_mask_index = redopt_pixel_mask_index[redopt_pixel_mask_index != 0]
                    #     #     redopt_pixel_mask_index = redopt_pixel_mask_index.astype(int)
                    #     #     redpixels_len = len(redopt_pixel_mask_index)
                    #     # redpixels_count = 0
                    #     # redpixels_prob = np.zeros(np.shape(redopt_pixel_mask_index), dtype=np.float32)
                    #     #
                    #     # greenopt_pixel_mask_index = np.unique(greenopt_pixel_mask)
                    #     # greenopt_pixel_mask_index = greenopt_pixel_mask_index[greenopt_pixel_mask_index != 0]
                    #     # greenopt_pixel_mask_index = greenopt_pixel_mask_index.astype(int)
                    #     # greenpixels_len = len(greenopt_pixel_mask_index)
                    #     # if greenpixels_len == 0:
                    #     #     greenopt_pixel_mask_index = np.unique(interfer_img_index[:, :, 1])
                    #     #     greenopt_pixel_mask_index = greenopt_pixel_mask_index[greenopt_pixel_mask_index != 0]
                    #     #     greenopt_pixel_mask_index = greenopt_pixel_mask_index.astype(int)
                    #     #     greenpixels_len = len(greenopt_pixel_mask_index)
                    #     # greenpixels_count = 0
                    #     # greenpixels_prob = np.zeros(np.shape(greenopt_pixel_mask_index), dtype=np.float32)
                    #     #
                    #     # blueopt_pixel_mask_index = np.unique(blueopt_pixel_mask)
                    #     # blueopt_pixel_mask_index = blueopt_pixel_mask_index[blueopt_pixel_mask_index != 0]
                    #     # blueopt_pixel_mask_index = blueopt_pixel_mask_index.astype(int)
                    #     # bluepixels_len = len(blueopt_pixel_mask_index)
                    #     # if bluepixels_len == 0:
                    #     #     blueopt_pixel_mask_index = np.unique(interfer_img_index[:, :, 2])
                    #     #     blueopt_pixel_mask_index = blueopt_pixel_mask_index[blueopt_pixel_mask_index != 0]
                    #     #     blueopt_pixel_mask_index = blueopt_pixel_mask_index.astype(int)
                    #     #     bluepixels_len = len(blueopt_pixel_mask_index)
                    #     # bluepixels_count = 0
                    #     # bluepixels_prob = np.zeros(np.shape(blueopt_pixel_mask_index), dtype=np.float32)
                    #
                    #
                    #     ## Smartphone
                    #
                    #     redopt_pixel_mask_index = np.arange(0, 27, 3)
                    #     greenopt_pixel_mask_index = np.arange(1, 28, 3)
                    #     blueopt_pixel_mask_index = np.arange(2, 29, 3)
                    #
                    #     redpixels_len = len(redopt_pixel_mask_index)
                    #     greenpixels_len = len(greenopt_pixel_mask_index)
                    #     bluepixels_len = len(blueopt_pixel_mask_index)
                    #
                    #     redpixels_count = 0
                    #     redpixels_prob = np.zeros(np.shape(redopt_pixel_mask_index), dtype=np.float32)
                    #
                    #     greenpixels_count = 0
                    #     greenpixels_prob = np.zeros(np.shape(greenopt_pixel_mask_index), dtype=np.float32)
                    #
                    #     bluepixels_count = 0
                    #     bluepixels_prob = np.zeros(np.shape(blueopt_pixel_mask_index), dtype=np.float32)
                    #
                    #     ## Color effect pick one pixel to test color effect
                    #
                    #     # redopt_pixel_pick = redopt_pixel_maxnumindex  # redopt_pixel_mask_index[int(len(redopt_pixel_mask_index) / 2)]
                    #     # greenopt_pixel_pick = greenopt_pixel_maxnumindex  # greenopt_pixel_mask_index[int(len(greenopt_pixel_mask_index) / 2)]
                    #     # blueopt_pixel_pick = blueopt_pixel_maxnumindex  # blueopt_pixel_mask_index[int(len(blueopt_pixel_mask_index) / 2)]
                    #
                    #     # redopt_pixel_pick_mask = np.where(interfer_img_index[:, :, 0] == redopt_pixel_pick, 1, 0)
                    #     # greenopt_pixel_pick_mask = np.where(interfer_img_index[:, :, 1] == greenopt_pixel_pick, 1, 0)
                    #     # blueopt_pixel_pick_mask = np.where(interfer_img_index[:, :, 2] == blueopt_pixel_pick, 1, 0)
                    #
                    #     # redopt_pixel_pick_mask = np.where(redopt_pixel_mask == redopt_pixel_pick, 1, 0)
                    #     # greenopt_pixel_pick_mask = np.where(greenopt_pixel_mask == greenopt_pixel_pick, 1, 0)
                    #     # blueopt_pixel_pick_mask = np.where(blueopt_pixel_mask == blueopt_pixel_pick, 1, 0)
                    #
                    #     # redopt_pixel_effect = np.sum(redopt_pixel_pick_mask * grayscale_cam)
                    #     # greenopt_pixel_effect = np.sum(greenopt_pixel_pick_mask * grayscale_cam)
                    #     # blueopt_pixel_effect = np.sum(blueopt_pixel_pick_mask * grayscale_cam)
                    #
                    #     redopt_pixel_effect_v1 = np.sum(red_captured_mask)
                    #     greenopt_pixel_effect_v1 = np.sum(green_captured_mask)
                    #     blueopt_pixel_effect_v1 = np.sum(blue_captured_mask)
                    #
                    #     if (
                    #             redopt_pixel_effect_v1 > greenopt_pixel_effect_v1 and redopt_pixel_effect_v1 > blueopt_pixel_effect_v1):
                    #         green_flag = 1
                    #         blue_flag = 2
                    #
                    #         if greenpixels_len == 0:
                    #             green_flag = 0
                    #         if bluepixels_len == 0:
                    #             blue_flag = 0
                    #         if greenpixels_len == 0 and bluepixels_len != 0:
                    #             blue_flag = 1
                    #         if greenpixels_len == 0 and bluepixels_len == 0:
                    #             if redpixels_len != 0:
                    #                 red_flag = 1
                    #             else:
                    #                 location_flag = 0
                    #
                    #
                    #     elif (
                    #             blueopt_pixel_effect_v1 > greenopt_pixel_effect_v1 and blueopt_pixel_effect_v1 > redopt_pixel_effect_v1):
                    #         green_flag = 1
                    #         red_flag = 2
                    #         if greenpixels_len == 0:
                    #             green_flag = 0
                    #         if redpixels_len == 0:
                    #             red_flag = 0
                    #         if greenpixels_len == 0 and redpixels_len != 0:
                    #             red_flag = 1
                    #         if greenpixels_len == 0 and redpixels_len == 0:
                    #             if bluepixels_len != 0:
                    #                 blue_flag = 1
                    #             else:
                    #                 location_flag = 0
                    #
                    #
                    #
                    #     elif (
                    #             greenopt_pixel_effect_v1 > redopt_pixel_effect_v1 and greenopt_pixel_effect_v1 > blueopt_pixel_effect_v1):
                    #         blue_flag = 1
                    #         red_flag = 2
                    #         if bluepixels_len == 0:
                    #             blue_flag = 0
                    #         if redpixels_len == 0:
                    #             red_flag = 0
                    #         if bluepixels_len == 0 and redpixels_len != 0:
                    #             red_flag = 1
                    #         if bluepixels_len == 0 and redpixels_len == 0:
                    #             if greenpixels_len != 0:
                    #                 green_flag = 1
                    #             else:
                    #                 location_flag = 0
                    #
                    #     else:
                    #         location_flag = 0
                    #
                    #         # red_flag = 0
                    #         # green_flag = 0
                    #         # blue_flag = 0
                    #
                    #     color_flag = 0
                    #
                    # else:
                    #     ## Perturbation pixel location search
                    #     target_category = idx_sort[0].numpy().tolist()  # tabby, tabby cat
                    #     ini_intensity = opt.itini_intensity
                    #     step_size = opt.itstep_size
                    #     max_brightness = opt.maxperturbation_power
                    #     judge_test = np.max(np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                    #     iter_count += 1
                    #     ## criteria 1
                    #     # judge_crteria = predict[predict_cla].numpy() - target_category_prob
                    #     ## criteria 2
                    #     judge_crteria = -np.max(np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                    #     if (target_category == target_category_ori): #target_category == target_category_ori judge_test[iter_count-1] < 100 judge_test < 50
                    #         if (green_flag == 1):
                    #             if (greenpixels_count <= greenpixels_len):
                    #                 if greenpixels_count < greenpixels_len:
                    #                     interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                     interfer_pixel[
                    #                         interfer_pixel_index == greenopt_pixel_mask_index[
                    #                             greenpixels_count]] = ini_intensity
                    #                 if greenpixels_count > 0:
                    #                     greenpixels_prob[greenpixels_count-1] = judge_crteria
                    #
                    #                 greenpixels_count += 1
                    #
                    #             else:
                    #                 if (blue_flag == 2):
                    #                     if (intensity_flag == 1):
                    #                         interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                         interfer_pixel[
                    #                             interfer_pixel_index == greenopt_pixel_mask_index[
                    #                                 np.argmin(greenpixels_prob)]] = ini_intensity
                    #                         interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                             np.argmin(np.absolute(
                    #                                 blueopt_pixel_mask_index - greenopt_pixel_mask_index[
                    #                                     np.argmin(greenpixels_prob)]))]] = ini_intensity
                    #                         intensity_flag = 0
                    #                     else:
                    #                         interfer_pixel = interfer_pixel.numpy()
                    #                         interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                    #                             np.argmin(greenpixels_prob)]] += step_size
                    #                         # interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                         #     np.argmin(bluepixels_prob)]] += step_size
                    #                         interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                             np.argmin(np.absolute(
                    #                                 blueopt_pixel_mask_index - greenopt_pixel_mask_index[
                    #                                     np.argmin(greenpixels_prob)]))]] += step_size
                    #
                    #                         # interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                         #     np.argmin(bluepixels_prob)]]
                    #                         # interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                         #     np.argmin(np.absolute(
                    #                         #         blueopt_pixel_mask_index - greenopt_pixel_mask_index[
                    #                         #             np.argmin(greenpixels_prob)]))]]
                    #                         if interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                    #                             np.argmin(greenpixels_prob)]] > (max_brightness + step_size) and interfer_pixel[
                    #                             interfer_pixel_index == blueopt_pixel_mask_index[
                    #                                 np.argmin(np.absolute(
                    #                                     blueopt_pixel_mask_index - greenopt_pixel_mask_index[
                    #                                         np.argmin(greenpixels_prob)]))]] > (max_brightness + step_size):
                    #                             location_flag = 0
                    #                 elif (red_flag == 2):
                    #                     if (intensity_flag == 1):
                    #                         interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                         interfer_pixel[
                    #                             interfer_pixel_index == greenopt_pixel_mask_index[
                    #                                 np.argmin(greenpixels_prob)]] = ini_intensity
                    #                         interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                    #                             np.argmin(np.absolute(
                    #                                 redopt_pixel_mask_index - greenopt_pixel_mask_index[
                    #                                     np.argmin(greenpixels_prob)]))]] = ini_intensity
                    #                         intensity_flag = 0
                    #                     else:
                    #                         interfer_pixel = interfer_pixel.numpy()
                    #                         interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                    #                             np.argmin(greenpixels_prob)]] += step_size
                    #                         interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                    #                             np.argmin(np.absolute(
                    #                                 redopt_pixel_mask_index - greenopt_pixel_mask_index[
                    #                                     np.argmin(greenpixels_prob)]))]] += step_size
                    #                         if interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                    #                             np.argmin(greenpixels_prob)]] > (max_brightness + step_size) and interfer_pixel[
                    #                             interfer_pixel_index == redopt_pixel_mask_index[
                    #                                 np.argmin(np.absolute(
                    #                                     redopt_pixel_mask_index - greenopt_pixel_mask_index[
                    #                                         np.argmin(greenpixels_prob)]))]] > (max_brightness + step_size):
                    #                             location_flag = 0
                    #                 else:
                    #                     if (intensity_flag == 1):
                    #                         interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                         interfer_pixel[
                    #                             interfer_pixel_index == greenopt_pixel_mask_index[
                    #                                 np.argmin(greenpixels_prob)]] = ini_intensity
                    #                         intensity_flag = 0
                    #                     else:
                    #                         interfer_pixel = interfer_pixel.numpy()
                    #                         interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                    #                             np.argmin(greenpixels_prob)]] += step_size
                    #                         if interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                    #                             np.argmin(greenpixels_prob)]] > (max_brightness + step_size):
                    #                             location_flag = 0
                    #
                    #
                    #
                    #         elif (blue_flag == 1):
                    #             if (bluepixels_count <= bluepixels_len):
                    #                 # interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                 # interfer_pixel[
                    #                 #     interfer_pixel_index == blueopt_pixel_mask_index[bluepixels_count]] = ini_intensity
                    #                 # bluepixels_prob[bluepixels_count] = judge_crteria
                    #                 if bluepixels_count < bluepixels_len:
                    #                     interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                     interfer_pixel[
                    #                         interfer_pixel_index == blueopt_pixel_mask_index[
                    #                             bluepixels_count]] = ini_intensity
                    #                 if bluepixels_count > 0:
                    #                     bluepixels_prob[bluepixels_count-1] = judge_crteria
                    #                 bluepixels_count += 1
                    #             else:
                    #                 if (red_flag == 2):
                    #                     if (intensity_flag == 1):
                    #                         interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                         interfer_pixel[
                    #                             interfer_pixel_index == blueopt_pixel_mask_index[
                    #                                 np.argmin(bluepixels_prob)]] = ini_intensity
                    #                         interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                    #                             np.argmin(np.absolute(
                    #                                 redopt_pixel_mask_index - blueopt_pixel_mask_index[
                    #                                     np.argmin(bluepixels_prob)]))]] = ini_intensity
                    #                         intensity_flag = 0
                    #                     else:
                    #                         interfer_pixel = interfer_pixel.numpy()
                    #                         interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                             np.argmin(bluepixels_prob)]] += step_size
                    #                         interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                    #                             np.argmin(np.absolute(
                    #                                 redopt_pixel_mask_index - blueopt_pixel_mask_index[
                    #                                     np.argmin(bluepixels_prob)]))]] += step_size
                    #                         if interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                             np.argmin(bluepixels_prob)]] > (max_brightness + step_size) and interfer_pixel[
                    #                             interfer_pixel_index == redopt_pixel_mask_index[
                    #                                 np.argmin(np.absolute(
                    #                                     redopt_pixel_mask_index - blueopt_pixel_mask_index[
                    #                                         np.argmin(bluepixels_prob)]))]] > (max_brightness + step_size):
                    #                             location_flag = 0
                    #                 else:
                    #                     if (intensity_flag == 1):
                    #                         interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                         interfer_pixel[
                    #                             interfer_pixel_index == blueopt_pixel_mask_index[
                    #                                 np.argmin(bluepixels_prob)]] = ini_intensity
                    #                         intensity_flag = 0
                    #                     else:
                    #                         interfer_pixel = interfer_pixel.numpy()
                    #                         interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                             np.argmin(bluepixels_prob)]] += step_size
                    #                         if interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                    #                             np.argmin(bluepixels_prob)]] > (max_brightness + step_size):
                    #                             location_flag = 0
                    #
                    #         elif (red_flag == 1):
                    #             if (redpixels_count < redpixels_len):
                    #                 if redpixels_count < redpixels_len:
                    #                     interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                     interfer_pixel[
                    #                         interfer_pixel_index == redopt_pixel_mask_index[
                    #                             redpixels_count]] = ini_intensity
                    #                 if redpixels_count > 0:
                    #                     redpixels_prob[redpixels_count - 1] = judge_crteria
                    #                 redpixels_count += 1
                    #             else:
                    #                 if (intensity_flag == 1):
                    #                     interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    #                     interfer_pixel[
                    #                         interfer_pixel_index == redopt_pixel_mask_index[
                    #                             np.argmin(redpixels_prob)]] = ini_intensity
                    #                     intensity_flag = 0
                    #                 else:
                    #                     interfer_pixel = interfer_pixel.numpy()
                    #                     interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                    #                         np.argmin(redpixels_prob)]] += step_size
                    #                     if interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                    #                         np.argmin(redpixels_prob)]] > (max_brightness + step_size):
                    #                         location_flag = 0
                    #
                    #     else:
                    #         location_flag = 0


                    # plt.imshow(redopt_pixel_pick_mask)
                    # plt.show()
                    # plt.imshow(greenopt_pixel_pick_mask)
                    # plt.show()
                    # plt.imshow(blueopt_pixel_pick_mask)
                    # plt.show()



                    # compute losses
                    # loss, transfer_funcs = criterion(mapped_pattern, PSFs, PSFs_RGB, deconved, batch_img, epoch=epoch_id)
                    # loss, _ = criterion(mapped_pattern, PSFs, PSFs_RGB, ssim_mean, deconved, batch_img_temp, epoch=epoch_id)

                    # losses.append(loss['total_loss'].numpy())
                    # avg_losses.append(np.mean(losses[-10:]))  # average losses from the latest 10 epochs





            # gradients = g.gradient(loss['total_loss'], vars)
            # optimizer.apply_gradients(zip(gradients, vars))

            # visualization and log
            # % 50
            # if batch_id % 1 == 0:
                # visualize images
            # visuals = {}
            # # visuals['pattern'] = (255*mapped_pattern[:,:,None]).numpy().astype(np.uint8)  # pixel opening pattern
            #
            # # visuals['display_all_pattern'] = (255 * interfer_all_pixel).numpy().astype(np.uint8)  # pixel opening pattern
            # # visuals['display_interfer_pattern'] = (255 * interfer_img[0,:,:,:]).numpy().astype(np.uint8)  # pixel opening pattern
            # # visuals['PSFs_RGB'] = vis.tensor_to_img(tf.math.log(PSFs_RGB / tf.reduce_max(PSFs_RGB)))  # PSF in log-scale
            # # visuals['original_0'] = vis.tensor_to_img(batch_img)  # captured image in current batch
            # visuals['captured_0'] = vis.tensor_to_img(captured)  # captured image in current batch
            # visuals['deconved_0'] = vis.tensor_to_img(deconved)  # deblurred image in current batch
            # vis.display_current_results(visuals, image_id)

            # visuals = {}
            # iiii = 0
            # visuals['captured_0'] = vis.tensor_to_img_save(captured, iiii)  # captured image in current batch
            # visuals['deconved_0'] = vis.tensor_to_img_save(deconved, iiii)  # deblurred image in current batch
            # vis.display_current_results(visuals, image_id, file_name)
            # image_id = image_id + 1

            # for iiii in range(batch_size):
            #
            #     visuals = {}
            #     # visuals['captured_0'] = vis.tensor_to_img_save(captured, iiii)  # captured image in current batch
            #     # visuals['deconved_0'] = vis.tensor_to_img_save(deconved, iiii)  # deblurred image in current batch
            #     # visuals['captured_0'] = captured_img  # captured image in current batch
            #     # visuals['deconved_0'] = deconved_img  # deblurred image in current batch
            #     if opt.save_mode == 'all':
            #         visuals['captured_0_all'] = vis.tensor_to_img_save(captured_temp, iiii)  # Save original size image
            #         visuals['deconved_0_all'] = vis.tensor_to_img_save(deconved_temp, iiii)  # Save original size image
            #     elif opt.save_mode == 'crop':
            #         visuals['captured_0_crop'] = captured_img  # Save cropped size image
            #         visuals['deconved_0_crop'] = deconved_img  # Save cropped size image
            #     elif opt.save_mode == 'both':
            #         visuals['captured_0_all'] = vis.tensor_to_img_save(captured_temp, iiii)  # Save original size image
            #         visuals['deconved_0_all'] = vis.tensor_to_img_save(deconved_temp, iiii)  # Save original size image
            #         visuals['captured_0_crop'] = captured_img  # Save cropped size image
            #         visuals['deconved_0_crop'] = deconved_img  # Save cropped size image
            #
            #     vis.display_current_results(visuals, image_id, file_name)
                image_id = image_id + 1



            # plot curves
            # sz = tf.shape(PSFs_RGB).numpy()[0]
            # vis.plot_current_curve(PSFs_RGB[int(sz / 2), :, :].numpy(), 'PSFs_RGB', display_id=10)  # a slice of PSF (ideally a Dirac delta function)
            # vis.plot_current_curve(transfer_funcs[int(sz/2), :, :].numpy(), 'Transfer function', display_id=15)
            #                                                                      # a slice of transfer functions (ideally all-ones)
            # vis.plot_current_curve(avg_losses, 'Total loss', display_id=9)   # losses

            # print losses to log file
            # vis.print_current_loss(img_number, image_id, loss, logfile)
            # print('Duration:{:.2f}'.format(time.time() - batch_start_time))




            # if loss['total_loss'] < best_loss:
            #     best_loss = loss['total_loss']

        # save temporary results
        # % 10
        # if epoch_id % 1 == 0:

            # Only top L2
            # sio.savemat('%s/Mapped_pixel.mat' % log_dir, {'Mapped_pixel': mapped_pixel.numpy()})
            # sio.savemat('%s/Mapped_all_pixel.mat' % log_dir, {'Mapped_all_pixel': mapped_all_pixel.numpy()})
            # sio.savemat('%s/PSFs_RGB.mat' % log_dir, {'PSFs_RGB': PSFs_RGB.numpy()})
            # sio.savemat('%s/avg_losses.mat' % log_dir, {'avg_losses': avg_losses})
            # cv2.imwrite('%s/display_all_pattern.png' % log_dir, (255 * interfer_all_pixel).numpy())  # both ok for 2D and 3D
            # cv2.imwrite('%s/display_interfer_pattern.png' % log_dir, (255 * interfer_img[0,:,:,:]).numpy())# both ok for 2D and 3D

            # Only inverse
            # sio.savemat('%s/Mapped_pixel_onlyinverse.mat' % log_dir, {'Mapped_pixel': mapped_pixel.numpy()})
            # sio.savemat('%s/Mapped_all_pixel_onlyinverse.mat' % log_dir, {'Mapped_all_pixel': mapped_all_pixel.numpy()})
            # sio.savemat('%s/PSFs_RGB_onlyinverse.mat' % log_dir, {'PSFs_RGB': PSFs_RGB.numpy()})
            # sio.savemat('%s/avg_losses_onlyinverse.mat' % log_dir, {'avg_losses': avg_losses})
            # cv2.imwrite('%s/display_all_pattern_onlyinverse.png' % log_dir, (255 * interfer_all_pixel).numpy())  # both ok for 2D and 3D
            # cv2.imwrite('%s/display_interfer_pattern_onlyinverse.png' % log_dir, (255 * interfer_img[0, :, :, :]).numpy())  # both ok for 2D and 3D

            # Only inverse
            # sio.savemat('%s/Mapped_pixel_imageinverse.mat' % log_dir, {'Mapped_pixel': mapped_pixel.numpy()})
            # sio.savemat('%s/Mapped_all_pixel_imageinverse.mat' % log_dir, {'Mapped_all_pixel': mapped_all_pixel.numpy()})
            # sio.savemat('%s/PSFs_RGB_imageinverse.mat' % log_dir, {'PSFs_RGB': PSFs_RGB.numpy()})
            # sio.savemat('%s/avg_losses_imageinverse.mat' % log_dir, {'avg_losses': avg_losses})
            # cv2.imwrite('%s/display_all_pattern_imageinverse.png' % log_dir, (255 * interfer_all_pixel).numpy())  # both ok for 2D and 3D
            # cv2.imwrite('%s/display_interfer_pattern_imageinverse.png' % log_dir, (255 * interfer_img[0, :, :, :]).numpy())  # both ok for 2D and 3D

    logfile.close()

    return mapped_pixel


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    # optimization options
    parser.add_argument('--tile_option', type=str, default='repeat', help='pixel tiling methods [repeat|randomRot]')
    # parser.add_argument('--use_data', action='store_true', help='use data-driven loss top-10 L2')
    # parser.add_argument('--invertible', action='store_true', help='use PSF-induced loss L_inv')

    parser.add_argument('--area', type=float, default=0.20, help='target pixel opening ratio 0~1')
    parser.add_argument('--area_gamma', type=float, default=10, help='area constraint weight')
    parser.add_argument('--l2_gamma', type=float, default=0.5, help='top-10 L2 loss weight') # 10
    parser.add_argument('--inv_gamma', type=float, default=0.5, help='L_inv loss weight') # 0.01

    parser.add_argument('--log_dir', type=str, default='log/', help='save optimized pattern and training log')
    parser.add_argument('--isTrain', action='store_true', help='train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate') # 1

    # display options
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_port', type=int, default=8999, help='visdom port of the web display') #8999 8097
    parser.add_argument('--display_env', type=str, default='VIS_NAME', help='visdom environment of the web display') # main VIS_NAME
    parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                             help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--checkpoints_dir', type=str, default='logs/')

    # optimization parameters
    parser.add_argument('--save_cap_dir', type=str, default='cap_XGAZE_all_MIX4_displayoff',
                        help='save perturbed images')  # cap_allcolorbar_p10_v1
    parser.add_argument('--save_dec_dir', type=str, default='dec_XGAZE_all_MIX4_displayoff',
                        help='save deblurred images')  # dec_allcolorbar_p10_v1
    parser.add_argument('--save_temp_dir', type=str, default='XGAZE_all_MIX4_displayoff_temp',
                        help='save processing temp images')
    parser.add_argument('--images_mode', type=str, default='images', help='images mode: images/images_test')
    parser.add_argument('--statusbarcolor_flag', type=int, default=3, help='statusbar color flag')
    parser.add_argument('--itstep_size', type=float, default=0.1, help='Iteration step size')
    parser.add_argument('--itini_intensity', type=float, default=0.5, help='Iteration initial intensity')
    parser.add_argument('--maxperturbation_power', type=float, default=10, help='max perturbation_power')
    parser.add_argument('--maxscreen_brightness', type=float, default=2, help='max screen_brightness')
    parser.add_argument('--probdiff_threshold', type=float, default=0, help='max screen_brightness')
    parser.add_argument('--pretrained', type=str, default='resnet50', help='pretrained network model')
    parser.add_argument('--save_mode', type=str, default='both', help='both, all or crop')
    # parser.add_argument('--gpu_flag', type=int, default=1, help='statusbar color flag')



    opt = parser.parse_args()
    opt.no_html = False
    opt.isTrain = True
    opt.use_data = True
    opt.invertible = True

    start = time.time()

    mapped_pixel = optimize_pattern_with_data(opt)

    end = time.time()
    print('total times:', (end-start))
    print('avg times:', (end-start) / 1000)
    print(str(end-start))
    print(opt.save_cap_dir)
    print(opt.pretrained)

    # optimize_pattern_with_data(opt)

    # python optimize_display.py --tile_option repeat --area_gamma 10 --l2_gamma 10 --inv_gamma 0.01 --display_env VIS_NAME
