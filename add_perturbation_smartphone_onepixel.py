import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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
import screeninfo
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

from models.inception_resnet_v1 import InceptionResnetV1
from models.mobilenet import MobileNetV1
from models.mtcnn import MTCNN, PNet, RNet, ONet, prewhiten, fixed_image_standardization
from models.utils.detect_face import extract_face
from models.utils import training

import time
import dropbox





# device_gradcam = torch.device("cpu")
device_gradcam = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hort, port = "192.168.0.126", 8100




# tf dataloader
batch_size = 1  # 12
img_height = 160
img_width = 160
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

def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    elif tf.is_tensor(img):
        return tf.shape(img).numpy().tolist()[1:3]
    else:
        return img.size

def face_crop(image,face_boxes):
    # （H,W,C）
    # margin = 0
    # image_size = 160
    # margin = [
    #     margin * (face_boxes[0][2] - face_boxes[0][0]) / (image_size - margin),
    #     margin * (face_boxes[0][3] - face_boxes[0][1]) / (image_size - margin),
    # ]
    # raw_image_size = get_size(image)
    # box = [
    #     int(max(face_boxes[0][0] - margin[0] / 2, 0)),
    #     int(max(face_boxes[0][1] - margin[1] / 2, 0)),
    #     int(min(face_boxes[0][2] + margin[0] / 2, raw_image_size[0])),
    #     int(min(face_boxes[0][3] + margin[1] / 2, raw_image_size[1])),
    # ]
    return image[:, int(face_boxes[1]):math.ceil(face_boxes[3]), int(face_boxes[0]):math.ceil(face_boxes[2]), :]

def face_crop_numpy(image,face_boxes):
    # （H,W,C）
    # margin = 0
    # image_size = 160
    # margin = [
    #     margin * (face_boxes[0][2] - face_boxes[0][0]) / (image_size - margin),
    #     margin * (face_boxes[0][3] - face_boxes[0][1]) / (image_size - margin),
    # ]
    # raw_image_size = get_size(image)
    # box = [
    #     int(max(face_boxes[0][0] - margin[0] / 2, 0)),
    #     int(max(face_boxes[0][1] - margin[1] / 2, 0)),
    #     int(min(face_boxes[0][2] + margin[0] / 2, raw_image_size[0])),
    #     int(min(face_boxes[0][3] + margin[1] / 2, raw_image_size[1])),
    # ]
    return image[int(face_boxes[1]):math.ceil(face_boxes[3]), int(face_boxes[0]):math.ceil(face_boxes[2]), :]


def center_crop_numpy(image):
    # （H,W,C）
    height, width, _ = np.shape(image)
    # input_height, input_width, _ = c.input_shape
    input_height = img_height
    input_width = img_width
    crop_x = (width - input_width) // 2
    crop_y = (height - input_height) // 2
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

def sp_crop_numpy(image):
    # （H,W,C）
    height, width, _ = np.shape(image)
    # input_height, input_width, _ = c.input_shape
    input_height = 900 # img_height
    input_width = 1400 # img_width
    crop_x = (width - input_width) // 2
    crop_y = (height - input_height) // 2
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]

def get_monitor_screen(x, y):
    monitors = screeninfo.get_monitors()
    for m in reversed(monitors):
        if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
            return m
    return monitors[0]

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

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def resize_and_rescale(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3, dct_method='INTEGER_ACCURATE')
        img = tf.cast(img, dtype=tf.float32) / 255

        return img

    train_ds = train_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    return train_ds, train_ds_name

def load_pattern():
    pattern_path = 'data/pixelPatterns/POLED_42.png'
    # pattern_path = 'data/pixelPatterns/POLED_21.png'
    pattern = cv2.imread(pattern_path,0)

    return pattern

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor




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
    img_orders_dir = os.path.join(img_dir, opt.save_orders_dir)
    mkdirs([img_dir, img_orders_dir])



    txt_path = os.path.join(img_temp_dir, 'orders.txt')  # C:\Users\yhtxy\Desktop\android_test.txt
    txt_back_path = os.path.join(img_temp_dir, 'orders_back.txt')  # C:\Users\yhtxy\Desktop\android_test.txt
    cap_path = os.path.join(img_temp_dir, 'capture.jpg')

    colorstatusbar_flag = opt.statusbarcolor_flag  # 0 red, 1 green, 2 blue, 3 white...
    ssim_min = 1
    max_sum_power = opt.maxscreen_brightness







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


    if opt.pretrained == 'inception_resnetv1_dec':
        model_gradcam = InceptionResnetV1(classify=True, pretrained='facescrub_dec', num_classes=530).to(device)
        target_layers = [model_gradcam.repeat_3]
    elif opt.pretrained == 'inception_resnetv1_cap':
        model_gradcam = InceptionResnetV1(classify=True, pretrained='facescrub_cap', num_classes=530).to(device)
        target_layers = [model_gradcam.repeat_3]
    elif opt.pretrained == 'inception_resnetv1_xgaze_robust':
        model_gradcam = InceptionResnetV1(classify=True, pretrained='xgaze_robust', num_classes=110).to(device)
        target_layers = [model_gradcam.repeat_3]
    elif opt.pretrained == 'mobilenet_dec':
        model_gradcam = MobileNetV1(classify=True, pretrained='facescrub_dec', num_classes=530).to(device)
        target_layers = [model_gradcam.stage3]
    elif opt.pretrained == 'mobilenet_cap':
        model_gradcam = MobileNetV1(classify=True, pretrained='facescrub_cap', num_classes=530).to(device)
        target_layers = [model_gradcam.stage3]
    elif opt.pretrained == 'mobilenet_xgaze_robust':
        model_gradcam = MobileNetV1(classify=True, pretrained='xgaze_robust', num_classes=110).to(device)
        target_layers = [model_gradcam.stage3]



    cam = GradCAM(model=model_gradcam, target_layers=target_layers, use_cuda=True)

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
        model = models.shufflenet_v2_x1_5(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x1_5_miniimagenet_Nointerfer_full_SGD.pth"
    elif opt.pretrained == 'shufflenet_v2_x2_0':
        model = models.shufflenet_v2_x2_0(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x2_miniimagenet_Nointerfer_full_SGD.pth"

    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location=device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    if opt.pretrained == 'inception_resnetv1_dec':
        model = InceptionResnetV1(classify=True, pretrained='facescrub_dec', num_classes=530).to(device)
    elif opt.pretrained == 'inception_resnetv1_cap':
        model = InceptionResnetV1(classify=True, pretrained='facescrub_cap', num_classes=530).to(device)
    elif opt.pretrained == 'inception_resnetv1_xgaze_robust':
        model = InceptionResnetV1(classify=True, pretrained='xgaze_robust', num_classes=110).to(device)
    elif opt.pretrained == 'mobilenet_dec':
        model = MobileNetV1(classify=True, pretrained='facescrub_dec', num_classes=530).to(device)
    elif opt.pretrained == 'mobilenet_cap':
        model = MobileNetV1(classify=True, pretrained='facescrub_cap', num_classes=530).to(device)
    elif opt.pretrained == 'mobilenet_xgaze_robust':
        model = MobileNetV1(classify=True, pretrained='xgaze_robust', num_classes=110).to(device)






    data_transform = transforms.Compose([transforms.ToTensor(), fixed_image_standardization])

    if opt.smartphone_model == 'AXON_40':
        pixel_number = 5328
    elif opt.smartphone_model == 'AXON_30':
        pixel_number = 1080
    elif opt.smartphone_model == 'MIX_4':
        pixel_number = 1080


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

            face_detect_flag = 1

            _, img_ori_height, img_ori_width, _ = tf.shape(batch_img)

            while (location_flag):
                with tf.GradientTape() as g:






                    # baseline: randome pattern
                    np.random.seed(0)


                    ## smartphone

                    if (color_flag == 1):
                        # interfer_pixel_index = np.arange(pixel_number*3).reshape(pixel_number, 3).flatten()  # np.zeros((1, 30), dtype=int)
                        # # interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                        # interfer_pixel = 0*np.ones(np.shape(interfer_pixel_index), dtype=np.float32)
                        # interfer_pixel[1400:2000] = 255
                        interfer_pixel_index = np.arange(60)
                        interfer_pixel = 0 * np.ones(np.shape(interfer_pixel_index), dtype=np.float32)


                    if tf.is_tensor(interfer_pixel):
                        interfer_pixel = interfer_pixel.numpy()





                    intefer_pixel_index_all = np.arange(pixel_number*3)
                    interfer_pixel_all = 0 * np.ones(np.shape(intefer_pixel_index_all), dtype=np.float32)
                    # interfer_pixel_copy = interfer_pixel.copy()
                    interfer_pixel_splice = []
                    for i in range(int(len(interfer_pixel)/3)):
                        for j in range(5):
                            interfer_pixel_splice = np.append(interfer_pixel_splice, interfer_pixel[3*i])
                            interfer_pixel_splice = np.append(interfer_pixel_splice, interfer_pixel[3*i + 1])
                            interfer_pixel_splice = np.append(interfer_pixel_splice, interfer_pixel[3*i + 2])

                            # interfer_pixel_splice = np.append(interfer_pixel_splice, interfer_pixel_copy[i])
                            # interfer_pixel_splice = np.append(interfer_pixel_splice, interfer_pixel_copy[i + 1])
                            # interfer_pixel_splice = np.append(interfer_pixel_splice, interfer_pixel_copy[i + 2])







                    interfer_pixel_all[1500:1800] = interfer_pixel_splice
                    interfer_pixel_all = np.where(interfer_pixel_all <= opt.maxperturbation_power, interfer_pixel_all, opt.maxperturbation_power)
                    interfer_pixel_orders = interfer_pixel_all.reshape(-1, 3)
                    np.savetxt(txt_path, interfer_pixel_orders, fmt="%d", delimiter=",")

                    phoneDir = '/sdcard/Android/data/com.example.statusbarpixelgrid/files/temp.txt'
                    # os.system('adb root remount')
                    # os.system('adb remount')
                    cmd_pull = ('adb pull %s %s' % (phoneDir, txt_back_path))
                    adb_result = os.system(cmd_pull)
                    while True:
                        if adb_result:
                            cmd_push = ('adb push %s %s' % (txt_path, phoneDir))
                            os.system(cmd_push)
                            break



                    alpha = 1
                    n = len(interfer_pixel)



                    while True:
                        if adb_result:
                            break












                    interfer_pixel = tf.constant(interfer_pixel, dtype=tf.float32)
                    # all_interfer_pixel = tf.constant(all_interfer_pixel, dtype=tf.float32)

                    # preprocessing ***************************************************************************************************
                    mapped_pixel = interfer_pixel



                    time.sleep(1)
                    event = threading.Event()

                    top = tk.Tk()  
                    top.attributes("-fullscreen", True)
                    top.attributes("-topmost", 1)
                    current_screen = get_monitor_screen(top.winfo_x(), top.winfo_y())
                    width = current_screen.width
                    height = current_screen.height
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
                    label = Label(top, bg='black')
                    label.pack(expand=YES, fill=BOTH)  
                    label.configure(image=photo)
                    top.bind("<Escape>", lambda event: top.destroy())

                    t = threading.Thread(target=ImageStreamingTest, args=(hort, port, cap_path))
                    t.start()
                    top.after(1000, top.destroy)

                    top.mainloop()









                    img_path = cap_path ##os.path.join(img_temp_dir, 'temp.jpg')


                    img = Image.open(img_path).convert('RGB')


                    if face_detect_flag == 1:
                        face_boxes, face_probs = mtcnn.detect(img)

                        if face_boxes is None: # face_boxes.size == 0:
                            location_flag = 0

                            img_np = np.array(img, dtype=np.uint8)
                            captured_temp = img_np
                            img.close()

                            img_np = sp_crop_numpy(img_np)
                            captured = random_size_numpy(img_np, target_size=160)
                            captured = center_crop_numpy(captured)
                            captured = tf.expand_dims(captured, axis=0)
                            captured_img = captured

                            # captured = random_size(captured, target_size=160)
                            # captured = center_crop(captured)
                        else:
                            margin = 0
                            image_size = 160
                            margin = [
                                margin * (face_boxes[0][2] - face_boxes[0][0]) / (image_size - margin),
                                margin * (face_boxes[0][3] - face_boxes[0][1]) / (image_size - margin),
                            ]
                            raw_image_size = get_size(img)
                            face_boxes = [
                                int(max(face_boxes[0][0] - margin[0] / 2, 0)),
                                int(max(face_boxes[0][1] - margin[1] / 2, 0)),
                                int(min(face_boxes[0][2] + margin[0] / 2, raw_image_size[0])),
                                int(min(face_boxes[0][3] + margin[1] / 2, raw_image_size[1])),
                            ]

                            face_detect_flag = 0



                    if location_flag == 1:

                        img_np = np.array(img, dtype=np.uint8)
                        captured_temp = img_np
                        img_np = sp_crop_numpy(img_np)
                        img.close()


                        if face_boxes is not None: # if len(face_boxes) != 0:

                            captured = face_crop_numpy(img_np, face_boxes)



                        captured = random_size_numpy(captured, target_size=160)
                        captured = center_crop_numpy(captured)

                        # [C, H, W]
                        img_tensor = data_transform(captured)
                        captured = tf.expand_dims(captured, axis=0)
                        captured_img = captured

                        # expand batch dimension
                        # [C, H, W] -> [N, C, H, W]
                        input_tensor = torch.unsqueeze(img_tensor, dim=0)

                        model.eval()
                        with torch.no_grad():
                            # predict class
                            output = torch.squeeze(model(input_tensor.to(device))).cpu()
                            predict = torch.softmax(output, dim=0)
                            predict_cla = torch.argmax(predict).numpy()

                            predict_sort, idx_sort = torch.sort(torch.Tensor(predict), dim=0, descending=True)

                        if (color_flag == 1):


                            target_category_ori = idx_sort[0].numpy().tolist()  # tabby, tabby cat
                            target_category_ori_all = idx_sort.numpy().tolist()
                            target_category_prob = predict[predict_cla].numpy()
                            target_category_prob_all = predict.numpy()

                            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category_ori)
                            grayscale_cam = grayscale_cam[0, :]



                            grayscale_cam_mean = 1 * np.mean(grayscale_cam)  # 1.5
                            grayscale_cam_mask = np.where(grayscale_cam <= grayscale_cam_mean, 0, 1)

                            if captured.shape[2] == 1:
                                captured_test = np.reshape(captured, (captured.shape[0], captured.shape[1]))
                                captured_test = captured_test[0, :, :, :].numpy()
                                captured_test = (captured_test - np.amin(captured_test)) / (
                                        np.amax(captured_test) - np.amin(captured_test))
                            else:
                                captured_test = captured
                                captured_test = captured_test[0, :, :, :].numpy()
                                captured_test = (captured_test - np.amin(captured_test)) / (
                                        np.amax(captured_test) - np.amin(captured_test))

                            red_captured_mask = captured_test[:, :, 0] * grayscale_cam_mask
                            green_captured_mask = captured_test[:, :, 1] * grayscale_cam_mask
                            blue_captured_mask = captured_test[:, :, 2] * grayscale_cam_mask

                            ## Smartphone

                            # redopt_pixel_mask_index = np.arange(0, pixel_number * 3 + 3 - 3, 3)
                            # greenopt_pixel_mask_index = np.arange(1, pixel_number * 3 + 3 - 2, 3)
                            # blueopt_pixel_mask_index = np.arange(2, pixel_number * 3 + 3 - 1, 3)

                            redopt_pixel_mask_index = np.arange(0, 20 * 3 + 3 - 3, 3)
                            greenopt_pixel_mask_index = np.arange(1, 20 * 3 + 3 - 2, 3)
                            blueopt_pixel_mask_index = np.arange(2, 20 * 3 + 3 - 1, 3)

                            redpixels_len = len(redopt_pixel_mask_index)
                            greenpixels_len = len(greenopt_pixel_mask_index)
                            bluepixels_len = len(blueopt_pixel_mask_index)

                            redpixels_count = 0
                            redpixels_prob = np.zeros(np.shape(redopt_pixel_mask_index), dtype=np.float32)

                            greenpixels_count = 0
                            greenpixels_prob = np.zeros(np.shape(greenopt_pixel_mask_index), dtype=np.float32)

                            bluepixels_count = 0
                            bluepixels_prob = np.zeros(np.shape(blueopt_pixel_mask_index), dtype=np.float32)

                            redopt_pixel_effect_v1 = np.sum(red_captured_mask)
                            greenopt_pixel_effect_v1 = np.sum(green_captured_mask)
                            blueopt_pixel_effect_v1 = np.sum(blue_captured_mask)

                            if (
                                    redopt_pixel_effect_v1 > greenopt_pixel_effect_v1 and redopt_pixel_effect_v1 > blueopt_pixel_effect_v1):
                                green_flag = 1
                                blue_flag = 2

                                if greenpixels_len == 0:
                                    green_flag = 0
                                if bluepixels_len == 0:
                                    blue_flag = 0
                                if greenpixels_len == 0 and bluepixels_len != 0:
                                    blue_flag = 1
                                if greenpixels_len == 0 and bluepixels_len == 0:
                                    if redpixels_len != 0:
                                        red_flag = 1
                                    else:
                                        location_flag = 0


                            elif (
                                    blueopt_pixel_effect_v1 > greenopt_pixel_effect_v1 and blueopt_pixel_effect_v1 > redopt_pixel_effect_v1):
                                green_flag = 1
                                red_flag = 2
                                if greenpixels_len == 0:
                                    green_flag = 0
                                if redpixels_len == 0:
                                    red_flag = 0
                                if greenpixels_len == 0 and redpixels_len != 0:
                                    red_flag = 1
                                if greenpixels_len == 0 and redpixels_len == 0:
                                    if bluepixels_len != 0:
                                        blue_flag = 1
                                    else:
                                        location_flag = 0



                            elif (
                                    greenopt_pixel_effect_v1 > redopt_pixel_effect_v1 and greenopt_pixel_effect_v1 > blueopt_pixel_effect_v1):
                                blue_flag = 1
                                red_flag = 2
                                if bluepixels_len == 0:
                                    blue_flag = 0
                                if redpixels_len == 0:
                                    red_flag = 0
                                if bluepixels_len == 0 and redpixels_len != 0:
                                    red_flag = 1
                                if bluepixels_len == 0 and redpixels_len == 0:
                                    if greenpixels_len != 0:
                                        green_flag = 1
                                    else:
                                        location_flag = 0

                            else:
                                location_flag = 0

                                # red_flag = 0
                                # green_flag = 0
                                # blue_flag = 0

                            color_flag = 0

                        else:
                            ## Perturbation pixel location search
                            target_category = idx_sort[0].numpy().tolist()  # tabby, tabby cat
                            ini_intensity = opt.itini_intensity
                            step_size = opt.itstep_size
                            max_brightness = opt.maxperturbation_power
                            judge_test = np.max(
                                np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                            iter_count += 1
                            judge_crteria = -np.max(
                                np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                            if (
                                    target_category == target_category_ori):  # target_category == target_category_ori judge_test[iter_count-1] < 100 judge_test < 50
                                if (green_flag == 1):
                                    if (greenpixels_count <= greenpixels_len):
                                        if greenpixels_count < greenpixels_len:
                                            interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                            interfer_pixel[
                                                interfer_pixel_index == greenopt_pixel_mask_index[
                                                    greenpixels_count]] = ini_intensity
                                        if greenpixels_count > 0:
                                            greenpixels_prob[greenpixels_count - 1] = judge_crteria

                                        greenpixels_count += 1

                                    else:
                                        if (blue_flag == 2):
                                            if (intensity_flag == 1):
                                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index),
                                                                          dtype=np.float32)
                                                interfer_pixel[
                                                    interfer_pixel_index == greenopt_pixel_mask_index[
                                                        np.argmin(greenpixels_prob)]] = ini_intensity
                                                interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                                                    np.argmin(np.absolute(
                                                        blueopt_pixel_mask_index - greenopt_pixel_mask_index[
                                                            np.argmin(greenpixels_prob)]))]] = ini_intensity
                                                intensity_flag = 0
                                            else:
                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                                                    np.argmin(greenpixels_prob)]] += step_size
                                                interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                                                    np.argmin(np.absolute(
                                                        blueopt_pixel_mask_index - greenopt_pixel_mask_index[
                                                            np.argmin(greenpixels_prob)]))]] += step_size

                                                if interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                                                    np.argmin(greenpixels_prob)]] >= (max_brightness + step_size) and \
                                                        interfer_pixel[
                                                            interfer_pixel_index == blueopt_pixel_mask_index[
                                                                np.argmin(np.absolute(
                                                                    blueopt_pixel_mask_index -
                                                                    greenopt_pixel_mask_index[
                                                                        np.argmin(greenpixels_prob)]))]] >= (
                                                        max_brightness + step_size):
                                                    location_flag = 0
                                        elif (red_flag == 2):
                                            if (intensity_flag == 1):
                                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index),
                                                                          dtype=np.float32)
                                                interfer_pixel[
                                                    interfer_pixel_index == greenopt_pixel_mask_index[
                                                        np.argmin(greenpixels_prob)]] = ini_intensity
                                                interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                                                    np.argmin(np.absolute(
                                                        redopt_pixel_mask_index - greenopt_pixel_mask_index[
                                                            np.argmin(greenpixels_prob)]))]] = ini_intensity
                                                intensity_flag = 0
                                            else:
                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                                                    np.argmin(greenpixels_prob)]] += step_size
                                                interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                                                    np.argmin(np.absolute(
                                                        redopt_pixel_mask_index - greenopt_pixel_mask_index[
                                                            np.argmin(greenpixels_prob)]))]] += step_size
                                                if interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                                                    np.argmin(greenpixels_prob)]] >= (max_brightness + step_size) and \
                                                        interfer_pixel[
                                                            interfer_pixel_index == redopt_pixel_mask_index[
                                                                np.argmin(np.absolute(
                                                                    redopt_pixel_mask_index - greenopt_pixel_mask_index[
                                                                        np.argmin(greenpixels_prob)]))]] >= (
                                                        max_brightness + step_size):
                                                    location_flag = 0
                                        else:
                                            if (intensity_flag == 1):
                                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index),
                                                                          dtype=np.float32)
                                                interfer_pixel[
                                                    interfer_pixel_index == greenopt_pixel_mask_index[
                                                        np.argmin(greenpixels_prob)]] = ini_intensity
                                                intensity_flag = 0
                                            else:
                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                                                    np.argmin(greenpixels_prob)]] += step_size
                                                if interfer_pixel[interfer_pixel_index == greenopt_pixel_mask_index[
                                                    np.argmin(greenpixels_prob)]] >= (max_brightness + step_size):
                                                    location_flag = 0



                                elif (blue_flag == 1):
                                    if (bluepixels_count <= bluepixels_len):
                                        if bluepixels_count < bluepixels_len:
                                            interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                            interfer_pixel[
                                                interfer_pixel_index == blueopt_pixel_mask_index[
                                                    bluepixels_count]] = ini_intensity
                                        if bluepixels_count > 0:
                                            bluepixels_prob[bluepixels_count - 1] = judge_crteria
                                        bluepixels_count += 1
                                    else:
                                        if (red_flag == 2):
                                            if (intensity_flag == 1):
                                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index),
                                                                          dtype=np.float32)
                                                interfer_pixel[
                                                    interfer_pixel_index == blueopt_pixel_mask_index[
                                                        np.argmin(bluepixels_prob)]] = ini_intensity
                                                interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                                                    np.argmin(np.absolute(
                                                        redopt_pixel_mask_index - blueopt_pixel_mask_index[
                                                            np.argmin(bluepixels_prob)]))]] = ini_intensity
                                                intensity_flag = 0
                                            else:
                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                                                    np.argmin(bluepixels_prob)]] += step_size
                                                interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                                                    np.argmin(np.absolute(
                                                        redopt_pixel_mask_index - blueopt_pixel_mask_index[
                                                            np.argmin(bluepixels_prob)]))]] += step_size
                                                if interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                                                    np.argmin(bluepixels_prob)]] >= (max_brightness + step_size) and \
                                                        interfer_pixel[
                                                            interfer_pixel_index == redopt_pixel_mask_index[
                                                                np.argmin(np.absolute(
                                                                    redopt_pixel_mask_index - blueopt_pixel_mask_index[
                                                                        np.argmin(bluepixels_prob)]))]] >= (
                                                        max_brightness + step_size):
                                                    location_flag = 0
                                        else:
                                            if (intensity_flag == 1):
                                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index),
                                                                          dtype=np.float32)
                                                interfer_pixel[
                                                    interfer_pixel_index == blueopt_pixel_mask_index[
                                                        np.argmin(bluepixels_prob)]] = ini_intensity
                                                intensity_flag = 0
                                            else:
                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                                                    np.argmin(bluepixels_prob)]] += step_size
                                                if interfer_pixel[interfer_pixel_index == blueopt_pixel_mask_index[
                                                    np.argmin(bluepixels_prob)]] >= (max_brightness + step_size):
                                                    location_flag = 0

                                elif (red_flag == 1):
                                    if (redpixels_count < redpixels_len):
                                        if redpixels_count < redpixels_len:
                                            interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                            interfer_pixel[
                                                interfer_pixel_index == redopt_pixel_mask_index[
                                                    redpixels_count]] = ini_intensity
                                        if redpixels_count > 0:
                                            redpixels_prob[redpixels_count - 1] = judge_crteria
                                        redpixels_count += 1
                                    else:
                                        if (intensity_flag == 1):
                                            interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                            interfer_pixel[
                                                interfer_pixel_index == redopt_pixel_mask_index[
                                                    np.argmin(redpixels_prob)]] = ini_intensity
                                            intensity_flag = 0
                                        else:
                                            interfer_pixel = interfer_pixel.numpy()
                                            interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                                                np.argmin(redpixels_prob)]] += step_size
                                            if interfer_pixel[interfer_pixel_index == redopt_pixel_mask_index[
                                                np.argmin(redpixels_prob)]] >= (max_brightness + step_size):
                                                location_flag = 0

                            else:
                                location_flag = 0






            for iiii in range(batch_size):

                visuals = {}
                if opt.save_mode == 'all':
                    visuals['captured_0_all'] = captured_temp  # Save original size image
                elif opt.save_mode == 'crop':
                    visuals['captured_0_crop'] = vis.tensor_to_img_save(captured_img, iiii)  # Save cropped size image
                elif opt.save_mode == 'both':
                    visuals['captured_0_all'] = captured_temp  # Save original size image
                    visuals['captured_0_crop'] = vis.tensor_to_img_save(captured_img, iiii)  # Save cropped size image

                vis.display_current_results(visuals, image_id, file_name)
                order_save_path = os.path.join(img_orders_dir, file_name[image_id].replace('.JPG','.txt'))
                np.savetxt(order_save_path, interfer_pixel_orders, fmt="%d", delimiter=",")
                image_id = image_id + 1




            # print losses to log file
            # vis.print_current_loss(img_number, image_id, loss, logfile)
            print('Duration:{:.2f}'.format(time.time() - batch_start_time))



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
    parser.add_argument('--save_cap_dir', type=str, default='cap_onepixel_example',
                        help='save perturbed images')  # cap_allcolorbar_p10_v1
    parser.add_argument('--save_dec_dir', type=str, default='dec_onepixel_example',
                        help='save deblurred images')  # dec_allcolorbar_p10_v1
    parser.add_argument('--save_temp_dir', type=str, default='onepixel_example_temp',
                        help='save processing temp images')
    parser.add_argument('--save_orders_dir', type=str, default='onepixel__example_orders',
                        help='save processing display pattern orders')
    parser.add_argument('--images_mode', type=str, default='images', help='images mode: images/images_test')
    parser.add_argument('--statusbarcolor_flag', type=int, default=3, help='statusbar color flag')
    parser.add_argument('--itstep_size', type=float, default=17, help='Iteration step size')
    parser.add_argument('--itini_intensity', type=float, default=102, help='Iteration initial intensity')
    parser.add_argument('--maxperturbation_power', type=float, default=255, help='max perturbation_power')
    parser.add_argument('--maxscreen_brightness', type=float, default=102, help='max screen_brightness')
    parser.add_argument('--probdiff_threshold', type=float, default=0, help='max screen_brightness')
    parser.add_argument('--pretrained', type=str, default='inception_resnetv1_xgaze_robust', help='pretrained network model mobilenet_xgaze_robust inception_resnetv1_xgaze_robust')
    parser.add_argument('--save_mode', type=str, default='both', help='both, all or crop')
    parser.add_argument('--smartphone_model', type=str, default='AXON_30', help='both, all or crop')
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
