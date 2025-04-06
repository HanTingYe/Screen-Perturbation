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

import time
import dropbox




device_gradcam = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hort, port = "192.168.0.126", 8100




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

def get_monitor_screen(x, y):
    monitors = screeninfo.get_monitors()
    for m in reversed(monitors):
        if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
            return m
    return monitors[0]

def load_data():


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

                    np.random.seed(0)
                    mapped_pixel = 1

                    # Camera capture images
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

                    label = Label(top, bg = 'black')
                    label.pack(expand=YES, fill=BOTH)  
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






 
  





                image_id = image_id + 1







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
    parser.add_argument('--save_cap_dir', type=str, default='cap_XGAZE_all_displayred5_AXON30_v3',
                        help='save perturbed images')  # cap_allcolorbar_p10_v1
    parser.add_argument('--save_dec_dir', type=str, default='dec_XGAZE_all_displayred5_AXON30_v3',
                        help='save deblurred images')  # dec_allcolorbar_p10_v1
    parser.add_argument('--save_temp_dir', type=str, default='XGAZE_all_displayred5_AXON30_v3_temp',
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
