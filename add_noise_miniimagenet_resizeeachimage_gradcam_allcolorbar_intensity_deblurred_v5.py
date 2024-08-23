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
from util.visualizer import Visualizer
from PIL import Image
from skimage.feature import blob_dog, blob_log, blob_doh
from ms_ssim import tf_ssim, tf_ms_ssim_resize

from wave_optics import Camera, set_params, capture, capture_alpha, wiener_deconv, get_interfer_img, get_wiener_loss
from loss import Loss
from utils import print_opt, crop_image

import torch
from torchvision import models
from torchvision import transforms
from GradCAMutils import GradCAM, show_cam_on_image, center_crop_img

import time


# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'


# device_gradcam = torch.device("cpu")
# tf.device('/gpu:0')
device_gradcam = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    # （batch,H,W,C）
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
    if height != 0 and width != 0:
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
    return cv2.resize(image, resize_shape, interpolation=cv2.INTER_NEAREST)  # INTER_NEAREST INTER_AREA


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


def ResziePadding(img, scale=1, fixed_side=128):
    h, w = img.shape[0], img.shape[1]
    # scale = max(w, h) / float(fixed_side)  # 获取缩放比例
    new_w, new_h = int(w * scale), int(h * scale)
    resize_img = cv2.resize(img, (new_w, new_h))  # 按比例缩放

    # # 计算需要填充的像素长度
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (
                    fixed_side - new_w) // 2 + 1, (
                                           fixed_side - new_w) // 2
    elif new_w % 2 == 0 and new_h % 2 != 0:
        top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (
                fixed_side - new_w) // 2, (fixed_side - new_w) // 2
    elif new_w % 2 == 0 and new_h % 2 == 0:
        top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (
                fixed_side - new_w) // 2, (fixed_side - new_w) // 2
    else:
        top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (
                    fixed_side - new_w) // 2 + 1, (
                                           fixed_side - new_w) // 2

    # top, bottom, left, right = (fixed_side - new_h), (fixed_side - new_h), (fixed_side - new_w), (fixed_side - new_w)
    # 填充图像
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return pad_img


def load_data():
    # base_dir = os.path.dirname(os.path.dirname(__file__))
    # # 获取当前文件目录
    # data_path = os.path.abspath(os.path.join(base_dir, 'Train/Poled/HQ/*.png', ""))
    # # 改为绝对路径
    # # 获取文件拼接后的路径
    # D: / Dropbox / TuD work / ScreenAI_Privacy_Underscreen / UPC_ICCP21_Code - main / Train / Poled / HQ / *.png

    # todo: change path to your training image directory
    if opt.images_mode == 'images_test':
        train_ds = tf.data.Dataset.list_files('miniimagenet/images_test/*.jpg', shuffle=False)  # images images_test
    else:
        train_ds = tf.data.Dataset.list_files('miniimagenet/images/*.jpg', shuffle=False)  # images images_test
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
    pattern = cv2.imread(pattern_path, 0)

    return pattern


def set_interfer_pixel(interfer_pixel, colorstatusbar_flag, interfer_pixel_red_index, interfer_pixel_green_index,
                       interfer_pixel_blue_index):
    # interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
    # 0 red 1 green 2 blue 3 white
    if colorstatusbar_flag == 0:
        interfer_pixel += interfer_pixel_red_index
    elif colorstatusbar_flag == 1:
        interfer_pixel += interfer_pixel_green_index
    elif colorstatusbar_flag == 2:
        interfer_pixel += interfer_pixel_blue_index
    elif colorstatusbar_flag == 3:
        interfer_pixel += interfer_pixel_red_index / 3
        interfer_pixel += interfer_pixel_green_index / 3
        interfer_pixel += interfer_pixel_blue_index / 3

    return interfer_pixel


def optimize_pattern_with_data(opt):
    tf.keras.backend.set_floatx('float32')

    # visualization and log
    vis = Visualizer(opt)
    log_dir = join('log', opt.display_env)  # directory that saves optimized pixel control and training log
    os.makedirs(log_dir, exist_ok=True)
    logfile = open('%s/log.txt' % log_dir, 'w')
    print_opt(logfile, opt)  # print opt parameters

    # set up pattern
    pattern = load_pattern()

    # set up camera
    cameraOpt = set_params()
    camera = Camera(pattern)

    # set up optimization
    optimizer = keras.optimizers.Adam(learning_rate=opt.lr,
                                      beta_1=0.9)  # lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    criterion = Loss(opt, cameraOpt)
    vars = []  # variables which we track gradient
    train_ds, train_ds_name = load_data()  # load training dataset

    file_name = []
    for fileindex in train_ds_name:
        if opt.images_mode == 'images_test':
            file_name.append(str(fileindex.numpy())[27:48])  # python的切片左闭右开
        else:
            file_name.append(str(fileindex.numpy())[22:43])

    # web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
    # img_dir = os.path.join(web_dir, 'images')
    # util.mkdirs([web_dir, img_dir])
    # img_cap_dir = os.path.join(img_dir, 'cap')
    # util.mkdirs([img_dir, img_cap_dir])
    # img_dec_dir = os.path.join(img_dir, 'dec')
    # util.mkdirs([img_dir, img_dec_dir])

    # initial pixel opening as all-open
    # pattern = np.ones((21, 21), dtype=np.float32)
    # pattern = tf.Variable(initial_value=(pattern * 2 - 1), trainable=True)  # later we use sigmoid to map 0 to 1.

    # vars.append(pattern)

    web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
    img_dir = os.path.join(web_dir, opt.images_mode)  # opt.images_mode 'images'
    img_temp_dir = os.path.join(img_dir, opt.save_temp_dir)
    mkdirs([img_dir, img_temp_dir])

    colorstatusbar_flag = opt.statusbarcolor_flag  # 0 red, 1 green, 2 blue, 3 white...
    ssim_min = 1
    max_sum_power = opt.maxscreen_brightness

    _, _, _, u1_RGB_real_mask = camera(None, DoR_option='small', mode_option='preprocessing')
    u1_RGB_real_mask_temp = np.array(u1_RGB_real_mask)
    # height_temp, width_temp = np.shape(u1_RGB_real_mask_temp)
    # u1_RGB_real_mask_temp = crop_image(u1_RGB_real_mask_temp, height_temp / 6, width_temp / 6)

    DOR_height, DOR_width = np.shape(u1_RGB_real_mask_temp)

    all_interfer_pixel_index = np.unique(u1_RGB_real_mask_temp)
    all_interfer_pixel_index = all_interfer_pixel_index[all_interfer_pixel_index != 0]
    all_interfer_pixel_index = all_interfer_pixel_index.astype(int)
    all_interfer_pixel_trans_index = np.zeros_like(all_interfer_pixel_index)

    all_interfer_pixel_red_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
    all_interfer_pixel_green_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
    all_interfer_pixel_blue_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)

    for ii in range(len(all_interfer_pixel_index)):
        if all_interfer_pixel_index[ii] % 3 == 2:
            all_interfer_pixel_trans_index[ii] = 2  # Red
            all_interfer_pixel_red_index[ii] = max_sum_power  # power for red pixel
        elif all_interfer_pixel_index[ii] % 3 == 0:
            all_interfer_pixel_trans_index[ii] = 3  # Green
            all_interfer_pixel_green_index[ii] = max_sum_power  # power for red pixel
        elif all_interfer_pixel_index[ii] % 3 == 1:
            all_interfer_pixel_trans_index[ii] = 4  # Blue
            all_interfer_pixel_blue_index[ii] = max_sum_power  # power for red pixel

    all_red_pixel_index = np.where(all_interfer_pixel_trans_index == 2)
    all_green_pixel_index = np.where(all_interfer_pixel_trans_index == 3)
    all_blue_pixel_index = np.where(all_interfer_pixel_trans_index == 4)

    all_Red_mask = np.zeros_like(u1_RGB_real_mask_temp)
    all_Green_mask = np.zeros_like(u1_RGB_real_mask_temp)
    all_Blue_mask = np.zeros_like(u1_RGB_real_mask_temp)

    for iii in range(len(all_red_pixel_index[0])):
        all_Red_mask = np.where(u1_RGB_real_mask_temp == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
                                all_interfer_pixel_index[all_red_pixel_index[0][iii]], all_Red_mask)

    for jjj in range(len(all_green_pixel_index[0])):
        all_Green_mask = np.where(u1_RGB_real_mask_temp == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
                                  all_interfer_pixel_index[all_green_pixel_index[0][jjj]], all_Green_mask)

    for kkk in range(len(all_blue_pixel_index[0])):
        all_Blue_mask = np.where(u1_RGB_real_mask_temp == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
                                 all_interfer_pixel_index[all_blue_pixel_index[0][kkk]], all_Blue_mask)

    # all_RGB_mask = np.stack([all_Red_mask, all_Green_mask, all_Blue_mask],
    #                         axis=2)

    all_display_red_mask_ori = all_Red_mask  # tf.cast(Red_mask)
    all_display_green_mask_ori = all_Green_mask  # tf.cast(Green_mask)
    all_display_blue_mask_ori = all_Blue_mask  # tf.cast(Blue_mask)

    all_interfer_pixel = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
    all_interfer_pixel = tf.constant(all_interfer_pixel)

    if colorstatusbar_flag == 0:
        all_interfer_pixel += all_interfer_pixel_red_index
    elif colorstatusbar_flag == 1:
        all_interfer_pixel += all_interfer_pixel_green_index
    elif colorstatusbar_flag == 2:
        all_interfer_pixel += all_interfer_pixel_blue_index
    elif colorstatusbar_flag == 3:
        all_interfer_pixel += all_interfer_pixel_red_index / 3
        all_interfer_pixel += all_interfer_pixel_green_index / 3
        all_interfer_pixel += all_interfer_pixel_blue_index / 3

    for iii in range(len(all_red_pixel_index[0])):
        all_display_red_mask_ori = tf.where(
            all_display_red_mask_ori == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
            all_interfer_pixel[all_red_pixel_index[0][iii]],
            all_display_red_mask_ori)  # mapped_pixel[iii], display_red_mask)
    for jjj in range(len(all_green_pixel_index[0])):
        all_display_green_mask_ori = tf.where(
            all_display_green_mask_ori == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
            all_interfer_pixel[all_green_pixel_index[0][jjj]],
            all_display_green_mask_ori)  # mapped_pixel[len(red_pixel_index[0]) + jjj], display_green_mask)
    for kkk in range(len(all_blue_pixel_index[0])):
        all_display_blue_mask_ori = tf.where(
            all_display_blue_mask_ori == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
            all_interfer_pixel[all_blue_pixel_index[0][kkk]], all_display_blue_mask_ori)

    all_display_RGB_mask_ori = np.stack(
        [all_display_red_mask_ori, all_display_green_mask_ori, all_display_blue_mask_ori],
        axis=2)

    # plt.imshow(all_display_RGB_mask_ori)
    # plt.show()

    ## CNN model
    # model 1 for gradcam
    if opt.pretrained == 'resnet50':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path_gradcam = "./resNet50_miniimagenet_robust.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet50_ori_capfine':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path_gradcam = "./resNet50_miniimagenet_Nointerfer_capfinetune_full_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet50_cap_orifine':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path_gradcam = "./resNet50_miniimagenet_baselinecap_originalfinetune_capzero_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet50_dec_baseline_allpixel_0_5':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path_gradcam = "./resNet50_miniimagenet_baselinedec_allpixel_0_5_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet50_cap_baseline_allpixel_1_0':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path_gradcam = "./resNet50_miniimagenet_baselinecap_allpixel_1_0_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet50_adv':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path_gradcam = "./resNet50_miniimagenet_PGD_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet50_zero':
        model_gradcam = models.resnet50(num_classes=100).to(device_gradcam)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path_gradcam = "./resNet50_miniimagenet_capzero_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'resnet18':
        model_gradcam = models.resnet18(num_classes=100).to(device_gradcam)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path_gradcam = "./resNet18_miniimagenet_robust.pth"  # resNet18_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.layer4]
    elif opt.pretrained == 'mobilenet_v3_large':
        model_gradcam = models.mobilenet_v3_large(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./mobilenet_v3_large_miniimagenet_robust.pth" # mobilenet_v3_large_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.features[-1]]
    elif opt.pretrained == 'mobilenet_v3_small':
        model_gradcam = models.mobilenet_v3_small(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./mobilenet_v3_small_miniimagenet_robust.pth" # mobilenet_v3_small_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.features[-1]]
    elif opt.pretrained == 'shufflenet_v2_x0_5':
        model_gradcam = models.shufflenet_v2_x0_5(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./shufflenetv2_x0_5_miniimagenet_robust.pth" # shufflenetv2_x0_5_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.conv5]
    elif opt.pretrained == 'shufflenet_v2_x1_0':
        model_gradcam = models.shufflenet_v2_x1_0(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./shufflenetv2_x1_0_miniimagenet_robust.pth" # shufflenetv2_x1_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.conv5]
    elif opt.pretrained == 'shufflenet_v2_x1_5':
        model_gradcam = models.shufflenet_v2_x1_5(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./shufflenetv2_x1_5_miniimagenet_robust.pth" # shufflenetv2_x1_5_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.conv5]
    elif opt.pretrained == 'shufflenet_v2_x2_0':
        model_gradcam = models.shufflenet_v2_x2_0(num_classes=100).to(device_gradcam)
        weights_path_gradcam = "./shufflenetv2_x2_0_miniimagenet_robust.pth" # shufflenetv2_x2_miniimagenet_Nointerfer_full_SGD.pth
        assert os.path.exists(weights_path_gradcam), "file: '{}' dose not exist.".format(weights_path_gradcam)
        model_gradcam.load_state_dict(torch.load(weights_path_gradcam, map_location=device_gradcam))
        target_layers = [model_gradcam.conv5]

    cam = GradCAM(model=model_gradcam, target_layers=target_layers, use_cuda=True)

    # model 2 for predict
    if opt.pretrained == 'resnet50':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path = "./resNet50_miniimagenet_robust.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet50_ori_capfine':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path = "./resNet50_miniimagenet_Nointerfer_capfinetune_full_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet50_cap_orifine':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path = "./resNet50_miniimagenet_baselinecap_originalfinetune_capzero_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet50_dec_baseline_allpixel_0_5':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path = "./resNet50_miniimagenet_baselinedec_allpixel_0_5_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet50_cap_baseline_allpixel_1_0':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path = "./resNet50_miniimagenet_baselinecap_allpixel_1_0_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet50_adv':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path = "./resNet50_miniimagenet_PGD_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet50_zero':
        model = models.resnet50(num_classes=100).to(device)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path = "./resNet50_miniimagenet_capzero_SGD.pth"  # resNet50_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'resnet18':
        model = models.resnet18(num_classes=100).to(device)  # num_classes=100
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, 100)

        # model.load_state_dict(torch.load('./resNet50_miniimagenet_Nointerfer_full_SGD.pth'))
        # target_layers = [model.layer4]

        weights_path = "./resNet18_miniimagenet_robust.pth"  # resNet18_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(num_classes=100).to(device)
        weights_path = "./mobilenet_v3_large_miniimagenet_robust.pth" # mobilenet_v3_large_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(num_classes=100).to(device)
        weights_path = "./mobilenet_v3_small_miniimagenet_robust.pth" # mobilenet_v3_small_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x0_5_miniimagenet_robust.pth" # shufflenetv2_x0_5_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x1_0_miniimagenet_robust.pth" # shufflenetv2_x1_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'shufflenet_v2_x1_5':
        model = models.shufflenet_v2_x1_5(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x1_5_miniimagenet_robust.pth" # shufflenetv2_x1_5_miniimagenet_Nointerfer_full_SGD.pth
    elif opt.pretrained == 'shufflenet_v2_x2_0':
        model = models.shufflenet_v2_x2_0(num_classes=100).to(device)
        weights_path = "./shufflenetv2_x2_0_miniimagenet_robust.pth" # shufflenetv2_x2_miniimagenet_Nointerfer_full_SGD.pth

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    losses = []  # loss for each batch
    avg_losses = []  # smoothed losses
    best_loss = 1e18
    epochs = 1  # 150
    image_id = 0
    theta_h = 40
    theta_w = 50
    Obs_d = 25  # unit: cm h_i 1cm
    gamma_DOR = DOR_height / DOR_width
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
            # judge_test = np.zeros(1000, dtype=np.float32)
            judge_test = 0
            find_color_flag = 1
            if colorstatusbar_flag == 3:
                color_red_flag = 1
                color_green_flag = 1
                color_blue_flag = 1

            prob_label_flag = 1  ## initial attack target
            diff_flag = 1

            redpixels_intensity_count = 0
            greenpixels_intensity_count = 0
            bluepixels_intensity_count = 0

            redpixels_intensity_prob = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            greenpixels_intensity_prob = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            bluepixels_intensity_prob = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            redpixels_intensity_diff = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            greenpixels_intensity_diff = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            bluepixels_intensity_diff = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)

            _, img_ori_height, img_ori_width, _ = tf.shape(batch_img)

            if img_ori_height > DOR_height * 3 or img_ori_width > DOR_width * 3:
                _, _, _, u1_RGB_real_mask = camera(None, DoR_option='large', mode_option='preprocessing')
                u1_RGB_real_mask_temp = np.array(u1_RGB_real_mask)
                # height_temp, width_temp = np.shape(u1_RGB_real_mask_temp)
                # u1_RGB_real_mask_temp = crop_image(u1_RGB_real_mask_temp, height_temp / 6, width_temp / 6)

                all_interfer_pixel_index = np.unique(u1_RGB_real_mask_temp)
                all_interfer_pixel_index = all_interfer_pixel_index[all_interfer_pixel_index != 0]
                all_interfer_pixel_index = all_interfer_pixel_index.astype(int)
                all_interfer_pixel_trans_index = np.zeros_like(all_interfer_pixel_index)

                all_interfer_pixel_red_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
                all_interfer_pixel_green_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
                all_interfer_pixel_blue_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)

                for ii in range(len(all_interfer_pixel_index)):
                    if all_interfer_pixel_index[ii] % 3 == 2:
                        all_interfer_pixel_trans_index[ii] = 2  # Red
                        all_interfer_pixel_red_index[ii] = max_sum_power  # power for red pixel
                    elif all_interfer_pixel_index[ii] % 3 == 0:
                        all_interfer_pixel_trans_index[ii] = 3  # Green
                        all_interfer_pixel_green_index[ii] = max_sum_power  # power for red pixel
                    elif all_interfer_pixel_index[ii] % 3 == 1:
                        all_interfer_pixel_trans_index[ii] = 4  # Blue
                        all_interfer_pixel_blue_index[ii] = max_sum_power  # power for red pixel

                all_red_pixel_index = np.where(all_interfer_pixel_trans_index == 2)
                all_green_pixel_index = np.where(all_interfer_pixel_trans_index == 3)
                all_blue_pixel_index = np.where(all_interfer_pixel_trans_index == 4)

                all_Red_mask = np.zeros_like(u1_RGB_real_mask_temp)
                all_Green_mask = np.zeros_like(u1_RGB_real_mask_temp)
                all_Blue_mask = np.zeros_like(u1_RGB_real_mask_temp)

                for iii in range(len(all_red_pixel_index[0])):
                    all_Red_mask = np.where(
                        u1_RGB_real_mask_temp == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
                        all_interfer_pixel_index[all_red_pixel_index[0][iii]], all_Red_mask)

                for jjj in range(len(all_green_pixel_index[0])):
                    all_Green_mask = np.where(
                        u1_RGB_real_mask_temp == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
                        all_interfer_pixel_index[all_green_pixel_index[0][jjj]], all_Green_mask)

                for kkk in range(len(all_blue_pixel_index[0])):
                    all_Blue_mask = np.where(
                        u1_RGB_real_mask_temp == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
                        all_interfer_pixel_index[all_blue_pixel_index[0][kkk]], all_Blue_mask)

                # all_RGB_mask = np.stack([all_Red_mask, all_Green_mask, all_Blue_mask],
                #                         axis=2)

                all_display_red_mask_ori = all_Red_mask  # tf.cast(Red_mask)
                all_display_green_mask_ori = all_Green_mask  # tf.cast(Green_mask)
                all_display_blue_mask_ori = all_Blue_mask  # tf.cast(Blue_mask)

                all_interfer_pixel = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
                all_interfer_pixel = tf.constant(all_interfer_pixel)

                if colorstatusbar_flag == 0:
                    all_interfer_pixel += all_interfer_pixel_red_index
                elif colorstatusbar_flag == 1:
                    all_interfer_pixel += all_interfer_pixel_green_index
                elif colorstatusbar_flag == 2:
                    all_interfer_pixel += all_interfer_pixel_blue_index
                elif colorstatusbar_flag == 3:
                    all_interfer_pixel += all_interfer_pixel_red_index / 3
                    all_interfer_pixel += all_interfer_pixel_green_index / 3
                    all_interfer_pixel += all_interfer_pixel_blue_index / 3

                for iii in range(len(all_red_pixel_index[0])):
                    all_display_red_mask_ori = tf.where(
                        all_display_red_mask_ori == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
                        all_interfer_pixel[all_red_pixel_index[0][iii]],
                        all_display_red_mask_ori)  # mapped_pixel[iii], display_red_mask)
                for jjj in range(len(all_green_pixel_index[0])):
                    all_display_green_mask_ori = tf.where(
                        all_display_green_mask_ori == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
                        all_interfer_pixel[all_green_pixel_index[0][jjj]],
                        all_display_green_mask_ori)  # mapped_pixel[len(red_pixel_index[0]) + jjj], display_green_mask)
                for kkk in range(len(all_blue_pixel_index[0])):
                    all_display_blue_mask_ori = tf.where(
                        all_display_blue_mask_ori == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
                        all_interfer_pixel[all_blue_pixel_index[0][kkk]], all_display_blue_mask_ori)

                all_display_RGB_mask_ori = np.stack(
                    [all_display_red_mask_ori, all_display_green_mask_ori, all_display_blue_mask_ori],
                    axis=2)


                itini_intensity_temp = opt.itini_intensity
                itstep_size_temp = opt.itstep_size
                opt.itstep_size = 1
                opt.itini_intensity = 1



            redpixels_intensity_prob = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            greenpixels_intensity_prob = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            bluepixels_intensity_prob = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            redpixels_intensity_diff = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            greenpixels_intensity_diff = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)
            bluepixels_intensity_diff = np.zeros((int(opt.maxperturbation_power / opt.itstep_size),), dtype=np.float32)

            while (location_flag):
                with tf.GradientTape() as g:



                    ## dynamic mask size
                    display_interference_pre = crop_image(
                        cv2.resize(np.array(u1_RGB_real_mask_temp), None, fx=3, fy=3, interpolation=cv2.INTER_AREA),
                        img_ori_height, img_ori_width)  # INTER_NEAREST INTER_LINEAR INTER_AREA

                    interfer_pixel_index = np.unique(display_interference_pre)
                    interfer_pixel_index = interfer_pixel_index[interfer_pixel_index != 0]
                    interfer_pixel_index = interfer_pixel_index.astype(int)
                    interfer_pixel_trans_index = np.zeros_like(interfer_pixel_index)

                    interfer_pixel_red_index = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    interfer_pixel_green_index = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                    interfer_pixel_blue_index = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)

                    for ii in range(len(interfer_pixel_index)):
                        if interfer_pixel_index[ii] % 3 == 2:
                            interfer_pixel_trans_index[ii] = 2  # Red
                            interfer_pixel_red_index[ii] = max_sum_power  # power for red pixel
                        elif interfer_pixel_index[ii] % 3 == 0:
                            interfer_pixel_trans_index[ii] = 3  # Green
                            interfer_pixel_green_index[ii] = max_sum_power  # power for red pixel
                        elif interfer_pixel_index[ii] % 3 == 1:
                            interfer_pixel_trans_index[ii] = 4  # Blue
                            interfer_pixel_blue_index[ii] = max_sum_power  # power for red pixel

                    red_pixel_index = np.where(interfer_pixel_trans_index == 2)
                    green_pixel_index = np.where(interfer_pixel_trans_index == 3)
                    blue_pixel_index = np.where(interfer_pixel_trans_index == 4)

                    Red_mask = np.zeros_like(u1_RGB_real_mask_temp)
                    Green_mask = np.zeros_like(u1_RGB_real_mask_temp)
                    Blue_mask = np.zeros_like(u1_RGB_real_mask_temp)

                    for iii in range(len(red_pixel_index[0])):
                        Red_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[red_pixel_index[0][iii]],
                                            interfer_pixel_index[red_pixel_index[0][iii]], Red_mask)

                    for jjj in range(len(green_pixel_index[0])):
                        Green_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[green_pixel_index[0][jjj]],
                                              interfer_pixel_index[green_pixel_index[0][jjj]], Green_mask)

                    for kkk in range(len(blue_pixel_index[0])):
                        Blue_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[blue_pixel_index[0][kkk]],
                                             interfer_pixel_index[blue_pixel_index[0][kkk]], Blue_mask)

                    # baseline: randome pattern
                    np.random.seed(0)
                    # interfer_pixel = np.random.rand(len(interfer_pixel_index))

                    # interfer_pixel = np.ones(np.shape(interfer_pixel_index), dtype=np.float32)
                    if (find_color_flag == 1):  # color_flag == 1
                        interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                        interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                                            interfer_pixel_red_index,
                                                            interfer_pixel_green_index, interfer_pixel_blue_index)
                    alpha = 1
                    n = len(interfer_pixel)
                    # interfer_pixel[:int(n/2)] = 1
                    # interfer_pixel[1] = 1 # Only one pixel

                    # interfer_pixel[int(n / 2):n] = 2

                    # One pixel denfender
                    # interfer_pixel[36] = 3 ##green
                    # interfer_pixel[37] = 3  ##blue
                    # interfer_pixel[38] = 3  ##red nouse
                    # interfer_pixel[39] = 3  ##green
                    # interfer_pixel[40] = 3  ##blue nouse
                    # interfer_pixel[35] = 3  ##red
                    # interfer_pixel[34] = 3  ##blue
                    # interfer_pixel[33] = 3  ##green nouse
                    # interfer_pixel[43] = 3  ##nothing nouse
                    # interfer_pixel[44] = 3  ##red nouse
                    # interfer_pixel[45] = 3  ##green nouse
                    # interfer_pixel[49] = 3  ##blue
                    # interfer_pixel[50] = 3  ##red nouse
                    # interfer_pixel[51] = 3  ##green

                    # interfer_pixel[33] = 3  ##green
                    # interfer_pixel[32] = 3  ##red
                    # interfer_pixel[31] = 3  ##blue
                    # 36 37 39 35 34 49 51

                    # Color denfender
                    # interfer_pixel = set_interfer_pixel(colorstatusbar_flag, interfer_pixel_red_index, interfer_pixel_green_index, interfer_pixel_blue_index)
                    #
                    # if colorstatusbar_flag == 0 and color_flag == 1:
                    #     interfer_pixel += interfer_pixel_red_index
                    # elif colorstatusbar_flag == 1 and color_flag == 1:
                    #     interfer_pixel += interfer_pixel_green_index
                    # elif colorstatusbar_flag == 2 and color_flag == 1:
                    #     interfer_pixel += interfer_pixel_blue_index
                    # elif colorstatusbar_flag == 3 and color_flag == 1:
                    #     interfer_pixel += interfer_pixel_red_index/3
                    #     interfer_pixel += interfer_pixel_green_index/3
                    #     interfer_pixel += interfer_pixel_blue_index/3

                    # interfer_pixel += interfer_pixel_red_index
                    # interfer_pixel += interfer_pixel_green_index
                    # interfer_pixel += interfer_pixel_blue_index

                    # Universal denfender
                    # interfer_pixel = np.random.rand(len(interfer_pixel_index))*1.5
                    # interfer_pixel[28] += 0.95
                    # interfer_pixel[39] += 0.9
                    # # interfer_pixel[29] += 0.8
                    # interfer_pixel[36] += 0.5

                    # interfer_pixel = tf.Variable(initial_value=interfer_pixel, trainable=False)  # later we use sigmoid to map 0 to 1

                    ## control kernel k
                    # all_interfer_pixel = np.random.rand(len(all_interfer_pixel_index))
                    # all_interfer_pixel = np.ones(np.shape(all_interfer_pixel_index), dtype=np.float32)
                    # all_interfer_pixel = tf.constant(all_interfer_pixel)

                    ## The default kernel is all 1 no longer design the kernel ###
                    # for ii in range(len(interfer_pixel_index)):
                    #     all_interfer_pixel = np.where(all_interfer_pixel_index == interfer_pixel_index[ii],
                    #                                   interfer_pixel[ii], all_interfer_pixel)

                    # all_interfer_pixel_ssim = np.ones(np.shape(all_interfer_pixel_index), dtype=np.float32)
                    all_interfer_pixel_ssim = all_interfer_pixel
                    for ii in range(len(interfer_pixel_index)):
                        all_interfer_pixel_ssim = np.where(all_interfer_pixel_index == interfer_pixel_index[ii],
                                                           interfer_pixel[ii], all_interfer_pixel_ssim)

                    all_display_red_mask_ssim = all_Red_mask  # tf.cast(Red_mask)
                    all_display_green_mask_ssim = all_Green_mask  # tf.cast(Green_mask)
                    all_display_blue_mask_ssim = all_Blue_mask

                    for iii in range(len(all_red_pixel_index[0])):
                        all_display_red_mask_ssim = tf.where(
                            all_display_red_mask_ssim == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
                            all_interfer_pixel_ssim[all_red_pixel_index[0][iii]],
                            all_display_red_mask_ssim)  # mapped_pixel[iii], display_red_mask)
                    for jjj in range(len(all_green_pixel_index[0])):
                        all_display_green_mask_ssim = tf.where(
                            all_display_green_mask_ssim == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
                            all_interfer_pixel_ssim[all_green_pixel_index[0][jjj]],
                            all_display_green_mask_ssim)  # mapped_pixel[len(red_pixel_index[0]) + jjj], display_green_mask)
                    for kkk in range(len(all_blue_pixel_index[0])):
                        all_display_blue_mask_ssim = tf.where(
                            all_display_blue_mask_ssim == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
                            all_interfer_pixel_ssim[all_blue_pixel_index[0][kkk]], all_display_blue_mask_ssim)

                    all_display_RGB_mask_ssim = np.stack(
                        [all_display_red_mask_ssim, all_display_green_mask_ssim, all_display_blue_mask_ssim],
                        axis=2)

                    # plt.imshow(all_display_RGB_mask_ssim)
                    # plt.show()

                    Z_sast = math.pow(math.sqrt(1 / (4 * math.tan(theta_h / 2 * math.pi / 180) * math.tan(
                        theta_w / 2 * math.pi / 180)) * math.pow(1 / Obs_d, 2) / gamma_DOR), (
                                                  1 - math.pow(math.fabs(gamma_DOR - gamma_o),
                                                               beta_z) / alpha_z))  # (1-math.pow(math.fabs(gamma_DOR-gamma_o),beta_z)/alpha_z)
                    # Z_sast = 1
                    # test = cv2.resize(blur, None,None, fx=Z_sast, fy=Z_sast, interpolation=cv2.INTER_LANCZOS4)
                    # test = ResziePadding(blur,Z_sast,DOR_height)
                    # plt.imshow(test)
                    # plt.show()

                    # blur = cv2.blur(all_display_RGB_mask_ssim, (5, 5))
                    # blur = cv2.GaussianBlur(all_display_RGB_mask_ssim, (5, 5), 9)

                    all_display_RGB_mask_ori_blur = cv2.bilateralFilter(all_display_RGB_mask_ori, 5, 50, 50)
                    # all_display_RGB_mask_ori_blur = all_display_RGB_mask_ori
                    all_display_RGB_mask_ori_old = cv2.resize(all_display_RGB_mask_ori_blur, None, None, fx=Z_sast,
                                                              fy=Z_sast,
                                                              interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR INTER_CUBIC
                    # plt.imshow(all_display_RGB_mask_ori_old)
                    # plt.show()

                    all_display_RGB_mask_ssim_blur = cv2.bilateralFilter(all_display_RGB_mask_ssim, 5, 50, 50)
                    # all_display_RGB_mask_ssim_blur = all_display_RGB_mask_ssim
                    all_display_RGB_mask_ssim_new = cv2.resize(all_display_RGB_mask_ssim_blur, None, None, fx=Z_sast,
                                                               fy=Z_sast,
                                                               interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR INTER_CUBIC
                    # plt.imshow(all_display_RGB_mask_ssim_new)
                    # plt.show()

                    all_display_RGB_mask_ori_old = tf.expand_dims(all_display_RGB_mask_ori_old, axis=0)
                    all_display_RGB_mask_ssim_new = tf.expand_dims(all_display_RGB_mask_ssim_new, axis=0)

                    ssim_value, cs_map_value = tf_ssim(all_display_RGB_mask_ori_old, all_display_RGB_mask_ssim_new,
                                                       cs_map=True, mean_metric=False, filter_size=11,
                                                       filter_sigma=1.5)

                    # plt.imshow(ssim_value[0, :].numpy())
                    # plt.show()
                    #
                    # plt.imshow(cs_map_value[0, :].numpy())
                    # plt.show()
                    #
                    ssim_mean = tf.reduce_mean(ssim_value)
                    cs_mean = tf.reduce_mean(cs_map_value)

                    if (ssim_mean < ssim_min):
                        ssim_min = ssim_mean

                    # print(ssim_mean)
                    # print(cs_mean)

                    # ms_ssim_value, return_ssim_map_l = tf_ms_ssim_resize(all_display_RGB_mask_ssim_old, all_display_RGB_mask_ssim_new, weights=None,
                    #                                                      return_ssim_map=1, filter_size=11,
                    #                                                      filter_sigma=1.5)  # ,None, return_ssim_map=1 , return_ssim_map

                    # plt.imshow(vis.tensor_to_img_save(batch_img_temp, 0))
                    # plt.show()
                    #
                    # plt.imshow(vis.tensor_to_img_save(return_ssim_map_l, 0))
                    # plt.show()

                    # test = tf.expand_dims(blur, axis=0)
                    # test = tf.image.resize_with_pad(test, int(DOR_height * Z_sast), int(DOR_width * Z_sast), 'area', antialias=False)
                    #
                    # plt.imshow(test[0,:].numpy())
                    # plt.show()

                    # all_interfer_pixel = np.ones(np.shape(all_interfer_pixel_index), dtype=np.float32)

                    interfer_pixel = tf.constant(interfer_pixel, dtype=tf.float32)
                    # all_interfer_pixel = tf.constant(all_interfer_pixel, dtype=tf.float32)
                    # all_interfer_pixel = tf.Variable(initial_value=all_interfer_pixel, trainable=False)  # later we use sigmoid to map 0 to 1

                    # vars.append(interfer_pixel)
                    # vars.append(all_interfer_pixel)

                    # blobs_log = blob_log(img_new, max_sigma=30, num_sigma=10, threshold=.1)
                    # blobs_log = blob_doh(img_new)
                    # plt.imshow(display_interference_pre)
                    # plt.show()

                    # preprocessing ***************************************************************************************************
                    # map pattern values to range [0, 1]
                    # mapped_pattern = tf.sigmoid(pattern)
                    mapped_pattern = pattern / 255
                    # mapped_pixel = tf.sigmoid(interfer_pixel)
                    # mapped_all_pixel = tf.sigmoid(all_interfer_pixel)
                    mapped_pixel = interfer_pixel

                    # constant kernel
                    ## The default kernel is all 1 no longer design the kernel ###
                    # for ii in range(len(interfer_pixel_index)):
                    #     all_interfer_pixel = np.where(all_interfer_pixel_index == interfer_pixel_index[ii],
                    #                                   interfer_pixel[ii], all_interfer_pixel)
                    mapped_all_pixel = np.ones(np.shape(all_interfer_pixel_index), dtype=np.float32)
                    mapped_all_pixel = tf.constant(mapped_all_pixel)

                    # mapped_all_pixel = all_interfer_pixel

                    # mapped_all_pixel_ssim = all_interfer_pixel_ssim
                    mapped_u1_RGB_real_mask = u1_RGB_real_mask
                    # display_red_mask = Red_mask
                    # display_green_mask = Green_mask
                    # display_blue_mask = Blue_mask

                    ## The default kernel is all 1 no longer design the kernel ###
                    # for ii in range(len(interfer_pixel_index)):
                    #     mapped_all_pixel = tf.where(all_interfer_pixel_index == interfer_pixel_index[ii], mapped_pixel[ii], mapped_all_pixel)

                    for iii in range(len(all_interfer_pixel_index)):
                        mapped_u1_RGB_real_mask = tf.where(
                            u1_RGB_real_mask == tf.constant(all_interfer_pixel_index[iii], dtype=tf.float32),
                            mapped_all_pixel[iii], mapped_u1_RGB_real_mask)

                    # plt.imshow(mapped_u1_RGB_real_mask)
                    # plt.show()
                    PSFs, PSFs_RGB, interfer_all_pixel, u1_RGB_real = camera(mapped_u1_RGB_real_mask, DoR_option='small',
                                                                             mode_option='running')

                    interfer_img = get_interfer_img(Red_mask, Green_mask, Blue_mask, red_pixel_index, green_pixel_index,
                                                    blue_pixel_index, interfer_pixel_index, mapped_pixel, batch_size,
                                                    img_ori_height,
                                                    img_ori_width)

                    ## Obtain all interference pixel index
                    Red_pixel_mask = crop_image(cv2.resize(Red_mask, None, fx=3, fy=3, interpolation=cv2.INTER_AREA),
                                                img_ori_height, img_ori_width)
                    Green_pixel_mask = crop_image(
                        cv2.resize(Green_mask, None, fx=3, fy=3, interpolation=cv2.INTER_AREA),
                        img_ori_height, img_ori_width)
                    Blue_pixel_mask = crop_image(cv2.resize(Blue_mask, None, fx=3, fy=3, interpolation=cv2.INTER_AREA),
                                                 img_ori_height, img_ori_width)
                    RGB_pixel_mask = np.stack([Red_pixel_mask, Green_pixel_mask, Blue_pixel_mask],
                                              axis=2)  ## with pixel index

                    # test = tf.squeeze(interfer_img)
                    # test = np.array(test)
                    # plt.imshow(interfer_img[0, :, :, :].numpy())
                    # plt.show()

                    # compute PSF from pixel opening pattern and tiling method

                    # img_new = cv2.resize(np.array(u1_RGB_R_real), None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

                    # for ii in range(len(red_pixel_index)):

                    # for ii in range(len(interfer_pixel_index)):
                    #     display_interference_mask = np.where(display_interference_mask == interfer_pixel_index[ii], np.array(mapped_pixel)[ii], display_interference_mask)
                    # blobs_log = blob_log(img_new, max_sigma=30, num_sigma=10, threshold=.1)
                    # blobs_log = blob_doh(Interference_mask)
                    # Interference_mask = crop_image(cv2.resize(np.array(display_interference_mask), None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR), img_height, img_width)
                    # plt.imshow(Interference_mask)
                    # plt.show()

                    # for iii in range(len(red_pixel_index[0])):
                    #     display_red_mask = tf.where(display_red_mask == interfer_pixel_index[red_pixel_index[0][iii]], mapped_pixel[iii], display_red_mask)
                    # for jjj in range(len(green_pixel_index[0])):
                    #     display_green_mask = tf.where(display_green_mask == interfer_pixel_index[green_pixel_index[0][jjj]], mapped_pixel[len(red_pixel_index[0])+jjj], display_green_mask)
                    # for kkk in range(len(blue_pixel_index[0])):
                    #     display_blue_mask = tf.where(display_blue_mask == interfer_pixel_index[blue_pixel_index[0][kkk]], mapped_pixel[len(red_pixel_index[0])+len(green_pixel_index[0])+kkk], display_blue_mask)
                    #
                    # display_red_mask_ = crop_image(display_red_mask, 1524/3, 1524/3)
                    # display_green_mask_ = crop_image(display_green_mask, 1524/3, 1524/3)
                    # display_blue_mask_ = crop_image(display_blue_mask, 1524/3, 1524/3)
                    # interfer_img = tf.stack([display_red_mask_, display_green_mask_, display_blue_mask_], axis=2)
                    # interfer_img = tf.expand_dims(interfer_img, axis=0)
                    # for ii in range(batch_size - 1):
                    #     interfer_img = tf.concat([interfer_img, interfer_img], 0)
                    #
                    # map_size = tf.constant([1524 / 3 * 10, 1524 / 3 * 10])
                    # map_size = tf.dtypes.cast(tf.math.ceil(map_size), tf.int32)
                    #
                    # interfer_img = tf.image.resize_with_crop_or_pad(tf.image.resize(interfer_img, map_size, method='bilinear'), img_height, img_width)

                    # Red_img_mask = crop_image(tf.image.resize(display_red_mask, [img_width * 10, img_height * 10], method='bilinear'), img_height, img_width)
                    # Green_img_mask = crop_image(tf.image.resize(display_green_mask, [img_width * 10, img_height * 10], method='bilinear'), img_height, img_width)
                    # Blue_img_mask = crop_image(tf.image.resize(display_blue_mask, [img_width * 10, img_height * 10], method='bilinear'), img_height, img_width)

                    # Red_img_mask = crop_image(cv2.resize(display_red_mask, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR), img_height, img_width)
                    # Green_img_mask = crop_image(cv2.resize(np.array(display_green_mask), None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR), img_height, img_width)
                    # Blue_img_mask = crop_image(cv2.resize(np.array(display_blue_mask), None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR), img_height, img_width)

                    # plt.imshow(Red_img_mask)
                    # plt.show()
                    # plt.imshow(Green_img_mask)
                    # plt.show()
                    # plt.imshow(Blue_img_mask)
                    # plt.show()

                    # interfer_img = tf.stack([Red_img_mask, Green_img_mask, Blue_img_mask], axis=2)
                    # interfer_img = tf.expand_dims(interfer_img, axis=0)
                    # for ii in range(batch_size-1):
                    #     interfer_img = tf.concat([interfer_img, interfer_img], 0)

                    # apply PSF to images

                    ## obtain only interference mask
                    # batch_img_mask = tf.zeros_like(batch_img)
                    # captured_mask = capture(batch_img_mask, interfer_img, PSFs, PSFs_RGB)
                    # captured = capture(batch_img_mask, interfer_img, PSFs, PSFs_RGB)

                    # captured = capture(batch_img, interfer_img, PSFs, PSFs_RGB)
                    captured = capture_alpha(batch_img, interfer_img, PSFs, PSFs_RGB, alpha)
                    deconved = wiener_deconv(captured, PSFs)
                    captured_temp = captured
                    deconved_temp = deconved

                    # if captured.shape[2] == 1:
                    #     captured_test = np.reshape(captured, (captured.shape[0], captured.shape[1]))
                    #     captured_test = captured_test[0, :, :, :].numpy()
                    #     captured_test = (captured_test - np.amin(captured_test)) / (
                    #             np.amax(captured_test) - np.amin(captured_test))
                    # else:
                    #     captured_test = captured
                    #     captured_test = captured_test[0, :, :, :].numpy()
                    #     captured_test = (captured_test - np.amin(captured_test)) / (
                    #             np.amax(captured_test) - np.amin(captured_test))

                    # plt.imshow(batch_img[0, :, :, :].numpy())
                    # plt.show()

                    # plt.imshow(captured_test)
                    # plt.show()
                    #
                    # if (batch_id == 86):
                    #     print(batch_id)

                    batch_img_temp = random_size(batch_img, target_size=256)
                    batch_img_temp = center_crop(batch_img_temp)
                    # test1 = tf.squeeze(batch_img)
                    # test1 = np.array(test1)
                    # plt.imshow(batch_img_temp[0, :, :, :].numpy())
                    # plt.show()

                    # RGB_pixel_mask = RGB_pixel_mask[None, :, :, :] # tf.expand_dims(interfer_img, axis=0)
                    # RGB_pixel_mask = tf.cast(RGB_pixel_mask, dtype=tf.int32)
                    interfer_img_index = random_size_numpy(RGB_pixel_mask, target_size=256)
                    interfer_img_index = center_crop_numpy(interfer_img_index)

                    captured = random_size(captured, target_size=256)
                    captured = center_crop(captured)

                    # deconved = wiener_deconv(captured, PSFs)
                    deconved = random_size(deconved, target_size=256)
                    deconved = center_crop(deconved)

                    # test2 = tf.squeeze(interfer_img)
                    # test2 = np.array(test2)
                    # plt.imshow(interfer_img_index)
                    # plt.show()

                    # test3 = tf.squeeze(captured)
                    # test3 = np.array(test3)

                    # if captured.shape[2] == 1:
                    #     captured_test = np.reshape(captured, (captured.shape[0], captured.shape[1]))
                    #     captured_test = captured_test[0, :, :, :].numpy()
                    #     captured_test = (captured_test - np.amin(captured_test)) / (
                    #             np.amax(captured_test) - np.amin(captured_test))
                    # else:
                    #     captured_test = captured
                    #     captured_test = captured_test[0, :, :, :].numpy()
                    #     captured_test = (captured_test - np.amin(captured_test)) / (
                    #             np.amax(captured_test) - np.amin(captured_test))
                    # plt.imshow(captured_test)
                    # plt.show()

                    # test4 = tf.squeeze(deconved)
                    # test4 = np.array(test4)
                    # plt.imshow(deconved[0, :, :, :].numpy())
                    # plt.show()

                    ## GradCAM build

                    # image_cap_pil = None
                    # if captured.shape[2] == 1:
                    #     captured = np.reshape(captured, (captured.shape[0], captured.shape[1]))
                    #     image_cap_pil = Image.fromarray(captured, 'L')
                    # else:
                    #     image_cap_pil = Image.fromarray(captured)

                    # ms_ssim_value, return_ssim_map_l = tf_ms_ssim_resize(batch_img_temp, batch_img_temp, weights=None, return_ssim_map=1, filter_size=11, filter_sigma=1.5)#,None, return_ssim_map=1 , return_ssim_map
                    #
                    # plt.imshow(vis.tensor_to_img_save(batch_img_temp, 0))
                    # plt.show()
                    #
                    # plt.imshow(vis.tensor_to_img_save(return_ssim_map_l, 0))
                    # plt.show()

                    for iiii in range(batch_size):
                        captured_img = vis.tensor_to_img_save(captured, iiii)  # captured image in current batch
                        deconved_img = vis.tensor_to_img_save(deconved, iiii)

                    # img_path = "/data/volume_2/optimize_display_POLED_400PPI/logs/test/web/images/process_temp/temp.jpg" # opt.save_temp_dir
                    img_path = os.path.join(img_temp_dir, 'temp.jpg')
                    # Util.save_image(captured_img, img_path)
                    image_pil = None
                    if deconved_img.shape[2] == 1:
                        deconved_img = np.reshape(deconved_img, (deconved_img.shape[0], deconved_img.shape[1]))
                        image_pil = Image.fromarray(deconved_img, 'L')
                    else:
                        image_pil = Image.fromarray(deconved_img)
                    image_pil.save(img_path)
                    img = Image.open(img_path).convert('RGB')
                    img_np = np.array(img, dtype=np.uint8)
                    img.close()

                    # img = captured_img # image_pil

                    # [C, H, W]
                    img_tensor = data_transform(img_np)

                    # expand batch dimension
                    # [C, H, W] -> [N, C, H, W]
                    input_tensor = torch.unsqueeze(img_tensor, dim=0)

                    # prediction
                    # model = model.to(device)
                    model.eval()
                    with torch.no_grad():
                        # predict class
                        output = torch.squeeze(model(input_tensor.to(device))).cpu()
                        predict = torch.softmax(output, dim=0)
                        predict_cla = torch.argmax(predict).numpy()

                        predict_sort, idx_sort = torch.sort(torch.Tensor(predict), dim=0, descending=True)

                    if (color_flag == 1):

                        if find_color_flag == 1:

                            # 0 1 31 4 29 original
                            # 31 0 4 29 57
                            target_category_ori = idx_sort[0].numpy().tolist()  # tabby, tabby cat
                            target_category_ori_all = idx_sort.numpy().tolist()
                            target_category_prob = predict[predict_cla].numpy()
                            target_category_prob_all = predict.numpy()
                            # target_category = 254  # pug, pug-dog

                            grayscale_cam = cam(input_tensor=input_tensor,
                                                target_category=target_category_ori)  # target_category_ori
                            grayscale_cam = grayscale_cam[0, :]

                            all_redopt_pixel_index = np.unique(interfer_img_index[:, :, 0])
                            all_redopt_pixel_index = all_redopt_pixel_index[all_redopt_pixel_index != 0]
                            all_redopt_pixel_index = all_redopt_pixel_index.astype(int)

                            all_greenopt_pixel_index = np.unique(interfer_img_index[:, :, 1])
                            all_greenopt_pixel_index = all_greenopt_pixel_index[all_greenopt_pixel_index != 0]
                            all_greenopt_pixel_index = all_greenopt_pixel_index.astype(int)

                            all_blueopt_pixel_index = np.unique(interfer_img_index[:, :, 2])
                            all_blueopt_pixel_index = all_blueopt_pixel_index[all_blueopt_pixel_index != 0]
                            all_blueopt_pixel_index = all_blueopt_pixel_index.astype(int)

                            grayscale_cam_mean = 1 * np.mean(grayscale_cam)  # 1.5 1
                            grayscale_cam_mask = np.where(grayscale_cam <= grayscale_cam_mean, 0, 1)

                            # plt.imshow(grayscale_cam_mask)
                            # plt.show()
                            #
                            # visualization = show_cam_on_image(captured_img.astype(dtype=np.float32) / 255.,
                            #                                   grayscale_cam,
                            #                                   use_rgb=True)
                            #
                            # plt.imshow(visualization)
                            # plt.show()

                            redopt_pixel_mask = interfer_img_index[:, :, 0] * grayscale_cam_mask
                            greenopt_pixel_mask = interfer_img_index[:, :, 1] * grayscale_cam_mask
                            blueopt_pixel_mask = interfer_img_index[:, :, 2] * grayscale_cam_mask

                            if deconved.shape[2] == 1:
                                deconved_test = np.reshape(deconved, (deconved.shape[0], deconved.shape[1]))
                                deconved_test = deconved_test[0, :, :, :].numpy()
                                deconved_test = (deconved_test - np.amin(deconved_test)) / (
                                        np.amax(deconved_test) - np.amin(deconved_test))
                            else:
                                deconved_test = deconved
                                deconved_test = deconved_test[0, :, :, :].numpy()
                                deconved_test = (deconved_test - np.amin(deconved_test)) / (
                                        np.amax(deconved_test) - np.amin(deconved_test))

                            red_deconved_mask = deconved_test[:, :, 0] * grayscale_cam_mask
                            green_deconved_mask = deconved_test[:, :, 1] * grayscale_cam_mask
                            blue_deconved_mask = deconved_test[:, :, 2] * grayscale_cam_mask

                            # test = redopt_pixel_mask.reshape(-1)

                            ## get the element with the most occurrences

                            # redopt_pixel_mask_count = np.bincount(interfer_img_index[:, :, 0].reshape(-1).astype(int))
                            # greenopt_pixel_mask_count = np.bincount(interfer_img_index[:, :, 1].reshape(-1).astype(int))
                            # blueopt_pixel_mask_count = np.bincount(interfer_img_index[:, :, 2].reshape(-1).astype(int))

                            redopt_pixel_mask_count = np.bincount(redopt_pixel_mask.reshape(-1).astype(int))
                            greenopt_pixel_mask_count = np.bincount(greenopt_pixel_mask.reshape(-1).astype(int))
                            blueopt_pixel_mask_count = np.bincount(blueopt_pixel_mask.reshape(-1).astype(int))

                            redopt_pixel_mask_count[
                                np.argmax(redopt_pixel_mask_count)] = 0  # np.min(redopt_pixel_mask_count)
                            greenopt_pixel_mask_count[
                                np.argmax(greenopt_pixel_mask_count)] = 0  # np.min(greenopt_pixel_mask_count)
                            blueopt_pixel_mask_count[
                                np.argmax(blueopt_pixel_mask_count)] = 0  # np.min(blueopt_pixel_mask_count)

                            redopt_pixel_maxnumindex = np.argmax(redopt_pixel_mask_count)
                            greenopt_pixel_maxnumindex = np.argmax(greenopt_pixel_mask_count)
                            blueopt_pixel_maxnumindex = np.argmax(blueopt_pixel_mask_count)

                            redopt_pixel_mask_index = np.unique(redopt_pixel_mask)
                            redopt_pixel_mask_index = redopt_pixel_mask_index[redopt_pixel_mask_index != 0]
                            redopt_pixel_mask_index = redopt_pixel_mask_index.astype(int)
                            redpixels_len = len(redopt_pixel_mask_index)

                            ##Find the complete set red
                            redopt_allpixel_mask_index = np.unique(interfer_img_index[:, :, 0])
                            redopt_allpixel_mask_index = redopt_allpixel_mask_index[redopt_allpixel_mask_index != 0]
                            redopt_allpixel_mask_index = redopt_allpixel_mask_index.astype(int)

                            if redpixels_len == 0:
                                redopt_pixel_mask_index = redopt_allpixel_mask_index
                                redpixels_len = len(redopt_pixel_mask_index)

                            ##Find the difference set red
                            redopt_respixel_mask_index = np.setdiff1d(redopt_allpixel_mask_index,
                                                                      redopt_pixel_mask_index)

                            redpixels_count = 0
                            redpixels_prob = np.zeros(np.shape(redopt_pixel_mask_index), dtype=np.float32)
                            pickinterfer_redopt_index = np.zeros_like(redopt_pixel_mask_index)
                            redopt_pickindex = 0

                            greenopt_pixel_mask_index = np.unique(greenopt_pixel_mask)
                            greenopt_pixel_mask_index = greenopt_pixel_mask_index[greenopt_pixel_mask_index != 0]
                            greenopt_pixel_mask_index = greenopt_pixel_mask_index.astype(int)
                            greenpixels_len = len(greenopt_pixel_mask_index)

                            ##Find the complete set green
                            greenopt_allpixel_mask_index = np.unique(interfer_img_index[:, :, 1])
                            greenopt_allpixel_mask_index = greenopt_allpixel_mask_index[
                                greenopt_allpixel_mask_index != 0]
                            greenopt_allpixel_mask_index = greenopt_allpixel_mask_index.astype(int)

                            if greenpixels_len == 0:
                                greenopt_pixel_mask_index = greenopt_allpixel_mask_index
                                greenpixels_len = len(greenopt_pixel_mask_index)

                            ##Find the difference set green
                            greenopt_respixel_mask_index = np.setdiff1d(greenopt_allpixel_mask_index,
                                                                        greenopt_pixel_mask_index)

                            greenpixels_count = 0
                            greenpixels_prob = np.zeros(np.shape(greenopt_pixel_mask_index), dtype=np.float32)
                            pickinterfer_greenopt_index = np.zeros_like(greenopt_pixel_mask_index)
                            greenopt_pickindex = 0

                            blueopt_pixel_mask_index = np.unique(blueopt_pixel_mask)
                            blueopt_pixel_mask_index = blueopt_pixel_mask_index[blueopt_pixel_mask_index != 0]
                            blueopt_pixel_mask_index = blueopt_pixel_mask_index.astype(int)
                            bluepixels_len = len(blueopt_pixel_mask_index)

                            ##Find the complete set blue
                            blueopt_allpixel_mask_index = np.unique(interfer_img_index[:, :, 2])
                            blueopt_allpixel_mask_index = blueopt_allpixel_mask_index[blueopt_allpixel_mask_index != 0]
                            blueopt_allpixel_mask_index = blueopt_allpixel_mask_index.astype(int)

                            if bluepixels_len == 0:
                                blueopt_pixel_mask_index = blueopt_allpixel_mask_index.astype(int)
                                bluepixels_len = len(blueopt_pixel_mask_index)

                            ##Find the difference set blue
                            blueopt_respixel_mask_index = np.setdiff1d(blueopt_allpixel_mask_index,
                                                                       blueopt_pixel_mask_index)

                            bluepixels_count = 0
                            bluepixels_prob = np.zeros(np.shape(blueopt_pixel_mask_index), dtype=np.float32)
                            pickinterfer_blueopt_index = np.zeros_like(blueopt_pixel_mask_index)
                            blueopt_pickindex = 0

                            ## Color effect pick one pixel to test color effect

                            redopt_pixel_pick = redopt_pixel_maxnumindex  # redopt_pixel_mask_index[int(len(redopt_pixel_mask_index) / 2)]
                            greenopt_pixel_pick = greenopt_pixel_maxnumindex  # greenopt_pixel_mask_index[int(len(greenopt_pixel_mask_index) / 2)]
                            blueopt_pixel_pick = blueopt_pixel_maxnumindex  # blueopt_pixel_mask_index[int(len(blueopt_pixel_mask_index) / 2)]

                            # redopt_pixel_pick_mask = np.where(interfer_img_index[:, :, 0] == redopt_pixel_pick, 1, 0)
                            # greenopt_pixel_pick_mask = np.where(interfer_img_index[:, :, 1] == greenopt_pixel_pick, 1, 0)
                            # blueopt_pixel_pick_mask = np.where(interfer_img_index[:, :, 2] == blueopt_pixel_pick, 1, 0)

                            redopt_pixel_pick_mask = np.where(redopt_pixel_mask == redopt_pixel_pick, 1, 0)
                            greenopt_pixel_pick_mask = np.where(greenopt_pixel_mask == greenopt_pixel_pick, 1, 0)
                            blueopt_pixel_pick_mask = np.where(blueopt_pixel_mask == blueopt_pixel_pick, 1, 0)

                            redopt_pixel_effect = np.sum(redopt_pixel_pick_mask * grayscale_cam)
                            greenopt_pixel_effect = np.sum(greenopt_pixel_pick_mask * grayscale_cam)
                            blueopt_pixel_effect = np.sum(blueopt_pixel_pick_mask * grayscale_cam)

                            redopt_pixel_effect_v1 = np.sum(red_deconved_mask)
                            greenopt_pixel_effect_v1 = np.sum(green_deconved_mask)
                            blueopt_pixel_effect_v1 = np.sum(blue_deconved_mask)

                            find_color_flag = 0
                        else:
                            ini_intensity = opt.itini_intensity  # 0.1 0.5
                            judge_crteria = -np.max(
                                np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                            if color_red_flag == 1:
                                if (redpixels_count <= redpixels_len):

                                    if redpixels_count < redpixels_len:
                                        interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                        interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                                                            interfer_pixel_red_index,
                                                                            interfer_pixel_green_index,
                                                                            interfer_pixel_blue_index)
                                        interfer_pixel[
                                            interfer_pixel_index == redopt_pixel_mask_index[
                                                redpixels_count]] += ini_intensity
                                    if redpixels_count > 0:
                                        redpixels_prob[redpixels_count - 1] = judge_crteria
                                    redpixels_count += 1
                                else:
                                    color_red_flag = 0
                            else:
                                if color_green_flag == 1:
                                    if (greenpixels_count <= greenpixels_len):
                                        if greenpixels_count < greenpixels_len:
                                            interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                            interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                                                                interfer_pixel_red_index,
                                                                                interfer_pixel_green_index,
                                                                                interfer_pixel_blue_index)
                                            interfer_pixel[
                                                interfer_pixel_index == greenopt_pixel_mask_index[
                                                    greenpixels_count]] += ini_intensity

                                        if greenpixels_count > 0:
                                            greenpixels_prob[greenpixels_count - 1] = judge_crteria

                                        greenpixels_count += 1
                                    else:
                                        color_green_flag = 0
                                else:
                                    if color_blue_flag == 1:
                                        if (bluepixels_count <= bluepixels_len):
                                            if bluepixels_count < bluepixels_len:
                                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index),
                                                                          dtype=np.float32)
                                                interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                                                                    interfer_pixel_red_index,
                                                                                    interfer_pixel_green_index,
                                                                                    interfer_pixel_blue_index)
                                                interfer_pixel[
                                                    interfer_pixel_index == blueopt_pixel_mask_index[
                                                        bluepixels_count]] += ini_intensity
                                            if bluepixels_count > 0:
                                                bluepixels_prob[bluepixels_count - 1] = judge_crteria
                                            bluepixels_count += 1
                                        else:
                                            color_blue_flag = 0

                        if color_red_flag == 0 and color_green_flag == 0 and color_blue_flag == 0:

                            redpixels_prob_temp = redpixels_prob
                            redopt_pixel_mask_index_temp = redopt_pixel_mask_index

                            greenpixels_prob_temp = greenpixels_prob
                            greenopt_pixel_mask_index_temp = greenopt_pixel_mask_index

                            bluepixels_prob_temp = bluepixels_prob
                            blueopt_pixel_mask_index_temp = blueopt_pixel_mask_index

                            if any(redpixels_prob):
                                redpixels_prob_min = np.min(redpixels_prob)
                                redpixels_prob_min_index = np.argmin(redpixels_prob)

                            else:
                                redpixels_prob_min = 0

                            if any(greenpixels_prob):
                                greenpixels_prob_min = np.min(greenpixels_prob)
                                greenpixels_prob_min_index = np.argmin(greenpixels_prob)


                            else:
                                greenpixels_prob_min = 0

                            if any(bluepixels_prob):
                                bluepixels_prob_min = np.min(bluepixels_prob)
                                bluepixels_prob_min_index = np.argmin(bluepixels_prob)


                            else:
                                bluepixels_prob_min = 0

                            # redpixels_prob_min = np.min(redpixels_prob)
                            # greenpixels_prob_min = np.min(greenpixels_prob)
                            # bluepixels_prob_min = np.min(bluepixels_prob)

                            if (
                                    greenpixels_prob_min < bluepixels_prob_min and bluepixels_prob_min < redpixels_prob_min) or (
                                    bluepixels_prob_min < greenpixels_prob_min and greenpixels_prob_min < redpixels_prob_min):
                                green_flag = 1
                                blue_flag = 2

                                if greenpixels_len == 0:
                                    green_flag = 0
                                if bluepixels_len == 0:
                                    blue_flag = 0
                                if greenpixels_len == 0 and bluepixels_len != 0:
                                    blue_flag = 1
                                    bluepixels_intensity_flag = 1
                                    bluepixels_intensity_count = 0
                                if greenpixels_len == 0 and bluepixels_len == 0:
                                    if redpixels_len != 0:
                                        red_flag = 1
                                        redpixels_intensity_flag = 1
                                        redpixels_intensity_count = 0

                                    else:
                                        location_flag = 0

                                if green_flag != 0:
                                    greenpixels_intensity_flag = 1
                                    greenpixels_intensity_count = 0

                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                                                    interfer_pixel_red_index,
                                                                    interfer_pixel_green_index,
                                                                    interfer_pixel_blue_index)


                            elif (
                                    greenpixels_prob_min < redpixels_prob_min and redpixels_prob_min < bluepixels_prob_min) or (
                                    redpixels_prob_min < greenpixels_prob_min and greenpixels_prob_min < bluepixels_prob_min):
                                green_flag = 1
                                red_flag = 2
                                if greenpixels_len == 0:
                                    green_flag = 0
                                if redpixels_len == 0:
                                    red_flag = 0
                                if greenpixels_len == 0 and redpixels_len != 0:
                                    red_flag = 1
                                    redpixels_intensity_flag = 1
                                    redpixels_intensity_count = 0
                                if greenpixels_len == 0 and redpixels_len == 0:
                                    if bluepixels_len != 0:
                                        blue_flag = 1
                                        bluepixels_intensity_flag = 1
                                        bluepixels_intensity_count = 0
                                    else:
                                        location_flag = 0

                                if green_flag != 0:
                                    greenpixels_intensity_flag = 1
                                    greenpixels_intensity_count = 0

                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                                                    interfer_pixel_red_index,
                                                                    interfer_pixel_green_index,
                                                                    interfer_pixel_blue_index)



                            elif (
                                    bluepixels_prob_min < redpixels_prob_min and redpixels_prob_min < greenpixels_prob_min) or (
                                    redpixels_prob_min < bluepixels_prob_min and bluepixels_prob_min < greenpixels_prob_min):
                                blue_flag = 1
                                red_flag = 2
                                if bluepixels_len == 0:
                                    blue_flag = 0
                                if redpixels_len == 0:
                                    red_flag = 0
                                if bluepixels_len == 0 and redpixels_len != 0:
                                    red_flag = 1
                                    redpixels_intensity_flag = 1
                                    redpixels_intensity_count = 0
                                if bluepixels_len == 0 and redpixels_len == 0:
                                    if greenpixels_len != 0:
                                        green_flag = 1
                                        greenpixels_intensity_flag = 1
                                        greenpixels_intensity_count = 0
                                    else:
                                        location_flag = 0

                                if blue_flag != 0:
                                    bluepixels_intensity_flag = 1
                                    bluepixels_intensity_count = 0

                                interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                                                    interfer_pixel_red_index,
                                                                    interfer_pixel_green_index,
                                                                    interfer_pixel_blue_index)


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
                        step_size = opt.itstep_size  # 0.1
                        max_brightness = opt.maxperturbation_power  # max_sum_power 10
                        probdiff_th = opt.probdiff_threshold

                        judge_test = np.max(
                            np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                        iter_count += 1
                        ## criteria 1
                        # judge_crteria = predict[predict_cla].numpy() - target_category_prob
                        ## criteria 2
                        judge_crteria = -np.max(
                            np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                        if prob_label_flag == 1:
                            if green_flag == 1:
                                if greenpixels_intensity_count == 1:
                                    judge_crteria_index = np.argmax(
                                        np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                                    prob_label_flag = 0

                            elif blue_flag == 1:
                                if bluepixels_intensity_count == 1:
                                    judge_crteria_index = np.argmax(
                                        np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                                    prob_label_flag = 0

                            elif red_flag == 1:
                                if redpixels_intensity_count == 1:
                                    judge_crteria_index = np.argmax(
                                        np.divide(predict.numpy() - target_category_prob_all, target_category_prob_all))
                                    prob_label_flag = 0

                        if greenpixels_intensity_count > 0 or bluepixels_intensity_count > 0 or redpixels_intensity_count > 0:
                            judge_targety_prob = predict[judge_crteria_index].numpy()

                        if (
                                target_category == target_category_ori):  # target_category == target_category_ori judge_test[iter_count-1] < 100 judge_test < 50
                            if (green_flag == 1):
                                if (greenpixels_intensity_flag):  # greenpixels_count <= greenpixels_len
                                    if greenpixels_intensity_count == 0:
                                        # interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                        # interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                        #                                     interfer_pixel_red_index,
                                        #                                     interfer_pixel_green_index,
                                        #                                     interfer_pixel_blue_index)
                                        # picked optimal green location
                                        interfer_pixel = interfer_pixel.numpy()
                                        pickinterfer_greenopt_index[greenopt_pickindex] = \
                                            np.where(interfer_pixel_index == greenopt_pixel_mask_index[
                                                np.argmin(greenpixels_prob)])[0]
                                        interfer_pixel[
                                            pickinterfer_greenopt_index[greenopt_pickindex]] += ini_intensity

                                    if greenpixels_intensity_count > 0:
                                        interfer_pixel = interfer_pixel.numpy()
                                        interfer_pixel[pickinterfer_greenopt_index[greenopt_pickindex]] += step_size
                                        greenpixels_intensity_prob[greenpixels_intensity_count - 1] = judge_targety_prob
                                        if greenpixels_intensity_count > 1:
                                            greenpixels_intensity_diff[greenpixels_intensity_count - 2] = \
                                            greenpixels_intensity_prob[greenpixels_intensity_count - 1] - \
                                            greenpixels_intensity_prob[greenpixels_intensity_count - 2]
                                        if greenpixels_intensity_count > 2:
                                            if greenpixels_intensity_diff[greenpixels_intensity_count - 2] - \
                                                    greenpixels_intensity_diff[greenpixels_intensity_count - 3] < probdiff_th:
                                                diff_flag = 0
                                                greenpixels_intensity_flag = 0
                                                if (blue_flag == 2) and (blueopt_pickindex <= bluepixels_len - 1):
                                                    bluepixels_intensity_flag = 1
                                                    bluepixels_intensity_count = 0
                                                elif (red_flag == 2) and (redopt_pickindex <= redpixels_len - 1):
                                                    redpixels_intensity_flag = 1
                                                    redpixels_intensity_count = 0
                                                else:

                                                    if greenopt_pickindex < greenpixels_len - 1:
                                                        greenopt_pickindex += 1
                                                        greenpixels_intensity_flag = 1
                                                        greenpixels_intensity_count = -1
                                                        greenopt_pixel_mask_index = np.delete(greenopt_pixel_mask_index,
                                                                                              np.argmin(
                                                                                                  greenpixels_prob))
                                                        greenpixels_prob = np.delete(greenpixels_prob,
                                                                                     np.argmin(greenpixels_prob))
                                                    else:
                                                        if (np.all(interfer_pixel[pickinterfer_greenopt_index[
                                                                                 0:greenpixels_len]] > (
                                                                          max_brightness + step_size)) and np.all(
                                                                interfer_pixel[
                                                                    pickinterfer_blueopt_index[
                                                                    0:bluepixels_len]] > (
                                                                        max_brightness + step_size))) or (np.all(interfer_pixel[pickinterfer_greenopt_index[
                                                                                 0:greenpixels_len]] > (
                                                                          max_brightness + step_size)) and np.all(
                                                                interfer_pixel[
                                                                    pickinterfer_redopt_index[
                                                                    0:redpixels_len]] > (
                                                                        max_brightness + step_size))):
                                                            location_flag = 0

                                                        else:
                                                            redpixels_prob = redpixels_prob_temp
                                                            greenpixels_prob = greenpixels_prob_temp
                                                            bluepixels_prob = bluepixels_prob_temp
                                                            redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                            greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                            blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                            redopt_pickindex = 0
                                                            greenopt_pickindex = 0
                                                            blueopt_pickindex = 0

                                                            greenpixels_intensity_flag = 1
                                                            greenpixels_intensity_count = -1


                                    greenpixels_intensity_count += 1


                                                        # location_flag = 0

                                    if greenopt_pickindex <= greenpixels_len - 1 and diff_flag:
                                        if np.all(interfer_pixel[pickinterfer_greenopt_index[greenopt_pickindex]] > (
                                                max_brightness + step_size)):
                                            greenpixels_intensity_flag = 0
                                            if (blue_flag == 2) and (blueopt_pickindex <= bluepixels_len - 1):
                                                # if blueopt_pickindex < bluepixels_len - 1:
                                                bluepixels_intensity_flag = 1
                                                bluepixels_intensity_count = 0
                                            elif (red_flag == 2) and (redopt_pickindex <= redpixels_len - 1):
                                                # if redopt_pickindex < redpixels_len - 1:
                                                redpixels_intensity_flag = 1
                                                redpixels_intensity_count = 0
                                            else:

                                                if greenopt_pickindex <= greenpixels_len - 1:
                                                    greenopt_pickindex += 1

                                                    if greenopt_pickindex <= greenpixels_len - 1:
                                                        greenpixels_intensity_flag = 1
                                                        greenpixels_intensity_count = 0
                                                        greenopt_pixel_mask_index = np.delete(greenopt_pixel_mask_index,
                                                                                              np.argmin(
                                                                                                  greenpixels_prob))
                                                        greenpixels_prob = np.delete(greenpixels_prob,
                                                                                     np.argmin(greenpixels_prob))

                                                else:
                                                    redpixels_prob = redpixels_prob_temp
                                                    greenpixels_prob = greenpixels_prob_temp
                                                    bluepixels_prob = bluepixels_prob_temp
                                                    redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                    greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                    blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                    redopt_pickindex = 0
                                                    greenopt_pickindex = 0
                                                    blueopt_pickindex = 0

                                                    greenpixels_intensity_flag = 1
                                                    greenpixels_intensity_count = 0

                                    elif greenopt_pickindex > greenpixels_len - 1 and diff_flag:
                                        greenpixels_intensity_flag = 0
                                        greenpixels_intensity_count = 0
                                        if (blue_flag == 2) and (blueopt_pickindex <= bluepixels_len - 1):
                                            # if blueopt_pickindex < bluepixels_len - 1:
                                            bluepixels_intensity_flag = 1
                                            bluepixels_intensity_count = 0
                                        elif (red_flag == 2) and (redopt_pickindex <= redpixels_len - 1):
                                            # if redopt_pickindex < redpixels_len - 1:
                                            redpixels_intensity_flag = 1
                                            redpixels_intensity_count = 0
                                        else:
                                            redpixels_prob = redpixels_prob_temp
                                            greenpixels_prob = greenpixels_prob_temp
                                            bluepixels_prob = bluepixels_prob_temp
                                            redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                            greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                            blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                            redopt_pickindex = 0
                                            greenopt_pickindex = 0
                                            blueopt_pickindex = 0

                                            greenpixels_intensity_flag = 1
                                            greenpixels_intensity_count = 0

                                    diff_flag = 1



                                    # greenpixels_intensity_count += 1


                                else:
                                    if (blue_flag == 2):
                                        if (bluepixels_intensity_flag):
                                            # picked optimal blue location
                                            if bluepixels_intensity_count == 0:
                                                pickinterfer_blueopt_index[blueopt_pickindex] = \
                                                np.where(interfer_pixel_index == blueopt_pixel_mask_index[
                                                    np.argmin(np.absolute(
                                                        blueopt_pixel_mask_index - greenopt_pixel_mask_index[
                                                            np.argmin(greenpixels_prob)]))])[0]

                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[
                                                    pickinterfer_blueopt_index[blueopt_pickindex]] += ini_intensity
                                            if bluepixels_intensity_count > 0:
                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[
                                                    pickinterfer_blueopt_index[blueopt_pickindex]] += step_size
                                                bluepixels_intensity_prob[
                                                    bluepixels_intensity_count - 1] = judge_targety_prob
                                                if bluepixels_intensity_count > 1:
                                                    bluepixels_intensity_diff[bluepixels_intensity_count - 2] = \
                                                    bluepixels_intensity_prob[bluepixels_intensity_count - 1] - \
                                                    bluepixels_intensity_prob[bluepixels_intensity_count - 2]
                                                if bluepixels_intensity_count > 2:
                                                    if bluepixels_intensity_diff[bluepixels_intensity_count - 2] - \
                                                            bluepixels_intensity_diff[
                                                                bluepixels_intensity_count - 3] < probdiff_th:

                                                        diff_flag = 0

                                                        if blueopt_pickindex <= bluepixels_len - 1:
                                                            blueopt_pickindex += 1
                                                            if blueopt_pickindex <= bluepixels_len - 1:
                                                                blueopt_pixel_mask_index = np.delete(
                                                                    blueopt_pixel_mask_index, np.argmin(np.absolute(
                                                                        blueopt_pixel_mask_index -
                                                                        greenopt_pixel_mask_index[
                                                                            np.argmin(greenpixels_prob)])))


                                                        if greenopt_pickindex <= greenpixels_len - 1:
                                                            greenopt_pickindex += 1
                                                            if greenopt_pickindex <= greenpixels_len - 1:
                                                                greenopt_pixel_mask_index = np.delete(
                                                                    greenopt_pixel_mask_index,
                                                                    np.argmin(greenpixels_prob))
                                                                greenpixels_prob = np.delete(greenpixels_prob,
                                                                                             np.argmin(
                                                                                                 greenpixels_prob))


                                                        bluepixels_intensity_flag = 0
                                                        if greenopt_pickindex <= greenpixels_len - 1:
                                                            greenpixels_intensity_flag = 1
                                                            greenpixels_intensity_count = 0
                                                        elif blueopt_pickindex <= bluepixels_len - 1:
                                                            bluepixels_intensity_flag = 1
                                                            bluepixels_intensity_count = -1
                                                        else:
                                                            if (np.all(interfer_pixel[pickinterfer_greenopt_index[
                                                                                      0:greenpixels_len]] > (
                                                                               max_brightness + step_size)) and np.all(
                                                                interfer_pixel[
                                                                    pickinterfer_blueopt_index[
                                                                    0:bluepixels_len]] > (
                                                                        max_brightness + step_size))):

                                                                location_flag = 0

                                                            else:
                                                                redpixels_prob = redpixels_prob_temp
                                                                greenpixels_prob = greenpixels_prob_temp
                                                                bluepixels_prob = bluepixels_prob_temp
                                                                redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                                greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                                blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                                redopt_pickindex = 0
                                                                greenopt_pickindex = 0
                                                                blueopt_pickindex = 0

                                                                greenpixels_intensity_flag = 1
                                                                greenpixels_intensity_count = 0




                                            bluepixels_intensity_count += 1

                                        if blueopt_pickindex <= bluepixels_len - 1 and diff_flag:
                                            if interfer_pixel[pickinterfer_blueopt_index[blueopt_pickindex]] > (
                                                    max_brightness + step_size):

                                                if blueopt_pickindex <= bluepixels_len - 1:
                                                    blueopt_pickindex += 1
                                                    if blueopt_pickindex <= bluepixels_len - 1:
                                                        blueopt_pixel_mask_index = np.delete(blueopt_pixel_mask_index,
                                                                                             np.argmin(np.absolute(
                                                                                                 blueopt_pixel_mask_index -
                                                                                                 greenopt_pixel_mask_index[
                                                                                                     np.argmin(
                                                                                                         greenpixels_prob)])))


                                                if greenopt_pickindex <= greenpixels_len - 1:
                                                    greenopt_pickindex += 1
                                                    if greenopt_pickindex <= greenpixels_len - 1:
                                                        greenopt_pixel_mask_index = np.delete(greenopt_pixel_mask_index,
                                                                                              np.argmin(
                                                                                                  greenpixels_prob))
                                                        greenpixels_prob = np.delete(greenpixels_prob,
                                                                                     np.argmin(greenpixels_prob))


                                                bluepixels_intensity_flag = 0
                                                if greenopt_pickindex <= greenpixels_len - 1:
                                                    greenpixels_intensity_flag = 1
                                                    greenpixels_intensity_count = 0
                                                elif blueopt_pickindex <= bluepixels_len - 1:
                                                    bluepixels_intensity_flag = 1
                                                    bluepixels_intensity_count = 0
                                                else:
                                                    redpixels_prob = redpixels_prob_temp
                                                    greenpixels_prob = greenpixels_prob_temp
                                                    bluepixels_prob = bluepixels_prob_temp
                                                    redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                    greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                    blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                    redopt_pickindex = 0
                                                    greenopt_pickindex = 0
                                                    blueopt_pickindex = 0

                                                    greenpixels_intensity_flag = 1
                                                    greenpixels_intensity_count = 0

                                        elif blueopt_pickindex > bluepixels_len - 1 and diff_flag:
                                            bluepixels_intensity_flag = 0
                                            bluepixels_intensity_count = 0
                                            if greenopt_pickindex <= greenpixels_len - 1:
                                                greenpixels_intensity_flag = 1
                                                greenpixels_intensity_count = 0
                                            else:
                                                redpixels_prob = redpixels_prob_temp
                                                greenpixels_prob = greenpixels_prob_temp
                                                bluepixels_prob = bluepixels_prob_temp
                                                redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                redopt_pickindex = 0
                                                greenopt_pickindex = 0
                                                blueopt_pickindex = 0

                                                greenpixels_intensity_flag = 1
                                                greenpixels_intensity_count = 0

                                        diff_flag = 1


                                        if type(interfer_pixel) is not np.ndarray:
                                            interfer_pixel = interfer_pixel.numpy()
                                        if (np.all(interfer_pixel[pickinterfer_greenopt_index[0:greenpixels_len]] > (max_brightness + step_size)) and np.all(interfer_pixel[pickinterfer_blueopt_index[0:bluepixels_len]] > (
                                                        max_brightness + step_size))):
                                            location_flag = 0












                                    elif (red_flag == 2):
                                        if (redpixels_intensity_flag):

                                            if redpixels_intensity_count == 0:
                                                pickinterfer_redopt_index[redopt_pickindex] = \
                                                    np.where(interfer_pixel_index == redopt_pixel_mask_index[
                                                        np.argmin(np.absolute(
                                                            redopt_pixel_mask_index - greenopt_pixel_mask_index[
                                                                np.argmin(greenpixels_prob)]))])[0]

                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[
                                                    pickinterfer_redopt_index[redopt_pickindex]] += ini_intensity
                                            if redpixels_intensity_count > 0:
                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[
                                                    pickinterfer_redopt_index[redopt_pickindex]] += step_size
                                                redpixels_intensity_prob[
                                                    redpixels_intensity_count - 1] = judge_targety_prob
                                                if redpixels_intensity_count > 1:
                                                    redpixels_intensity_diff[redpixels_intensity_count - 2] = \
                                                        redpixels_intensity_prob[redpixels_intensity_count - 1] - \
                                                        redpixels_intensity_prob[redpixels_intensity_count - 2]
                                                if redpixels_intensity_count > 2:
                                                    if redpixels_intensity_diff[redpixels_intensity_count - 2] - \
                                                            redpixels_intensity_diff[
                                                                redpixels_intensity_count - 3] < probdiff_th:

                                                        diff_flag = 0

                                                        if redopt_pickindex <= redpixels_len - 1:
                                                            redopt_pickindex += 1
                                                            if redopt_pickindex <= redpixels_len - 1:
                                                                redopt_pixel_mask_index = np.delete(
                                                                    redopt_pixel_mask_index, np.argmin(np.absolute(
                                                                        redopt_pixel_mask_index -
                                                                        greenopt_pixel_mask_index[
                                                                            np.argmin(greenpixels_prob)])))


                                                        if greenopt_pickindex <= greenpixels_len - 1:
                                                            greenopt_pickindex += 1
                                                            if greenopt_pickindex <= greenpixels_len - 1:
                                                                greenopt_pixel_mask_index = np.delete(
                                                                    greenopt_pixel_mask_index,
                                                                    np.argmin(greenpixels_prob))
                                                                greenpixels_prob = np.delete(greenpixels_prob,
                                                                                             np.argmin(
                                                                                                 greenpixels_prob))



                                                        redpixels_intensity_flag = 0
                                                        if greenopt_pickindex <= greenpixels_len - 1:
                                                            greenpixels_intensity_flag = 1
                                                            greenpixels_intensity_count = 0
                                                        elif redopt_pickindex <= redpixels_len - 1:
                                                            redpixels_intensity_flag = 1
                                                            redpixels_intensity_count = -1
                                                        else:
                                                            if (np.all(interfer_pixel[pickinterfer_greenopt_index[
                                                                                      0:greenpixels_len]] > (
                                                                               max_brightness + step_size)) and np.all(
                                                                interfer_pixel[
                                                                    pickinterfer_redopt_index[
                                                                    0:redpixels_len]] > (
                                                                        max_brightness + step_size))):

                                                                location_flag = 0

                                                            else:
                                                                redpixels_prob = redpixels_prob_temp
                                                                greenpixels_prob = greenpixels_prob_temp
                                                                bluepixels_prob = bluepixels_prob_temp
                                                                redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                                greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                                blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                                redopt_pickindex = 0
                                                                greenopt_pickindex = 0
                                                                blueopt_pickindex = 0

                                                                greenpixels_intensity_flag = 1
                                                                greenpixels_intensity_count = 0




                                            redpixels_intensity_count += 1

                                        if redopt_pickindex <= redpixels_len - 1 and diff_flag:
                                            if interfer_pixel[pickinterfer_redopt_index[redopt_pickindex]] > (
                                                    max_brightness + step_size):

                                                if redopt_pickindex <= redpixels_len - 1:
                                                    redopt_pickindex += 1
                                                    if redopt_pickindex <= redpixels_len - 1:
                                                        redopt_pixel_mask_index = np.delete(redopt_pixel_mask_index,
                                                                                            np.argmin(np.absolute(
                                                                                                redopt_pixel_mask_index -
                                                                                                greenopt_pixel_mask_index[
                                                                                                    np.argmin(
                                                                                                        greenpixels_prob)])))


                                                if greenopt_pickindex <= greenpixels_len - 1:
                                                    greenopt_pickindex += 1
                                                    if greenopt_pickindex <= greenpixels_len - 1:
                                                        greenopt_pixel_mask_index = np.delete(greenopt_pixel_mask_index,
                                                                                              np.argmin(
                                                                                                  greenpixels_prob))
                                                        greenpixels_prob = np.delete(greenpixels_prob,
                                                                                     np.argmin(greenpixels_prob))


                                                redpixels_intensity_flag = 0
                                                if greenopt_pickindex <= greenpixels_len - 1:
                                                    greenpixels_intensity_flag = 1
                                                    greenpixels_intensity_count = 0
                                                elif redopt_pickindex <= redpixels_len - 1:
                                                    redpixels_intensity_flag = 1
                                                    redpixels_intensity_count = 0
                                                else:
                                                    redpixels_prob = redpixels_prob_temp
                                                    greenpixels_prob = greenpixels_prob_temp
                                                    bluepixels_prob = bluepixels_prob_temp
                                                    redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                    greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                    blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                    redopt_pickindex = 0
                                                    greenopt_pickindex = 0
                                                    blueopt_pickindex = 0

                                                    greenpixels_intensity_flag = 1
                                                    greenpixels_intensity_count = 0

                                        elif redopt_pickindex > redpixels_len - 1 and diff_flag:
                                            redpixels_intensity_flag = 0
                                            redpixels_intensity_count = 0
                                            if greenopt_pickindex <= greenpixels_len - 1:
                                                greenpixels_intensity_flag = 1
                                                greenpixels_intensity_count = 0
                                            else:
                                                redpixels_prob = redpixels_prob_temp
                                                greenpixels_prob = greenpixels_prob_temp
                                                bluepixels_prob = bluepixels_prob_temp
                                                redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                redopt_pickindex = 0
                                                greenopt_pickindex = 0
                                                blueopt_pickindex = 0

                                                greenpixels_intensity_flag = 1
                                                greenpixels_intensity_count = 0

                                        diff_flag  = 1

                                        if type(interfer_pixel) is not np.ndarray:
                                            interfer_pixel = interfer_pixel.numpy()
                                        if (np.all(interfer_pixel[pickinterfer_greenopt_index[
                                                                  0:greenpixels_len]] > (
                                                           max_brightness + step_size)) and np.all(
                                            interfer_pixel[
                                                pickinterfer_redopt_index[
                                                0:redpixels_len]] > (
                                                    max_brightness + step_size))):
                                            location_flag = 0







                                    else:

                                        if greenopt_pickindex <= greenpixels_len - 1:
                                            greenopt_pickindex += 1
                                            if greenopt_pickindex <= greenpixels_len - 1:
                                                greenopt_pixel_mask_index = np.delete(greenopt_pixel_mask_index,
                                                                                      np.argmin(greenpixels_prob))
                                                greenpixels_prob = np.delete(greenpixels_prob,
                                                                             np.argmin(greenpixels_prob))


                                        if greenopt_pickindex <= greenpixels_len - 1:
                                            greenpixels_intensity_flag = 1
                                            greenpixels_intensity_count = 0
                                        else:
                                            redpixels_prob = redpixels_prob_temp
                                            greenpixels_prob = greenpixels_prob_temp
                                            bluepixels_prob = bluepixels_prob_temp
                                            redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                            greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                            blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                            redopt_pickindex = 0
                                            greenopt_pickindex = 0
                                            blueopt_pickindex = 0

                                            greenpixels_intensity_flag = 1
                                            greenpixels_intensity_count = 0
                                            # location_flag = 0

                                        if type(interfer_pixel) is not np.ndarray:
                                            interfer_pixel = interfer_pixel.numpy()

                                        if np.all(interfer_pixel[pickinterfer_greenopt_index[
                                                                  0:greenpixels_len]] > (
                                                           max_brightness + step_size)):
                                            location_flag = 0





                            elif (blue_flag == 1):

                                if (bluepixels_intensity_flag):  # greenpixels_count <= greenpixels_len
                                    if bluepixels_intensity_count == 0:
                                        # interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                        # interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                        #                                     interfer_pixel_red_index,
                                        #                                     interfer_pixel_green_index,
                                        #                                     interfer_pixel_blue_index)
                                        # picked optimal green location
                                        interfer_pixel = interfer_pixel.numpy()
                                        pickinterfer_blueopt_index[blueopt_pickindex] = \
                                            np.where(interfer_pixel_index == blueopt_pixel_mask_index[
                                                np.argmin(bluepixels_prob)])[0]
                                        interfer_pixel[
                                            pickinterfer_blueopt_index[blueopt_pickindex]] += ini_intensity

                                    if bluepixels_intensity_count > 0:
                                        interfer_pixel = interfer_pixel.numpy()
                                        interfer_pixel[pickinterfer_blueopt_index[blueopt_pickindex]] += step_size
                                        bluepixels_intensity_prob[bluepixels_intensity_count - 1] = judge_targety_prob
                                        if bluepixels_intensity_count > 1:
                                            bluepixels_intensity_diff[bluepixels_intensity_count - 2] = \
                                            bluepixels_intensity_prob[bluepixels_intensity_count - 1] - \
                                            bluepixels_intensity_prob[bluepixels_intensity_count - 2]
                                        if bluepixels_intensity_count > 2:
                                            if bluepixels_intensity_diff[bluepixels_intensity_count - 2] - \
                                                    bluepixels_intensity_diff[bluepixels_intensity_count - 3] < probdiff_th:
                                                diff_flag = 0
                                                bluepixels_intensity_flag = 0
                                                if (red_flag == 2) and (redopt_pickindex <= redpixels_len - 1):
                                                    redpixels_intensity_flag = 1
                                                    redpixels_intensity_count = 0
                                                else:

                                                    if blueopt_pickindex < bluepixels_len - 1:
                                                        blueopt_pickindex += 1
                                                        bluepixels_intensity_flag = 1
                                                        bluepixels_intensity_count = -1
                                                        blueopt_pixel_mask_index = np.delete(blueopt_pixel_mask_index,
                                                                                             np.argmin(
                                                                                                 bluepixels_prob))
                                                        bluepixels_prob = np.delete(bluepixels_prob,
                                                                                    np.argmin(bluepixels_prob))
                                                    else:
                                                        if (np.all(interfer_pixel[pickinterfer_blueopt_index[
                                                                                 0:bluepixels_len]] > (
                                                                          max_brightness + step_size)) and np.all(
                                                                interfer_pixel[
                                                                    pickinterfer_redopt_index[
                                                                    0:redpixels_len]] > (
                                                                        max_brightness + step_size))):
                                                            location_flag = 0

                                                        else:

                                                            redpixels_prob = redpixels_prob_temp
                                                            greenpixels_prob = greenpixels_prob_temp
                                                            bluepixels_prob = bluepixels_prob_temp
                                                            redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                            greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                            blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                            redopt_pickindex = 0
                                                            greenopt_pickindex = 0
                                                            blueopt_pickindex = 0

                                                            bluepixels_intensity_flag = 1
                                                            bluepixels_intensity_count = -1

                                                        # location_flag = 0

                                    bluepixels_intensity_count += 1

                                    if blueopt_pickindex <= bluepixels_len - 1 and diff_flag:
                                        if np.all(interfer_pixel[pickinterfer_blueopt_index[blueopt_pickindex]] > (
                                                max_brightness + step_size)):
                                            bluepixels_intensity_flag = 0
                                            if (red_flag == 2) and (redopt_pickindex <= redpixels_len - 1):
                                                redpixels_intensity_flag = 1
                                                redpixels_intensity_count = 0
                                            else:

                                                if blueopt_pickindex <= bluepixels_len - 1:
                                                    blueopt_pickindex += 1
                                                    if blueopt_pickindex <= bluepixels_len - 1:
                                                        bluepixels_intensity_flag = 1
                                                        bluepixels_intensity_count = 0
                                                        blueopt_pixel_mask_index = np.delete(blueopt_pixel_mask_index,
                                                                                             np.argmin(
                                                                                                 bluepixels_prob))
                                                        bluepixels_prob = np.delete(bluepixels_prob,
                                                                                    np.argmin(bluepixels_prob))

                                                else:
                                                    redpixels_prob = redpixels_prob_temp
                                                    greenpixels_prob = greenpixels_prob_temp
                                                    bluepixels_prob = bluepixels_prob_temp
                                                    redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                    greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                    blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                    redopt_pickindex = 0
                                                    greenopt_pickindex = 0
                                                    blueopt_pickindex = 0

                                                    bluepixels_intensity_flag = 1
                                                    bluepixels_intensity_count = 0

                                    elif blueopt_pickindex > bluepixels_len - 1 and diff_flag:
                                        bluepixels_intensity_flag = 0
                                        bluepixels_intensity_count = 0
                                        if (red_flag == 2) and (redopt_pickindex <= redpixels_len - 1):
                                            redpixels_intensity_flag = 1
                                            redpixels_intensity_count = 0
                                        else:
                                            redpixels_prob = redpixels_prob_temp
                                            greenpixels_prob = greenpixels_prob_temp
                                            bluepixels_prob = bluepixels_prob_temp
                                            redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                            greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                            blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                            redopt_pickindex = 0
                                            greenopt_pickindex = 0
                                            blueopt_pickindex = 0

                                            bluepixels_intensity_flag = 1
                                            bluepixels_intensity_count = 0

                                    diff_flag = 1





                                else:
                                    if (red_flag == 2):
                                        if (redpixels_intensity_flag):

                                            if redpixels_intensity_count == 0:
                                                pickinterfer_redopt_index[redopt_pickindex] = \
                                                    np.where(interfer_pixel_index == redopt_pixel_mask_index[
                                                        np.argmin(np.absolute(
                                                            redopt_pixel_mask_index - blueopt_pixel_mask_index[
                                                                np.argmin(bluepixels_prob)]))])[0]

                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[
                                                    pickinterfer_redopt_index[redopt_pickindex]] += ini_intensity
                                            if redpixels_intensity_count > 0:
                                                interfer_pixel = interfer_pixel.numpy()
                                                interfer_pixel[
                                                    pickinterfer_redopt_index[redopt_pickindex]] += step_size
                                                redpixels_intensity_prob[
                                                    redpixels_intensity_count - 1] = judge_targety_prob
                                                if redpixels_intensity_count > 1:
                                                    redpixels_intensity_diff[redpixels_intensity_count - 2] = \
                                                        redpixels_intensity_prob[redpixels_intensity_count - 1] - \
                                                        redpixels_intensity_prob[redpixels_intensity_count - 2]
                                                if redpixels_intensity_count > 2:
                                                    if redpixels_intensity_diff[redpixels_intensity_count - 2] - \
                                                            redpixels_intensity_diff[
                                                                redpixels_intensity_count - 3] < probdiff_th:

                                                        diff_flag = 0

                                                        if redopt_pickindex <= redpixels_len - 1:
                                                            redopt_pickindex += 1
                                                            if redopt_pickindex <= redpixels_len - 1:
                                                                redopt_pixel_mask_index = np.delete(
                                                                    redopt_pixel_mask_index, np.argmin(np.absolute(
                                                                        redopt_pixel_mask_index -
                                                                        blueopt_pixel_mask_index[
                                                                            np.argmin(bluepixels_prob)])))


                                                        if blueopt_pickindex <= bluepixels_len - 1:
                                                            blueopt_pickindex += 1
                                                            if blueopt_pickindex <= bluepixels_len - 1:
                                                                blueopt_pixel_mask_index = np.delete(
                                                                    blueopt_pixel_mask_index,
                                                                    np.argmin(bluepixels_prob))
                                                                bluepixels_prob = np.delete(bluepixels_prob,
                                                                                            np.argmin(bluepixels_prob))


                                                        redpixels_intensity_flag = 0
                                                        if blueopt_pickindex <= bluepixels_len - 1:
                                                            bluepixels_intensity_flag = 1
                                                            bluepixels_intensity_count = 0
                                                        elif redopt_pickindex <= redpixels_len - 1:
                                                            redpixels_intensity_flag = 1
                                                            redpixels_intensity_count = -1
                                                        else:
                                                            if (np.all(interfer_pixel[pickinterfer_blueopt_index[
                                                                                      0:bluepixels_len]] > (
                                                                               max_brightness + step_size)) and np.all(
                                                                interfer_pixel[
                                                                    pickinterfer_redopt_index[
                                                                    0:redpixels_len]] > (
                                                                        max_brightness + step_size))):
                                                                location_flag = 0

                                                            else:

                                                                redpixels_prob = redpixels_prob_temp
                                                                greenpixels_prob = greenpixels_prob_temp
                                                                bluepixels_prob = bluepixels_prob_temp
                                                                redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                                greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                                blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                                redopt_pickindex = 0
                                                                greenopt_pickindex = 0
                                                                blueopt_pickindex = 0

                                                                bluepixels_intensity_flag = 1
                                                                bluepixels_intensity_count = 0



                                            redpixels_intensity_count += 1

                                        if redopt_pickindex <= redpixels_len - 1 and diff_flag:
                                            if interfer_pixel[pickinterfer_redopt_index[redopt_pickindex]] > (
                                                    max_brightness + step_size):

                                                if redopt_pickindex <= redpixels_len - 1:
                                                    redopt_pickindex += 1
                                                    if redopt_pickindex <= redpixels_len - 1:
                                                        redopt_pixel_mask_index = np.delete(redopt_pixel_mask_index,
                                                                                            np.argmin(np.absolute(
                                                                                                redopt_pixel_mask_index -
                                                                                                blueopt_pixel_mask_index[
                                                                                                    np.argmin(
                                                                                                        bluepixels_prob)])))


                                                if blueopt_pickindex <= bluepixels_len - 1:
                                                    blueopt_pickindex += 1
                                                    if blueopt_pickindex <= bluepixels_len - 1:
                                                        blueopt_pixel_mask_index = np.delete(blueopt_pixel_mask_index,
                                                                                             np.argmin(bluepixels_prob))
                                                        bluepixels_prob = np.delete(bluepixels_prob,
                                                                                    np.argmin(bluepixels_prob))


                                                redpixels_intensity_flag = 0
                                                if blueopt_pickindex <= bluepixels_len - 1:
                                                    bluepixels_intensity_flag = 1
                                                    bluepixels_intensity_count = 0
                                                elif redopt_pickindex <= redpixels_len - 1:
                                                    redpixels_intensity_flag = 1
                                                    redpixels_intensity_count = 0
                                                else:
                                                    redpixels_prob = redpixels_prob_temp
                                                    greenpixels_prob = greenpixels_prob_temp
                                                    bluepixels_prob = bluepixels_prob_temp
                                                    redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                    greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                    blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                    redopt_pickindex = 0
                                                    greenopt_pickindex = 0
                                                    blueopt_pickindex = 0

                                                    bluepixels_intensity_flag = 1
                                                    bluepixels_intensity_count = 0

                                        elif redopt_pickindex > redpixels_len - 1 and diff_flag:
                                            redpixels_intensity_flag = 0
                                            redpixels_intensity_count = 0
                                            if blueopt_pickindex <= bluepixels_len - 1:
                                                bluepixels_intensity_flag = 1
                                                bluepixels_intensity_count = 0
                                            else:
                                                redpixels_prob = redpixels_prob_temp
                                                greenpixels_prob = greenpixels_prob_temp
                                                bluepixels_prob = bluepixels_prob_temp
                                                redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                redopt_pickindex = 0
                                                greenopt_pickindex = 0
                                                blueopt_pickindex = 0

                                                bluepixels_intensity_flag = 1
                                                bluepixels_intensity_count = 0

                                        diff_flag = 1

                                        if type(interfer_pixel) is not np.ndarray:
                                            interfer_pixel = interfer_pixel.numpy()




                                        if (np.all(interfer_pixel[pickinterfer_blueopt_index[
                                                                  0:bluepixels_len]] > (
                                                           max_brightness + step_size)) and np.all(
                                            interfer_pixel[
                                                pickinterfer_redopt_index[
                                                0:redpixels_len]] > (
                                                    max_brightness + step_size))):
                                            location_flag = 0






                                    else:

                                        if blueopt_pickindex <= bluepixels_len - 1:
                                            blueopt_pickindex += 1
                                            if blueopt_pickindex <= bluepixels_len - 1:
                                                blueopt_pixel_mask_index = np.delete(blueopt_pixel_mask_index,
                                                                                     np.argmin(bluepixels_prob))
                                                bluepixels_prob = np.delete(bluepixels_prob,
                                                                            np.argmin(bluepixels_prob))


                                        if blueopt_pickindex <= bluepixels_len - 1:
                                            bluepixels_intensity_flag = 1
                                            bluepixels_intensity_count = 0
                                        else:
                                            redpixels_prob = redpixels_prob_temp
                                            greenpixels_prob = greenpixels_prob_temp
                                            bluepixels_prob = bluepixels_prob_temp
                                            redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                            greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                            blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                            redopt_pickindex = 0
                                            greenopt_pickindex = 0
                                            blueopt_pickindex = 0

                                            bluepixels_intensity_flag = 1
                                            bluepixels_intensity_count = 0

                                        if type(interfer_pixel) is not np.ndarray:
                                            interfer_pixel = interfer_pixel.numpy()

                                        if np.all(interfer_pixel[pickinterfer_blueopt_index[
                                                                  0:bluepixels_len]] > (
                                                           max_brightness + step_size)):
                                            location_flag = 0









                            elif (red_flag == 1):

                                if (redpixels_intensity_flag):  # greenpixels_count <= greenpixels_len
                                    if redpixels_intensity_count == 0:
                                        # interfer_pixel = np.zeros(np.shape(interfer_pixel_index), dtype=np.float32)
                                        # interfer_pixel = set_interfer_pixel(interfer_pixel, colorstatusbar_flag,
                                        #                                     interfer_pixel_red_index,
                                        #                                     interfer_pixel_green_index,
                                        #                                     interfer_pixel_blue_index)
                                        # picked optimal green location
                                        interfer_pixel = interfer_pixel.numpy()
                                        pickinterfer_redopt_index[redopt_pickindex] = \
                                            np.where(interfer_pixel_index == redopt_pixel_mask_index[
                                                np.argmin(redpixels_prob)])[0]
                                        interfer_pixel[
                                            pickinterfer_redopt_index[redopt_pickindex]] += ini_intensity

                                    if redpixels_intensity_count > 0:
                                        interfer_pixel = interfer_pixel.numpy()
                                        interfer_pixel[pickinterfer_redopt_index[redopt_pickindex]] += step_size
                                        redpixels_intensity_prob[redpixels_intensity_count - 1] = judge_targety_prob
                                        if redpixels_intensity_count > 1:
                                            redpixels_intensity_diff[redpixels_intensity_count - 2] = \
                                            redpixels_intensity_prob[redpixels_intensity_count - 1] - \
                                            redpixels_intensity_prob[redpixels_intensity_count - 2]
                                        if redpixels_intensity_count > 2:
                                            if redpixels_intensity_diff[redpixels_intensity_count - 2] - \
                                                    redpixels_intensity_diff[redpixels_intensity_count - 3] < probdiff_th:
                                                redpixels_intensity_flag = 0

                                    if redopt_pickindex <= redpixels_len - 1:
                                        if np.all(interfer_pixel[pickinterfer_redopt_index[redopt_pickindex]] > (
                                                max_brightness + step_size)):
                                            if redopt_pickindex <= redpixels_len - 1:
                                                redpixels_intensity_flag = 0
                                            else:
                                                if np.all(interfer_pixel[pickinterfer_redopt_index[0:redpixels_len]] > (
                                                        max_brightness + step_size)):
                                                    location_flag = 0

                                                else:

                                                    redpixels_prob = redpixels_prob_temp
                                                    greenpixels_prob = greenpixels_prob_temp
                                                    bluepixels_prob = bluepixels_prob_temp
                                                    redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                                    greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                                    blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                                    redopt_pickindex = 0
                                                    greenopt_pickindex = 0
                                                    blueopt_pickindex = 0

                                                    redpixels_intensity_flag = 1
                                                    redpixels_intensity_count = -1
                                                # location_flag = 0




                                    redpixels_intensity_count += 1





                                else:


                                    if redopt_pickindex <= redpixels_len - 1:
                                        redopt_pickindex += 1
                                        if redopt_pickindex <= redpixels_len - 1:
                                            redopt_pixel_mask_index = np.delete(redopt_pixel_mask_index,
                                                                                np.argmin(redpixels_prob))
                                            redpixels_prob = np.delete(redpixels_prob,
                                                                       np.argmin(redpixels_prob))


                                    if redopt_pickindex <= redpixels_len - 1:
                                        redpixels_intensity_flag = 1
                                        redpixels_intensity_count = 0
                                    else:
                                        redpixels_prob = redpixels_prob_temp
                                        greenpixels_prob = greenpixels_prob_temp
                                        bluepixels_prob = bluepixels_prob_temp
                                        redopt_pixel_mask_index = redopt_pixel_mask_index_temp
                                        greenopt_pixel_mask_index = greenopt_pixel_mask_index_temp
                                        blueopt_pixel_mask_index = blueopt_pixel_mask_index_temp
                                        redopt_pickindex = 0
                                        greenopt_pickindex = 0
                                        blueopt_pickindex = 0

                                        redpixels_intensity_flag = 1
                                        redpixels_intensity_count = 0
                                        # location_flag = 0

                                    if type(interfer_pixel) is not np.ndarray:
                                        interfer_pixel = interfer_pixel.numpy()

                                    if np.all(interfer_pixel[pickinterfer_redopt_index[0:redpixels_len]] > (
                                            max_brightness + step_size)):
                                        location_flag = 0





                            else:

                                location_flag = 0







                        else:
                            location_flag = 0

                    # plt.imshow(redopt_pixel_pick_mask)
                    # plt.show()
                    # plt.imshow(greenopt_pixel_pick_mask)
                    # plt.show()
                    # plt.imshow(blueopt_pixel_pick_mask)
                    # plt.show()

                    # visualization = show_cam_on_image(captured_img.astype(dtype=np.float32) / 255.,
                    #                                   grayscale_cam,
                    #                                   use_rgb=True)

                    # plt.imshow(visualization)
                    # plt.show()

                    # compute losses
                    # loss, transfer_funcs = criterion(mapped_pattern, PSFs, PSFs_RGB, deconved, batch_img, epoch=epoch_id)
                    loss, _ = criterion(mapped_pattern, PSFs, PSFs_RGB, ssim_mean, deconved, batch_img_temp, epoch=epoch_id)

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
            # # image_id = image_id + 1

            for iiii in range(batch_size):
                visuals = {}
                # visuals['captured_0'] = vis.tensor_to_img_save(captured, iiii)  # captured image in current batch
                # visuals['deconved_0'] = vis.tensor_to_img_save(deconved, iiii)  # deblurred image in current batch
                # visuals['captured_0'] = captured_img  # captured image in current batch
                # visuals['deconved_0'] = deconved_img  # deblurred image in current batch
                if opt.save_mode == 'all':
                    visuals['captured_0_all'] = vis.tensor_to_img_save(captured_temp, iiii)  # Save original size image
                    visuals['deconved_0_all'] = vis.tensor_to_img_save(deconved_temp, iiii)  # Save original size image
                elif opt.save_mode == 'crop':
                    visuals['captured_0_crop'] = captured_img  # Save cropped size image
                    visuals['deconved_0_crop'] = deconved_img  # Save cropped size image
                elif opt.save_mode == 'both':
                    visuals['captured_0_all'] = vis.tensor_to_img_save(captured_temp, iiii)  # Save original size image
                    visuals['deconved_0_all'] = vis.tensor_to_img_save(deconved_temp, iiii)  # Save original size image
                    visuals['captured_0_crop'] = captured_img  # Save cropped size image
                    visuals['deconved_0_crop'] = deconved_img  # Save cropped size image

                vis.display_current_results(visuals, image_id, file_name)
                image_id = image_id + 1

            # plot curves
            # sz = tf.shape(PSFs_RGB).numpy()[0]
            # vis.plot_current_curve(PSFs_RGB[int(sz / 2), :, :].numpy(), 'PSFs_RGB', display_id=10)  # a slice of PSF (ideally a Dirac delta function)
            # vis.plot_current_curve(transfer_funcs[int(sz/2), :, :].numpy(), 'Transfer function', display_id=15)
            #                                                                      # a slice of transfer functions (ideally all-ones)
            # vis.plot_current_curve(avg_losses, 'Total loss', display_id=9)   # losses

            # print losses to log file
            vis.print_current_loss(img_number, image_id, loss, logfile)
            print('Duration:{:.2f}'.format(time.time() - batch_start_time))

            _, img_ori_height, img_ori_width, _ = tf.shape(batch_img)
            if img_ori_height > DOR_height * 3 or img_ori_width > DOR_width * 3:
                _, _, _, u1_RGB_real_mask = camera(None, DoR_option='small', mode_option='preprocessing')
                u1_RGB_real_mask_temp = np.array(u1_RGB_real_mask)
                # height_temp, width_temp = np.shape(u1_RGB_real_mask_temp)
                # u1_RGB_real_mask_temp = crop_image(u1_RGB_real_mask_temp, height_temp / 6, width_temp / 6)

                all_interfer_pixel_index = np.unique(u1_RGB_real_mask_temp)
                all_interfer_pixel_index = all_interfer_pixel_index[all_interfer_pixel_index != 0]
                all_interfer_pixel_index = all_interfer_pixel_index.astype(int)
                all_interfer_pixel_trans_index = np.zeros_like(all_interfer_pixel_index)

                all_interfer_pixel_red_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
                all_interfer_pixel_green_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
                all_interfer_pixel_blue_index = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)

                for ii in range(len(all_interfer_pixel_index)):
                    if all_interfer_pixel_index[ii] % 3 == 2:
                        all_interfer_pixel_trans_index[ii] = 2  # Red
                        all_interfer_pixel_red_index[ii] = max_sum_power  # power for red pixel
                    elif all_interfer_pixel_index[ii] % 3 == 0:
                        all_interfer_pixel_trans_index[ii] = 3  # Green
                        all_interfer_pixel_green_index[ii] = max_sum_power  # power for red pixel
                    elif all_interfer_pixel_index[ii] % 3 == 1:
                        all_interfer_pixel_trans_index[ii] = 4  # Blue
                        all_interfer_pixel_blue_index[ii] = max_sum_power  # power for red pixel

                all_red_pixel_index = np.where(all_interfer_pixel_trans_index == 2)
                all_green_pixel_index = np.where(all_interfer_pixel_trans_index == 3)
                all_blue_pixel_index = np.where(all_interfer_pixel_trans_index == 4)

                all_Red_mask = np.zeros_like(u1_RGB_real_mask_temp)
                all_Green_mask = np.zeros_like(u1_RGB_real_mask_temp)
                all_Blue_mask = np.zeros_like(u1_RGB_real_mask_temp)

                for iii in range(len(all_red_pixel_index[0])):
                    all_Red_mask = np.where(
                        u1_RGB_real_mask_temp == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
                        all_interfer_pixel_index[all_red_pixel_index[0][iii]], all_Red_mask)

                for jjj in range(len(all_green_pixel_index[0])):
                    all_Green_mask = np.where(
                        u1_RGB_real_mask_temp == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
                        all_interfer_pixel_index[all_green_pixel_index[0][jjj]], all_Green_mask)

                for kkk in range(len(all_blue_pixel_index[0])):
                    all_Blue_mask = np.where(
                        u1_RGB_real_mask_temp == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
                        all_interfer_pixel_index[all_blue_pixel_index[0][kkk]], all_Blue_mask)

                # all_RGB_mask = np.stack([all_Red_mask, all_Green_mask, all_Blue_mask],
                #                         axis=2)

                all_display_red_mask_ori = all_Red_mask  # tf.cast(Red_mask)
                all_display_green_mask_ori = all_Green_mask  # tf.cast(Green_mask)
                all_display_blue_mask_ori = all_Blue_mask  # tf.cast(Blue_mask)

                all_interfer_pixel = np.zeros(np.shape(all_interfer_pixel_index), dtype=np.float32)
                all_interfer_pixel = tf.constant(all_interfer_pixel)

                if colorstatusbar_flag == 0:
                    all_interfer_pixel += all_interfer_pixel_red_index
                elif colorstatusbar_flag == 1:
                    all_interfer_pixel += all_interfer_pixel_green_index
                elif colorstatusbar_flag == 2:
                    all_interfer_pixel += all_interfer_pixel_blue_index
                elif colorstatusbar_flag == 3:
                    all_interfer_pixel += all_interfer_pixel_red_index / 3
                    all_interfer_pixel += all_interfer_pixel_green_index / 3
                    all_interfer_pixel += all_interfer_pixel_blue_index / 3

                for iii in range(len(all_red_pixel_index[0])):
                    all_display_red_mask_ori = tf.where(
                        all_display_red_mask_ori == all_interfer_pixel_index[all_red_pixel_index[0][iii]],
                        all_interfer_pixel[all_red_pixel_index[0][iii]],
                        all_display_red_mask_ori)  # mapped_pixel[iii], display_red_mask)
                for jjj in range(len(all_green_pixel_index[0])):
                    all_display_green_mask_ori = tf.where(
                        all_display_green_mask_ori == all_interfer_pixel_index[all_green_pixel_index[0][jjj]],
                        all_interfer_pixel[all_green_pixel_index[0][jjj]],
                        all_display_green_mask_ori)  # mapped_pixel[len(red_pixel_index[0]) + jjj], display_green_mask)
                for kkk in range(len(all_blue_pixel_index[0])):
                    all_display_blue_mask_ori = tf.where(
                        all_display_blue_mask_ori == all_interfer_pixel_index[all_blue_pixel_index[0][kkk]],
                        all_interfer_pixel[all_blue_pixel_index[0][kkk]], all_display_blue_mask_ori)

                all_display_RGB_mask_ori = np.stack(
                    [all_display_red_mask_ori, all_display_green_mask_ori, all_display_blue_mask_ori],
                    axis=2)

                opt.itini_intensity = itini_intensity_temp
                opt.itstep_size = itstep_size_temp

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
    # plt.imshow(all_display_RGB_mask_ssim)
    # plt.show()

    return mapped_pixel, mapped_all_pixel, ssim_min


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # optimization options
    parser.add_argument('--tile_option', type=str, default='repeat', help='pixel tiling methods [repeat|randomRot]')
    # parser.add_argument('--use_data', action='store_true', help='use data-driven loss top-10 L2')
    # parser.add_argument('--invertible', action='store_true', help='use PSF-induced loss L_inv')

    parser.add_argument('--area', type=float, default=0.20, help='target pixel opening ratio 0~1')
    parser.add_argument('--area_gamma', type=float, default=10, help='area constraint weight')
    parser.add_argument('--l2_gamma', type=float, default=0.5, help='top-10 L2 loss weight')  # 10
    parser.add_argument('--inv_gamma', type=float, default=0.5, help='L_inv loss weight')  # 0.01

    parser.add_argument('--log_dir', type=str, default='log/', help='save optimized pattern and training log')
    parser.add_argument('--isTrain', action='store_true', help='train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # 1

    # display options
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_port', type=int, default=8997, help='visdom port of the web display')  # 8999 8097
    parser.add_argument('--display_env', type=str, default='VIS_NAME',
                        help='visdom environment of the web display')  # main VIS_NAME
    parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--checkpoints_dir', type=str, default='logs/')

    # optimization parameters
    parser.add_argument('--save_cap_dir', type=str, default='cap_allcolorbar_intensity_test',
                        help='save perturbed images')  # cap_allcolorbar_p10_v1
    parser.add_argument('--save_dec_dir', type=str, default='dec_allcolorbar_intensity_test',
                        help='save deblurred images')  # dec_allcolorbar_p10_v1
    parser.add_argument('--save_temp_dir', type=str, default='allcolor_intensity_process_temp',
                        help='save processing temp images')
    parser.add_argument('--images_mode', type=str, default='images', help='images mode: images/images_test')
    parser.add_argument('--statusbarcolor_flag', type=int, default=3, help='statusbar color flag')
    parser.add_argument('--itstep_size', type=float, default=0.1, help='Iteration step size')
    parser.add_argument('--itini_intensity', type=float, default=0.5, help='Iteration initial intensity')
    parser.add_argument('--maxperturbation_power', type=float, default=10, help='max perturbation_power')
    parser.add_argument('--maxscreen_brightness', type=float, default=2, help='max screen_brightness')
    parser.add_argument('--probdiff_threshold', type=float, default=0, help='max screen_brightness')
    parser.add_argument('--pretrained', type=str, default='resnet18', help='pretrained network model')
    parser.add_argument('--save_mode', type=str, default='both', help='both, all or crop')
    # parser.add_argument('--gpu_flag', type=int, default=1, help='statusbar color flag')

    opt = parser.parse_args()
    opt.no_html = False
    opt.isTrain = True
    opt.use_data = True
    opt.invertible = True

    start = time.time()

    mapped_pixel, mapped_all_pixel, ssim_min = optimize_pattern_with_data(opt)

    end = time.time()
    print('total times:', (end - start))
    print('avg times:', (end - start) / 1000)
    print(str(end - start))
    print(opt.save_cap_dir)
    print(opt.pretrained)

    # optimize_pattern_with_data(opt)

    # python optimize_display.py --tile_option repeat --area_gamma 10 --l2_gamma 10 --inv_gamma 0.01 --display_env VIS_NAME
