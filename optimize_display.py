import os
from os.path import join
import numpy as np
import scipy.io as sio
import cv2
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from util.visualizer import Visualizer
from PIL import Image
from skimage.feature import blob_dog, blob_log, blob_doh

from wave_optics import Camera, set_params, capture, wiener_deconv, get_interfer_img, get_wiener_loss
from loss import Loss
from utils import print_opt, crop_image

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done # run this in command

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
img_height = 512
img_width = 512
# tf.keras.backend.set_floatx('float32')


def load_data():

    # base_dir = os.path.dirname(os.path.dirname(__file__))
    # # 获取当前文件目录
    # data_path = os.path.abspath(os.path.join(base_dir, 'Train/Poled/HQ/*.png', ""))
    # # 改为绝对路径
    # # 获取文件拼接后的路径
    # D: / Dropbox / TuD work / ScreenAI_Privacy_Underscreen / UPC_ICCP21_Code - main / Train / Poled / HQ / *.png

    # todo: change path to your training image directory
    train_ds = tf.data.Dataset.list_files('Train/Poled/HQ/*.png', shuffle=False)
    # train_ds = tf.data.Dataset.list_files(data_path, shuffle=False)
    train_ds = train_ds.shuffle(240, reshuffle_each_iteration=True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def resize_and_rescale(img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img)
        img = tf.cast(img, dtype=tf.float32) / 255

        # data augmentation
        img = tf.image.random_crop(img, size=tf.constant([img_height, img_width, 3]))
        return img

    train_ds = train_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    return train_ds

def load_pattern():
    pattern_path = 'data/pixelPatterns/POLED_32.png' #POLED_42 300PPI POLED_32 400DPI
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
    pattern = load_pattern()

    # set up camera
    cameraOpt = set_params()
    camera = Camera(pattern)


    # set up optimization
    optimizer = keras.optimizers.Adam(learning_rate=opt.lr, beta_1=0.9)#lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    criterion = Loss(opt, cameraOpt)
    vars = []               # variables which we track gradient
    train_ds = load_data()  # load training dataset

    # initial pixel opening as all-open
    # pattern = np.ones((21, 21), dtype=np.float32)
    # pattern = tf.Variable(initial_value=(pattern * 2 - 1), trainable=True)  # later we use sigmoid to map 0 to 1.

    # vars.append(pattern)

    _, _, _, u1_RGB_real_mask = camera(None, mode_option = 'preprocessing')
    u1_RGB_real_mask_temp = np.array(u1_RGB_real_mask)
    height_temp, width_temp = np.shape(u1_RGB_real_mask_temp)
    u1_RGB_real_mask_temp = crop_image(u1_RGB_real_mask_temp, height_temp / 6, width_temp / 6)
    display_interference_pre = crop_image(cv2.resize(u1_RGB_real_mask_temp, None, fx=1, fy=1, interpolation=cv2.INTER_AREA), img_height, img_width) #INTER_LINEAR INTER_AREA

    interfer_pixel_index = np.unique(display_interference_pre)
    interfer_pixel_index = interfer_pixel_index[interfer_pixel_index != 0]
    interfer_pixel_index = interfer_pixel_index.astype(int)
    interfer_pixel_trans_index = np.zeros_like(interfer_pixel_index)

    all_interfer_pixel_index = np.unique(u1_RGB_real_mask_temp)
    all_interfer_pixel_index = all_interfer_pixel_index[all_interfer_pixel_index != 0]
    all_interfer_pixel_index = all_interfer_pixel_index.astype(int)
    # all_interfer_pixel_trans_index = np.zeros_like(all_interfer_pixel_index)

    for ii in range(len(interfer_pixel_index)):
        if interfer_pixel_index[ii] % 3 == 2:
            interfer_pixel_trans_index[ii] = 2
        elif interfer_pixel_index[ii] % 3 == 0:
            interfer_pixel_trans_index[ii] = 3
        elif interfer_pixel_index[ii] % 3 == 1:
            interfer_pixel_trans_index[ii] = 4

    red_pixel_index = np.where(interfer_pixel_trans_index == 2)
    green_pixel_index = np.where(interfer_pixel_trans_index == 3)
    blue_pixel_index = np.where(interfer_pixel_trans_index == 4)

    Red_mask = np.zeros_like(u1_RGB_real_mask_temp)
    Green_mask = np.zeros_like(u1_RGB_real_mask_temp)
    Blue_mask = np.zeros_like(u1_RGB_real_mask_temp)

    for iii in range(len(red_pixel_index[0])):
        Red_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[red_pixel_index[0][iii]], interfer_pixel_index[red_pixel_index[0][iii]], Red_mask)

    for jjj in range(len(green_pixel_index[0])):
        Green_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[green_pixel_index[0][jjj]], interfer_pixel_index[green_pixel_index[0][jjj]], Green_mask)

    for kkk in range(len(blue_pixel_index[0])):
        Blue_mask = np.where(u1_RGB_real_mask_temp == interfer_pixel_index[blue_pixel_index[0][kkk]], interfer_pixel_index[blue_pixel_index[0][kkk]], Blue_mask)



    interfer_pixel = np.ones(np.shape(interfer_pixel_index), dtype=np.float32)
    interfer_pixel = tf.Variable(initial_value=(interfer_pixel * 2 - 1), trainable=True)  # later we use sigmoid to map 0 to 1

    all_interfer_pixel = np.ones(np.shape(all_interfer_pixel_index), dtype=np.float32)
    all_interfer_pixel = tf.constant(all_interfer_pixel)
    # all_interfer_pixel = tf.Variable(initial_value=(all_interfer_pixel * 2 - 1), trainable=True)  # later we use sigmoid to map 0 to 1


    vars.append(interfer_pixel)
    # vars.append(all_interfer_pixel)


    # blobs_log = blob_log(img_new, max_sigma=30, num_sigma=10, threshold=.1)
    # blobs_log = blob_doh(img_new)
    plt.imshow(display_interference_pre)
    plt.show()

    losses = []         # loss for each batch
    avg_losses = []     # smoothed losses
    best_loss = 1e18
    epochs = 150 # 150
    for epoch_id in range(epochs):

        for batch_id, batch_img in enumerate(train_ds):
            with tf.GradientTape() as g:
                # map pattern values to range [0, 1]
                # mapped_pattern = tf.sigmoid(pattern)
                mapped_pattern = pattern/255
                mapped_pixel = tf.sigmoid(interfer_pixel)
                # mapped_all_pixel = tf.sigmoid(all_interfer_pixel)
                mapped_all_pixel = all_interfer_pixel
                mapped_u1_RGB_real_mask = u1_RGB_real_mask
                # display_red_mask = Red_mask
                # display_green_mask = Green_mask
                # display_blue_mask = Blue_mask

                for ii in range(len(interfer_pixel_index)):
                    mapped_all_pixel = tf.where(all_interfer_pixel_index == interfer_pixel_index[ii], mapped_pixel[ii], mapped_all_pixel)

                for iii in range(len(all_interfer_pixel_index)):
                    mapped_u1_RGB_real_mask = tf.where(u1_RGB_real_mask == tf.constant(all_interfer_pixel_index[iii], dtype=tf.float32), mapped_all_pixel[iii], mapped_u1_RGB_real_mask)

                # plt.imshow(mapped_u1_RGB_real_mask)
                # plt.show()
                PSFs, PSFs_RGB, interfer_all_pixel, u1_RGB_real = camera(mapped_u1_RGB_real_mask, mode_option='running')



                interfer_img = get_interfer_img(Red_mask, Green_mask, Blue_mask, red_pixel_index, green_pixel_index, blue_pixel_index, interfer_pixel_index, mapped_pixel, batch_size, img_height, img_width)

                # test = tf.squeeze(interfer_img)
                # test = np.array(test)
                # plt.imshow(test)
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
                captured = capture(batch_img, interfer_img, PSFs, PSFs_RGB)
                deconved = wiener_deconv(captured, PSFs)

                # compute losses
                loss, transfer_funcs = criterion(mapped_pattern, PSFs, PSFs_RGB, deconved, batch_img, epoch=epoch_id)
                losses.append(loss['total_loss'].numpy())
                avg_losses.append(np.mean(losses[-10:]))  # average losses from the latest 10 epochs

            gradients = g.gradient(loss['total_loss'], vars)
            optimizer.apply_gradients(zip(gradients, vars))

            # visualization and log
            if batch_id % 50 == 0:
                # visualize images
                visuals = {}
                # visuals['pattern'] = (255*mapped_pattern[:,:,None]).numpy().astype(np.uint8)  # pixel opening pattern
                visuals['display_all_pattern'] = (255 * interfer_all_pixel).numpy().astype(np.uint8)  # pixel opening pattern
                visuals['display_interfer_pattern'] = (255 * interfer_img[0,:,:,:]).numpy().astype(np.uint8)  # pixel opening pattern
                visuals['PSFs_RGB'] = vis.tensor_to_img(tf.math.log(PSFs_RGB / tf.reduce_max(PSFs_RGB)))  # PSF in log-scale
                visuals['original_0'] = vis.tensor_to_img(batch_img)  # captured image in current batch
                visuals['captured_0'] = vis.tensor_to_img(captured)  # captured image in current batch
                visuals['deconved_0'] = vis.tensor_to_img(deconved)  # deblurred image in current batch
                vis.display_current_results(visuals, epoch_id)

                # plot curves
                sz = tf.shape(PSFs_RGB).numpy()[0]
                vis.plot_current_curve(PSFs_RGB[int(sz / 2), :, :].numpy(), 'PSFs_RGB', display_id=10)  # a slice of PSF (ideally a Dirac delta function)
                vis.plot_current_curve(transfer_funcs[int(sz/2), :, :].numpy(), 'Transfer function', display_id=15)
                                                                                 # a slice of transfer functions (ideally all-ones)
                vis.plot_current_curve(avg_losses, 'Total loss', display_id=9)   # losses

                # print losses to log file
                vis.print_current_loss(epochs, epoch_id, loss, logfile)

            if loss['total_loss'] < best_loss:
                best_loss = loss['total_loss']

        # save temporary results
        if epoch_id % 10 == 0:

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
            sio.savemat('%s/Mapped_pixel_imageinverse.mat' % log_dir, {'Mapped_pixel': mapped_pixel.numpy()})
            sio.savemat('%s/Mapped_all_pixel_imageinverse.mat' % log_dir, {'Mapped_all_pixel': mapped_all_pixel.numpy()})
            sio.savemat('%s/PSFs_RGB_imageinverse.mat' % log_dir, {'PSFs_RGB': PSFs_RGB.numpy()})
            sio.savemat('%s/avg_losses_imageinverse.mat' % log_dir, {'avg_losses': avg_losses})
            cv2.imwrite('%s/display_all_pattern_imageinverse.png' % log_dir, (255 * interfer_all_pixel).numpy())  # both ok for 2D and 3D
            cv2.imwrite('%s/display_interfer_pattern_imageinverse.png' % log_dir, (255 * interfer_img[0, :, :, :]).numpy())  # both ok for 2D and 3D

    logfile.close()

    return mapped_pixel, mapped_all_pixel


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

    opt = parser.parse_args()
    opt.no_html = False
    opt.isTrain = True
    opt.use_data = True
    opt.invertible = True

    mapped_pixel, mapped_all_pixel = optimize_pattern_with_data(opt)

    # python optimize_display.py --tile_option repeat --area_gamma 10 --l2_gamma 10 --inv_gamma 0.01 --display_env VIS_NAME
