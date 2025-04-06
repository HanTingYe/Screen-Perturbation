# %%

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior

import numpy as np
import torch

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# %%
# Function to mimic the 'fspecial' gaussian MATLAB function
def _tf_fspecial_gauss(size, sigma, channels=1):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

    window = g / tf.reduce_sum(g)
    return tf.tile(window, (1, 1, channels, channels))


# 计算ssim
def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img1.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    window = _tf_fspecial_gauss(size, sigma, ch)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Calculate the mean value Ux Uy and the mean square value Ux_sq in the slider
    padded_img1 = tf.pad(img1, [[0, 0], [size // 2, size // 2], [size // 2, size // 2], [0, 0]],
                         mode="CONSTANT")  # img1 fills zeros on top, bottom, left and right
    padded_img2 = tf.pad(img2, [[0, 0], [size // 2, size // 2], [size // 2, size // 2], [0, 0]],
                         mode="CONSTANT")  # img2 fills zeros on top, bottom, left and right
    mu1 = tf.nn.conv2d(padded_img1, window, strides=[1, 1, 1, 1], padding='VALID')  # Use a sliding window to obtain the weighted average of the image in the window
    mu2 = tf.nn.conv2d(padded_img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1  # img(x,y) Ux*Ux 均方
    mu2_sq = mu2 * mu2  # img(x,y) Uy*Uy
    mu1_mu2 = mu1 * mu2  # img(x,y) Ux*Uy

    # Calculate the variance, which is equal to the expectation of the square minus the square of the expectation, and the mean of the square minus the square of the mean
    paddedimg11 = padded_img1 * padded_img1
    paddedimg22 = padded_img2 * padded_img2
    paddedimg12 = padded_img1 * padded_img2

    sigma1_sq = tf.nn.conv2d(paddedimg11, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq  # sigma1 variance
    sigma2_sq = tf.nn.conv2d(paddedimg22, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq  # sigma2 variance
    sigma12 = tf.nn.conv2d(paddedimg12, window, strides=[1, 1, 1, 1],
                           padding='VALID') - mu1_mu2  # sigma12 covariance, the mean of the products minus the product of the means

    ssim_value = tf.clip_by_value(
        ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), 0, 1)
    if cs_map:  # Only consider contrast and structure, not light brightness
        cs_map_value = tf.clip_by_value((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2), 0, 1)  # Contrast structure map
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:  # Find the mean of the matrix, otherwise return the ssim matrix
        value = tf.reduce_mean(value)
    return value


# Calculate the cross-scale structural similarity index (by scaling the original image)
def tf_ms_ssim_resize(img1, img2, weights=None, return_ssim_map=None, filter_size=11, filter_sigma=1.5):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Parameters metioned in the paper
    level = len(weights)
    assert return_ssim_map is None or return_ssim_map < level
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    _, h, w, _ = img1.get_shape().as_list()
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size,
                                   filter_sigma=filter_sigma)

        # ssim_map = tf_ssim(img1, img2, cs_map=False, mean_metric=False, filter_size=filter_size,
        #                            filter_sigma=filter_sigma)
        if return_ssim_map == l:
            return_ssim_map_l = tf.image.resize_images(ssim_map, size=(h, w),
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) #tf.image.resize_images f.image.resize

        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        img1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    # ms-ssim公式
    value = tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) * (mssim[level - 1] ** weight[level - 1])
    if return_ssim_map is not None:
        return value, return_ssim_map_l
    else:
        return value


# Calculate the cross-scale structural similarity index (by expanding the receptive field)
def tf_ms_ssim(img1, img2, weights=None, mean_metric=False):
    if weights is None:
        weights = [1, 1, 1, 1, 1]  # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] #[1, 1, 1, 1, 1] #

    level = len(weights)
    sigmas = [0.5]
    for i in range(level - 1):
        sigmas.append(sigmas[-1] * 2)
    weight = tf.constant(weights, dtype=tf.float32)

    mssim = []
    mcs = []
    for l, sigma in enumerate(sigmas):
        filter_size = int(max(sigma * 4 + 1, 11))
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size,
                                   filter_sigma=sigma)
        mssim.append(ssim_map)
        mcs.append(cs_map)

    # list to tensor of dim D+1
    value = mssim[level - 1] ** weight[level - 1]
    for l in range(level):
        value = value * (mcs[l] ** weight[l])
    if mean_metric:
        return tf.reduce_mean(value)
    else:
        return value


img1 = np.arange(10000, dtype=np.float32).reshape([1, 100, 100, 1])
img2 = np.arange(10000, dtype=np.float32).reshape([1, 100, 100, 1])

with tf.Session() as sess:
    value = tf_ms_ssim(tf.constant(img1), tf.constant(img2), mean_metric=True)
    value1 = tf_ms_ssim_resize(tf.constant(img1), tf.constant(img2))
    print(sess.run(value))
    print(sess.run(value1))