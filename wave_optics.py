import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
# import math
import matplotlib.pyplot as plt
# import gc
# import torch
from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift
from scipy.spatial import distance

from utils import load_display, load_rect, crop_image

# gc.collect()
# torch.cuda.empty_cache()


class LensProp(layers.Layer):
    def __init__(self):
        super(LensProp, self).__init__()

    def call(self, u1, dx1, dx2, lambd, f, d):
        m, _ = tf.shape(u1).numpy()
        k = 2 * np.pi / lambd

        L2 = m * dx2
        x2 = tf.constant(np.arange(-m / 2, m / 2) * dx2, dtype=tf.float32)
        X2, Y2 = tf.meshgrid(x2, x2)

        j = tf.complex(real=np.zeros_like(X2),
                       imag=np.ones_like(X2))

        c_imag = k * (1 - d / f) / (2 * f) * (tf.pow(X2, 2) + tf.pow(Y2, 2))
        c_real = np.zeros_like(c_imag)
        c0 = tf.exp(tf.complex(real=c_real, imag=c_imag))
        c = tf.multiply(-j / (lambd * f), c0)

        u2_1 = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(u1))) * dx1 ** 2
        u2 = tf.multiply(c, u2_1)
        return u2, L2


def capture(img, interfer_img, psf, psf_RGB):
    '''
    convolve images with psf
    :param img_batch:  tensor [n, h, w ,c]
    :param psf:        tensor [hp, wp, c]
    :return:           tensor [n, h, w, c]
    '''
    n, h, w, c = tf.shape(img).numpy()
    hp, wp, _ = tf.shape(psf).numpy()
    img = tf.cast(img, dtype=tf.float32)
    interfer_img = tf.cast(interfer_img, dtype=tf.float32)
    psf = tf.cast(psf, dtype=tf.float32)
    psf_RGB = tf.cast(psf_RGB, dtype=tf.float32)

    img_blur = []
    for cc in range(c):
        img_real = tf.pad(img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        psf_real = tf.pad(psf[:, :, cc], tf.constant([[0, h], [0, w]]), 'CONSTANT')

        interfer_img_real = tf.pad(interfer_img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        psf_RGB_real = tf.pad(psf_RGB[:, :, cc], tf.constant([[0, h], [0, w]]), 'CONSTANT')

        img_complex = tf.complex(real=img_real, imag=tf.zeros_like(img_real))
        psf_complex = tf.complex(real=psf_real, imag=tf.zeros_like(psf_real))

        interfer_img_complex = tf.complex(real=interfer_img_real, imag=tf.zeros_like(interfer_img_real))
        psf_RGB_complex = tf.complex(real=psf_RGB_real, imag=tf.zeros_like(psf_RGB_real))



        fft_img = tf.signal.fft2d(img_complex)
        fft_psf = tf.signal.fft2d(psf_complex)

        fft_interfer_img = tf.signal.fft2d(interfer_img_complex)
        fft_pixel_psf = tf.signal.fft2d(psf_RGB_complex)

        fft_out = fft_img * tf.tile(fft_psf[None, :, :], tf.constant([n, 1, 1])) + fft_interfer_img * tf.tile(fft_pixel_psf[None, :, :], tf.constant([n, 1, 1]))

        # fft_out1 = fft_img * tf.tile(fft_psf[None, :, :], tf.constant([n, 1, 1]))
        #
        # fft_out2 = fft_interfer_img * tf.tile(fft_pixel_psf[None, :, :], tf.constant([n, 1, 1]))
        #
        # fft_out = fft_out1 + fft_out2

        # fft_out = fft_img * tf.tile(fft_psf[None, :, :], tf.constant([n, 1, 1], dtype=tf.float32)) + fft_interfer_img * tf.tile(fft_pixel_psf[None, :, :], tf.constant([n, 1, 1], dtype=tf.float32))

        out = tf.abs(tf.signal.ifft2d(fft_out))
        crop_x, crop_y = int(hp/2), int(wp/2)
        out = out[:, crop_x: crop_x + h, crop_y: crop_y + w]
        img_blur.append(out)

    return tf.stack(img_blur, axis=3)

def capture_alpha(img, interfer_img, psf, psf_RGB, alpha):
    '''
    convolve images with psf
    :param img_batch:  tensor [n, h, w ,c]
    :param psf:        tensor [hp, wp, c]
    :return:           tensor [n, h, w, c]
    '''
    n, h, w, c = tf.shape(img).numpy()
    hp, wp, _ = tf.shape(psf).numpy()
    img = tf.cast(img, dtype=tf.float32)
    interfer_img = tf.cast(interfer_img, dtype=tf.float32)
    psf = tf.cast(psf, dtype=tf.float32)
    psf_RGB = tf.cast(psf_RGB, dtype=tf.float32)

    img_blur = []
    for cc in range(c):
        img_real = tf.pad(img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        psf_real = tf.pad(psf[:, :, cc], tf.constant([[0, h], [0, w]]), 'CONSTANT')

        interfer_img_real = tf.pad(interfer_img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        psf_RGB_real = tf.pad(psf_RGB[:, :, cc], tf.constant([[0, h], [0, w]]), 'CONSTANT')

        img_complex = tf.complex(real=img_real, imag=tf.zeros_like(img_real))
        psf_complex = tf.complex(real=psf_real, imag=tf.zeros_like(psf_real))

        interfer_img_complex = tf.complex(real=interfer_img_real, imag=tf.zeros_like(interfer_img_real))
        psf_RGB_complex = tf.complex(real=psf_RGB_real, imag=tf.zeros_like(psf_RGB_real))



        fft_img = tf.signal.fft2d(img_complex)
        fft_psf = tf.signal.fft2d(psf_complex)

        fft_interfer_img = tf.signal.fft2d(interfer_img_complex)
        fft_pixel_psf = tf.signal.fft2d(psf_RGB_complex)

        fft_out = 1/(1+alpha) * fft_img * tf.tile(fft_psf[None, :, :], tf.constant([n, 1, 1])) + alpha/(1+alpha) * fft_interfer_img * tf.tile(fft_pixel_psf[None, :, :], tf.constant([n, 1, 1]))

        # fft_out1 = fft_img * tf.tile(fft_psf[None, :, :], tf.constant([n, 1, 1]))
        #
        # fft_out2 = fft_interfer_img * tf.tile(fft_pixel_psf[None, :, :], tf.constant([n, 1, 1]))
        #
        # fft_out = fft_out1 + fft_out2

        # fft_out = fft_img * tf.tile(fft_psf[None, :, :], tf.constant([n, 1, 1], dtype=tf.float32)) + fft_interfer_img * tf.tile(fft_pixel_psf[None, :, :], tf.constant([n, 1, 1], dtype=tf.float32))

        out = tf.abs(tf.signal.ifft2d(fft_out))
        crop_x, crop_y = int(hp/2), int(wp/2)
        out = out[:, crop_x: crop_x + h, crop_y: crop_y + w]
        img_blur.append(out)

    return tf.stack(img_blur, axis=3)


def wiener_deconv(img, psf):
    '''
    convolve images with psf
    :param img_batch:  tensor [n, h, w ,c]
    :param psf:        tensor [hp, wp, c]
    :return:           tensor [n, h, w, c]
    '''
    n, h, w, c = tf.shape(img).numpy()
    hp, wp, _ = tf.shape(psf).numpy()
    img = tf.cast(img, dtype=tf.float32)
    psf = tf.cast(psf, dtype=tf.float32)

    img_recon = []
    snr = 0.0001 / tf.math.reduce_std(img) ** 2
    for cc in range(c):
        img_real = tf.pad(img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        psf_real = tf.pad(psf[:, :, cc], tf.constant([[0, h], [0, w]]), 'CONSTANT')

        img_complex = tf.complex(real=img_real, imag=tf.zeros_like(img_real))
        psf_complex = tf.complex(real=psf_real, imag=tf.zeros_like(psf_real))

        fft_img = tf.signal.fft2d(img_complex)
        fft_psf = tf.signal.fft2d(psf_complex)

        # wiener deconvolution kernel
        invK = tf.complex(real=1 / (tf.abs(fft_psf) ** 2 + snr), imag=tf.zeros_like(psf_real))
        K = tf.math.conj(fft_psf) * invK
        fft_out = fft_img * tf.tile(K[None, :, :], tf.constant([n, 1, 1]))

        out = tf.abs(tf.signal.ifftshift(tf.signal.ifft2d(fft_out), axes=(1, 2)))
        crop_x0, crop_y0 = int(h/2), int(w/2)

        # if crop_x0 + h > h + hp:
        #     offset = int(h/2 - hp)
        #     out = tf.roll(out, shift=[-offset, -offset], axis=[1,2])
        #     # import ipdb; ipdb.set_trace()
        #     crop_x0 -= offset
        #     crop_y0 -= offset

        if crop_x0 + h > h + hp:
            offset = int(h/2 - hp)
            # out = tf.roll(out, shift=[-offset, -offset], axis=[1,2])
            out = tf.roll(out, shift=-offset, axis=1)
            # import ipdb; ipdb.set_trace()
            crop_x0 -= offset

        if crop_y0 + w > w + wp:
            offset = int(w/2 - wp)
            # out = tf.roll(out, shift=[-offset, -offset], axis=[1,2])
            out = tf.roll(out, shift=-offset, axis=2)
            # import ipdb; ipdb.set_trace()
            crop_y0 -= offset

        out = out[:, crop_x0:crop_x0+h, crop_y0: crop_y0+w]
        out = tf.where(out < 0, 0, out)
        img_recon.append(out)

    return tf.stack(img_recon, axis=3)


def get_wiener_loss(psf):
    '''
    This function computes the invertible loss.
    :param psf:
    :return: scalar
    '''
    hp, wp, _ = tf.shape(psf).numpy()
    psf = tf.cast(psf, dtype=tf.float32)

    def get_overall_func(blur_func):

        blur_func = blur_func / tf.reduce_sum(blur_func)  # normalize to one
        blur_func = tf.complex(real=blur_func, imag=tf.zeros_like(blur_func))

        fft_blur_func = tf.signal.fft2d(blur_func)
        inv_fft_blur_func = tf.complex(real=1 / (tf.abs(fft_blur_func) ** 2 + 0.015),
                                       imag=tf.zeros([hp, wp], dtype=tf.float32))
        overall_func = tf.abs(tf.math.conj(fft_blur_func) * fft_blur_func * inv_fft_blur_func)
        return tf.signal.fftshift(overall_func)

    # compute system frequency response for RGB channels
    overall_funcs = []
    for channel in range(3):
        overall_funcs.append(get_overall_func(psf[:,:,channel]))
    overall_funcs = tf.stack(overall_funcs, axis=2)

    # compute invertibility loss
    sorted_overall_funcs = tf.sort(tf.reshape(overall_funcs, [-1]), direction='ASCENDING')
    num_low = int(0.3 * hp * wp * 3)
    score = -tf.reduce_mean(sorted_overall_funcs[:num_low])

    return score, overall_funcs

def get_wiener_RGB_loss(psf, psf_RGB):
    '''
    This function computes the invertible loss.
    :param psf:
    :return: scalar
    '''
    hp, wp, _ = tf.shape(psf_RGB).numpy()
    psf_RGB = tf.cast(psf_RGB, dtype=tf.float32)
    psf = tf.cast(psf, dtype=tf.float32)


    def get_overall_func(blur_func, passive_blur_func):

        blur_func = blur_func / tf.reduce_sum(blur_func)  # normalize to one
        passive_blur_func = passive_blur_func / tf.reduce_sum(passive_blur_func)  # normalize to one
        blur_func = tf.complex(real=blur_func, imag=tf.zeros_like(blur_func))
        passive_blur_func = tf.complex(real=passive_blur_func, imag=tf.zeros_like(passive_blur_func))

        fft_blur_func = tf.signal.fft2d(blur_func)
        fft_passive_blur_func = tf.signal.fft2d(passive_blur_func)
        inv_fft_passive_blur_func = tf.complex(real=1 / (tf.abs(fft_passive_blur_func) ** 2 + 0.015), imag=tf.zeros([hp, wp], dtype=tf.float32))
        overall_func = tf.abs(tf.math.conj(fft_passive_blur_func) * fft_blur_func * inv_fft_passive_blur_func)
        return tf.signal.fftshift(overall_func)

    # compute system frequency response for RGB channels
    overall_funcs = []
    for channel in range(3):
        overall_funcs.append(get_overall_func(psf_RGB[:,:,channel], psf[:,:,channel]))
    overall_funcs = tf.stack(overall_funcs, axis=2)

    # compute invertibility loss
    sorted_overall_funcs = tf.sort(tf.reshape(overall_funcs, [-1]), direction='ASCENDING') #ASCENDING 'DESCENDING'
    num_low = int(0.3 * hp * wp * 3)
    score = -tf.reduce_mean(sorted_overall_funcs[:num_low]) # default is minimize - represents maximize +represents minimize

    return score, overall_funcs


class Camera(keras.Model):
    def __init__(self, pattern, delta1=8e-6): # 8e-6 1e-6
        super(Camera, self).__init__()
        self.lens2sensor = LensProp()

        self.pattern = pattern / 255
        self.T = tf.shape(self.pattern)[0].numpy()

        self.D1 = 4e-3
        self.unitPattern_sz = 336e-6 # 400PPI 64e-6 delta=1e-6; 75PPI(600PPI) 336e-6 delta=8e-6; 150PPI(1200PPI) 168e-6 delta=8e-6

        self.wvls = {'R': 0.61e-6,
                     'G': 0.53e-6,
                     'B': 0.47e-6}
        self.f = 0.003  # focal length [m] 0.01 0.003
        self.delta1 = delta1  # spacing of lens aperture [m]
        self.delta2 = 0.5e-6  # spacing of sensor aperture [m]
        self.pixel_sz = 2e-6  # pixel pitch size [m] 2e-6 4e-6

        self.T_exp = self.unitPattern_sz / self.delta1

        self.dilation = int(self.T_exp / self.T)


        self.pattern_new = np.kron(self.pattern, np.ones((self.dilation, self.dilation)))
        # tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix((self.pattern,  dtype=tf.float32), tf.linalg.LinearOperatorFullMatrix(np.ones((self.dilation, self.dilation)), dtype=tf.float32)])
        # self.pattern_new = np.empty(shape=(self.T_exp, 0))  # tf.zeros([np.shape(pattern_RGB)[0] * np.array(c)[0], 0])
        # for jj in range(self.dilation):
        #     pattern_column_temp = np.empty(shape=(0, self.T))  # tf.zeros([0, np.shape(pattern_RGB)[0]])
        #     for ii in range(self.dilation):
        #         pattern_temp = self.pattern
        #         pattern_column_temp = np.vstack((pattern_column_temp, pattern_temp))  # tf.stack((pattern_column_temp, pattern_temp), axis = 0)
        #     self.pattern_new = np.hstack((self.pattern_new, pattern_column_temp))  # tf.stack((display_RGB, pattern_column_temp), axis = 1)

        self.M_R = self.get_nsamples(self.wvls['R'])
        self.M_G = self.get_nsamples(self.wvls['G'])
        self.M_B = self.get_nsamples(self.wvls['B'])

        # generate aperture mask of max size
        self.aperture_R = tf.constant(load_rect(self.delta1, self.D1, self.M_R), dtype=tf.float32)

        # generate pattern
        # # self.delta1 * self.T_exp  / 4
        self.display_index, self.display_RGB_allpixel = load_display(self.pattern_new, self.delta1, self.M_R, self.delta1 * self.T_exp, option = 'repeat', option2 = 'preprocessing') # self.unitPattern_sz = 64e-6 = self.delta1 * self.T_exp

        _, self.display_RGB_rgbpixel = load_display(self.pattern_new, self.delta1, self.M_R, self.delta1 * self.T_exp, option = 'repeat', option2 = 'running')

    def get_nsamples(self, wvl):
        return int((wvl * self.f
                    / self.delta1 / self.delta2) // 2 * 2)  # number of samples

    def dn_sample(self, psf):
        dn_scale = int(self.pixel_sz / self.delta2)
        kernel = tf.constant(np.ones((dn_scale, dn_scale, 1, 1)), dtype=tf.float32)
        return tf.squeeze(tf.nn.conv2d(psf[None, :, :, None], kernel, strides=dn_scale, padding='VALID'))

    def call(self, u1_RGB_real_mask=None, DoR_option=None, mode_option=None):


        if mode_option == 'preprocessing':
            # display, display_RGB = load_display(pattern, self.delta1, self.M_R, self.delta1 * T, tile_option, mode_option)
            PSFs = PSFs_RGB =[]
            # u1_real = self.display_index * self.aperture_R
            # u1_RGB_real = self.display_RGB_allpixel * self.aperture_R
            # u1_real = self.display_index[:self.M_R, :self.M_R] * self.aperture_R
            # u1_RGB_real = self.display_RGB_allpixel[:self.M_R, :self.M_R] * self.aperture_R
            u1_real = self.display_index
            u1_RGB_real = self.display_RGB_allpixel
            if DoR_option == 'large':
                return PSFs, PSFs_RGB, u1_real, u1_RGB_real
            elif DoR_option == 'small':
                return PSFs, PSFs_RGB, u1_real[:self.M_R, :self.M_R], u1_RGB_real[:self.M_R, :self.M_R]


        elif mode_option == 'running':
            # display, display_RGB = load_display(pattern, self.delta1, self.M_R, self.delta1 * T, tile_option, mode_option)
            height_temp, width_temp = np.shape(u1_RGB_real_mask)
            display = tf.constant(self.display_index, dtype=tf.float32)
            display_RGB = tf.constant(self.display_RGB_rgbpixel, dtype=tf.float32)
            # u1_real = display * self.aperture_R
            # u1_RGB_real = display_RGB * self.aperture_R
            u1_real = display[:self.M_R, :self.M_R] * self.aperture_R
            u1_RGB_real = display_RGB[:self.M_R, :self.M_R] * self.aperture_R
            # u1_RGB_real = tf.image.resize_with_crop_or_pad(u1_RGB_real[:, :, None], height_temp, width_temp)
            # u1_RGB_real = tf.squeeze(u1_RGB_real)
            red_value = tf.constant(2, dtype=tf.float32)
            green_value = tf.constant(3, dtype=tf.float32)
            blue_value = tf.constant(4, dtype=tf.float32)

            # pixel_R_ind = tf.where(u1_RGB_real == 2)
            # pixel_G_ind = tf.where(u1_RGB_real == 3)
            # pixel_B_ind = tf.where(u1_RGB_real == 4)
            # u1_RGB_R_real = tf.zeros_like(u1_RGB_real)
            # u1_RGB_R_real = tf.assign(u1_RGB_R_real,1, pixel_R_ind)

            # obtain center
            # dis_matrix_R = distance.cdist(pixel_R_ind, pixel_R_ind, 'euclidean')
            # dis_matrix_G = distance.cdist(pixel_G_ind, pixel_G_ind, 'euclidean')
            # dis_matrix_B = distance.cdist(pixel_B_ind, pixel_B_ind, 'euclidean')

            # mask_R = u1_RGB_real == 2
            # mask_R_ones = tf.ones(tf.shape(u1_RGB_real))
            # mask_R_zeros = tf.zeros(tf.shape(u1_RGB_real))
            # u1_RGB_R_real = tf.where(mask_R, mask_R_ones, mask_R_zeros)
            # u1_RGB_R_real = tf.where(u1_RGB_real == 2, tf.ones(tf.shape(u1_RGB_real), dtype=tf.float32),
            #                          tf.zeros(tf.shape(u1_RGB_real), dtype=tf.float32))
            # u1_RGB_G_real = tf.where(u1_RGB_real == 3, tf.ones(tf.shape(u1_RGB_real), dtype=tf.float32),
            #                          tf.zeros(tf.shape(u1_RGB_real), dtype=tf.float32))
            # u1_RGB_B_real = tf.where(u1_RGB_real == 4, tf.ones(tf.shape(u1_RGB_real), dtype=tf.float32),
            #                          tf.zeros(tf.shape(u1_RGB_real), dtype=tf.float32))

            u1_RGB_R_real = tf.where(u1_RGB_real == red_value, tf.constant(1, dtype=tf.float32), tf.constant(0, dtype=tf.float32))
            u1_RGB_G_real = tf.where(u1_RGB_real == green_value, tf.constant(1, dtype=tf.float32), tf.constant(0, dtype=tf.float32))
            u1_RGB_B_real = tf.where(u1_RGB_real == blue_value, tf.constant(1, dtype=tf.float32), tf.constant(0, dtype=tf.float32))

            # plt.imshow(u1_RGB_R_real)
            # plt.show()
            # plt.imshow(u1_RGB_G_real)
            # plt.show()
            # plt.imshow(u1_RGB_B_real)
            # plt.show()

            # u1_RGB_R_real = tf.multiply(u1_RGB_R_real, u1_RGB_real_mask)
            #
            # u1_RGB_G_real = tf.multiply(u1_RGB_G_real, u1_RGB_real_mask)
            #
            # u1_RGB_B_real = tf.multiply(u1_RGB_B_real, u1_RGB_real_mask)

            u1_RGB_R_real = tf.multiply(u1_RGB_R_real, u1_RGB_real_mask[:self.M_R, :self.M_R])

            u1_RGB_G_real = tf.multiply(u1_RGB_G_real, u1_RGB_real_mask[:self.M_R, :self.M_R])

            u1_RGB_B_real = tf.multiply(u1_RGB_B_real, u1_RGB_real_mask[:self.M_R, :self.M_R])

            # plt.imshow(u1_RGB_real_mask)
            # plt.show()
            # plt.imshow(u1_RGB_R_real)
            # plt.show()
            # plt.imshow(u1_RGB_G_real)
            # plt.show()
            # plt.imshow(u1_RGB_B_real)
            # plt.show()

            # Test codes
            # test = np.array(u1_RGB_R_real)
            # test1 = np.array(u1_RGB_real)



            crop_G = int((self.M_R - self.M_G) / 2)
            crop_B = int((self.M_R - self.M_B) / 2)

            # image_red
            u1_R = u1_real
            u1_RGB_R = u1_RGB_R_real

            # image_green
            u1_G = u1_real[crop_G:-crop_G, crop_G:-crop_G]
            u1_RGB_G = u1_RGB_G_real[crop_G:-crop_G, crop_G:-crop_G]

            # image_blue
            u1_B = u1_real[crop_B:-crop_B, crop_B:-crop_B]
            u1_RGB_B = u1_RGB_B_real[crop_B:-crop_B, crop_B:-crop_B]



            u1_R = tf.complex(real=u1_R, imag=tf.zeros_like(u1_R))
            u1_G = tf.complex(real=u1_G, imag=tf.zeros_like(u1_G))
            u1_B = tf.complex(real=u1_B, imag=tf.zeros_like(u1_B))
            u1_RGB_R = tf.complex(real=u1_RGB_R, imag=tf.zeros_like(u1_RGB_R))
            u1_RGB_G = tf.complex(real=u1_RGB_G, imag=tf.zeros_like(u1_RGB_G))
            u1_RGB_B = tf.complex(real=u1_RGB_B, imag=tf.zeros_like(u1_RGB_B))

            # lens propagation
            u2_R, _ = self.lens2sensor(u1_R, self.delta1, self.delta2, self.wvls['R'], self.f, 0)
            u2_G, _ = self.lens2sensor(u1_G, self.delta1, self.delta2, self.wvls['G'], self.f, 0)
            u2_B, _ = self.lens2sensor(u1_B, self.delta1, self.delta2, self.wvls['B'], self.f, 0)
            u2_RGB_R, _ = self.lens2sensor(u1_RGB_R, self.delta1, self.delta2, self.wvls['R'], self.f, 0)
            u2_RGB_G, _ = self.lens2sensor(u1_RGB_G, self.delta1, self.delta2, self.wvls['G'], self.f, 0)
            u2_RGB_B, _ = self.lens2sensor(u1_RGB_B, self.delta1, self.delta2, self.wvls['B'], self.f, 0)

            dn_u2_R = self.dn_sample(tf.pow(tf.abs(u2_R), 2))
            dn_u2_G = self.dn_sample(tf.pow(tf.abs(u2_G), 2))
            dn_u2_B = self.dn_sample(tf.pow(tf.abs(u2_B), 2))
            dn_u2_RGB_R = self.dn_sample(tf.pow(tf.abs(u2_RGB_R), 2))
            dn_u2_RGB_G = self.dn_sample(tf.pow(tf.abs(u2_RGB_G), 2))
            dn_u2_RGB_B = self.dn_sample(tf.pow(tf.abs(u2_RGB_B), 2))
            # dn_u2_RGB_R_zeros = tf.zeros_like(dn_u2_RGB_R)
            # dn_u2_RGB_G_zeros = tf.zeros_like(dn_u2_RGB_G)
            # dn_u2_RGB_B_zeros = tf.zeros_like(dn_u2_RGB_B)

            # cat three channels
            crop_G = int((dn_u2_G.shape[0] - dn_u2_B.shape[0]) / 2)
            crop_R = int((dn_u2_R.shape[0] - dn_u2_B.shape[0]) / 2)

            PSFs = tf.stack([dn_u2_R[crop_R:crop_R + dn_u2_B.shape[0], crop_R:crop_R + dn_u2_B.shape[0]],
                             dn_u2_G[crop_G:crop_G + dn_u2_B.shape[0], crop_G:crop_G + dn_u2_B.shape[0]],
                             dn_u2_B], axis=2)

            # PSFs_R = tf.stack([dn_u2_RGB_R[crop_R:crop_R + dn_u2_B.shape[0], crop_R:crop_R + dn_u2_B.shape[0]],
            #                    dn_u2_RGB_G_zeros[crop_G:crop_G + dn_u2_B.shape[0], crop_G:crop_G + dn_u2_B.shape[0]],
            #                    dn_u2_RGB_B_zeros], axis=2)
            #
            # PSFs_G = tf.stack([dn_u2_RGB_R_zeros[crop_R:crop_R + dn_u2_B.shape[0], crop_R:crop_R + dn_u2_B.shape[0]],
            #                    dn_u2_RGB_G[crop_G:crop_G + dn_u2_B.shape[0], crop_G:crop_G + dn_u2_B.shape[0]],
            #                    dn_u2_RGB_B_zeros], axis=2)
            #
            # PSFs_B = tf.stack([dn_u2_RGB_R_zeros[crop_R:crop_R + dn_u2_B.shape[0], crop_R:crop_R + dn_u2_B.shape[0]],
            #                    dn_u2_RGB_G_zeros[crop_G:crop_G + dn_u2_B.shape[0], crop_G:crop_G + dn_u2_B.shape[0]],
            #                    dn_u2_RGB_B], axis=2)

            PSFs_RGB = tf.stack([dn_u2_RGB_R[crop_R:crop_R + dn_u2_B.shape[0], crop_R:crop_R + dn_u2_B.shape[0]],
                               dn_u2_RGB_G[crop_G:crop_G + dn_u2_B.shape[0], crop_G:crop_G + dn_u2_B.shape[0]],
                               dn_u2_RGB_B], axis=2)

            interfer_all_pixel = tf.stack([u1_RGB_R_real, u1_RGB_G_real, u1_RGB_B_real], axis=2)

            # u1_RGB_real = tf.image.resize_with_crop_or_pad(u1_RGB_real[:, :, None], height_temp, width_temp)

            return PSFs, PSFs_RGB, interfer_all_pixel, u1_RGB_real
        else:
            print('Invalid code run  method.')




def set_params():
    args = dict()
    args['D1'] = 4e-3  # diam of lens/disp aperture [m]
    args['f'] = 0.003  # focal length [m] 0.01 0.003
    args['z'] = 0.05  # display lens distance [m]

    args['delta1'] = 8e-6  # spacing of lens aperture [m] 8e-6 1e-6
    args['delta2'] = 0.5e-6  # spacing of sensor aperture [m]
    args['pixel_sz'] = 2e-6  # pixel pitch size [m] 2e-6 4e-6

    args['wvls'] = {'R': 0.61e-6,
                    'G': 0.53e-6,
                    'B': 0.47e-6}

    def get_nsamples(wvl):
        m = np.ceil(wvl * args['f']
                    / args['delta1'] / args['delta2'])  # number of samples
        return int(m // 2 * 2)

    args['M_R'] = get_nsamples(args['wvls']['R'])
    args['M_G'] = get_nsamples(args['wvls']['G'])
    args['M_B'] = get_nsamples(args['wvls']['B'])
    return args

def get_interfer_img(Red_mask, Green_mask, Blue_mask, red_pixel_index, green_pixel_index, blue_pixel_index, interfer_pixel_index, mapped_pixel, batch_size, img_height, img_width):
    display_red_mask = Red_mask #tf.cast(Red_mask)
    display_green_mask = Green_mask #tf.cast(Green_mask)
    display_blue_mask = Blue_mask #tf.cast(Blue_mask)

    for iii in range(len(red_pixel_index[0])):
        display_red_mask = tf.where(display_red_mask == interfer_pixel_index[red_pixel_index[0][iii]],
                                    mapped_pixel[red_pixel_index[0][iii]], display_red_mask) # mapped_pixel[iii], display_red_mask)
    for jjj in range(len(green_pixel_index[0])):
        display_green_mask = tf.where(display_green_mask == interfer_pixel_index[green_pixel_index[0][jjj]],
                                      mapped_pixel[green_pixel_index[0][jjj]], display_green_mask) # mapped_pixel[len(red_pixel_index[0]) + jjj], display_green_mask)
    for kkk in range(len(blue_pixel_index[0])):
        display_blue_mask = tf.where(display_blue_mask == interfer_pixel_index[blue_pixel_index[0][kkk]],
                                     mapped_pixel[blue_pixel_index[0][kkk]], display_blue_mask) # mapped_pixel[len(red_pixel_index[0]) + len(green_pixel_index[0]) + kkk], display_blue_mask)

    height_temp, width_temp = np.shape(display_red_mask)

    display_red_mask_ = crop_image(display_red_mask, height_temp, width_temp)
    display_green_mask_ = crop_image(display_green_mask, height_temp, width_temp)
    display_blue_mask_ = crop_image(display_blue_mask, height_temp, width_temp)

    interfer_img = tf.stack([display_red_mask_, display_green_mask_, display_blue_mask_], axis=2)


    interfer_img = tf.tile(interfer_img[None, :, :, :], tf.constant([batch_size, 1, 1, 1]))

    # or
    # interfer_img = tf.expand_dims(interfer_img, axis=0)
    # interfer_img = tf.tile(interfer_img[:, :, :], tf.constant([batch_size, 1, 1, 1]))


    # if batch_size > 1:
    #     for ii in range(batch_size - 1):
    #         interfer_img = tf.concat([interfer_img, interfer_img], 0)
    map_size = tf.constant([height_temp * 3, width_temp * 3], dtype=tf.float32)
    map_size = tf.dtypes.cast(tf.math.ceil(map_size), tf.int32)

    interfer_img = tf.image.resize_with_crop_or_pad(tf.image.resize(interfer_img, map_size, method='bilinear'), img_height, img_width)

    return interfer_img