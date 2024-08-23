import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from wave_optics import get_wiener_loss
from wave_optics import get_wiener_RGB_loss


class Loss(keras.Model):
    def __init__(self, opt, cameraOpt):
        super(Loss, self).__init__()
        self.area = opt.area
        self.opt = opt

    def get_area_loss(self, mapped_pattern, epoch=0, target_area=None):

        if target_area is None:
            target_area = max(self.area, 1 - 0.05 * (epoch // 5))  # decrease target area every 5 epochs

        area_ratio = tf.reduce_sum(mapped_pattern) / tf.cast(tf.reduce_prod(tf.shape(mapped_pattern)), tf.float32)
        area_loss = (area_ratio - target_area) ** 2
        return area_loss, area_ratio, target_area

    def call(self, mapped_pattern, PSFs, PSFs_RGB, ssim_mean, recons, gt=None, epoch=0):
        total_loss = 0
        out = {}

        # area loss
        # area_loss, area, target_area = self.get_area_loss(mapped_pattern, epoch=epoch)
        # total_loss += self.opt.area_gamma * area_loss
        # out['area'] = area
        # out['target_area'] = tf.constant(target_area, tf.float32)
        # out['area_loss'] = self.opt.area_gamma * area_loss

        # top-10 l2 loss
        if self.opt.use_data:
            n, w, h, c = tf.shape(gt).numpy()
            residual = tf.abs(tf.reshape(gt - recons, [n, -1]))
            residual = tf.sort(residual, axis=1, direction='ASCENDING') # 'DESCENDING'
            num_top = int(0.1 * w * h * c)
            residual = residual[:, :num_top]
            out['top_l2'] = self.opt.l2_gamma * (-tf.nn.l2_loss(residual) / num_top) / 0.0000005
            total_loss += out['top_l2']

        # invertible loss
        # if self.opt.invertible:
        #     invert_loss, transfer_funcs = get_wiener_loss(PSFs)
        #     total_loss += self.opt.inv_gamma * invert_loss  # (minimize loss, invertible loss has minus sign inside)
        #     out['invertible_loss'] = self.opt.inv_gamma * invert_loss

        # screen interference
        # if self.opt.invertible:
        #     invert_loss, transfer_funcs = get_wiener_RGB_loss(PSFs, PSFs_RGB)
        #     out['invertible_loss'] = self.opt.inv_gamma * invert_loss / 0.00000307  # (minimize loss, invertible loss has minus sign inside)
        #     total_loss += out['invertible_loss']
        transfer_funcs = []


        # out['total_loss'] = total_loss

        out['ssim_mean'] = ssim_mean

        return out, transfer_funcs