import numpy as np
import os
import ntpath
import time
from . import util
from . import html


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port, env=opt.display_env)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, opt.images_mode)
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
            # self.img_cap_dir = os.path.join(self.img_dir, 'cap_redcolorbar_p10_v1')
            self.img_cap_crop_dir = os.path.join(self.img_dir, opt.save_cap_dir + '_crop')#cap_testnew cap_testnew_36 cap_testoriginal cap_testmask cap_testmask_32 cap_onepixelscheme_v1 cap_test_onepixel cap_onepixelscheme_v2 cap_test_redcolorbar
            util.mkdirs([self.img_dir, self.img_cap_crop_dir])
            # self.img_dec_dir = os.path.join(self.img_dir, 'dec_redcolorbar_p10_v1')
            self.img_dec_crop_dir = os.path.join(self.img_dir, opt.save_dec_dir + '_crop')#dec_testnew dec_testnew_36 dec_testoriginal dec_testmask dec_testmask_32 dec_onepixelscheme_v1 dec_test_onepixel dec_onepixelscheme_v2 dec_test_redcolorbar
            util.mkdirs([self.img_dir, self.img_dec_crop_dir])
            self.img_cap_all_dir = os.path.join(self.img_dir,
                                                 opt.save_cap_dir + '_all')  # cap_testnew cap_testnew_36 cap_testoriginal cap_testmask cap_testmask_32 cap_onepixelscheme_v1 cap_test_onepixel cap_onepixelscheme_v2 cap_test_redcolorbar
            util.mkdirs([self.img_dir, self.img_cap_all_dir])
            # self.img_dec_dir = os.path.join(self.img_dir, 'dec_redcolorbar_p10_v1')
            self.img_dec_all_dir = os.path.join(self.img_dir,
                                                 opt.save_dec_dir + '_all')  # dec_testnew dec_testnew_36 dec_testoriginal dec_testmask dec_testmask_32 dec_onepixelscheme_v1 dec_test_onepixel dec_onepixelscheme_v2 dec_test_redcolorbar
            util.mkdirs([self.img_dir, self.img_dec_all_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, file_name):
        if self.display_id > 0: # show images in the browser
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + opt.images_mode))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win = self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    #image_numpy = np.flipud(image_numpy)
                    self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label, store_history=False),
                                       win=self.display_id + idx)
                    idx += 1

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                # img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label)) # used for optimization
                # util.save_image(image_numpy, img_path)

                # if label == 'captured_0':
                #     self.img_cap_dir = os.path.join(self.img_dir, 'cap')
                #     img_path = os.path.join(self.img_cap_dir,'ILSVRC2012_val_%.8d.JPEG' % epoch)  # used for dataset generation
                #     util.save_image(image_numpy, img_path)
                # elif label == 'deconved_0':
                #     self.img_dec_dir = os.path.join(self.img_dir, 'dec')
                #     img_path = os.path.join(self.img_dec_dir, 'ILSVRC2012_val_%.8d.JPEG' % epoch)  # used for dataset generation
                #     util.save_image(image_numpy, img_path)

                if label == 'captured_0_crop':
                    img_path = os.path.join(self.img_cap_crop_dir, file_name[epoch])  # used for dataset generation
                    util.save_image(image_numpy, img_path)
                elif label == 'deconved_0_crop':
                    img_path = os.path.join(self.img_dec_crop_dir, file_name[epoch])  # used for dataset generation
                    util.save_image(image_numpy, img_path)
                elif label == 'captured_0_all':
                    img_path = os.path.join(self.img_cap_all_dir, file_name[epoch])  # used for dataset generation
                    util.save_image(image_numpy, img_path)
                elif label == 'deconved_0_all':
                    img_path = os.path.join(self.img_dec_all_dir, file_name[epoch])  # used for dataset generation
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('Results of Epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.jpeg' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_img(self, img, name, display_id=0):
        self.vis.image(img, caption=name, win=display_id)

    def tensor_to_img(self, img_tensor):
        if len(img_tensor.shape) > 3:
            img = img_tensor[0, ...].numpy()
        else:
            img = img_tensor.numpy()
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        return (255 * img).astype(np.uint8)

    def tensor_to_img_save(self, img_tensor, temp_id):
        if len(img_tensor.shape) > 3:
            img = img_tensor[temp_id, ...].numpy()
        else:
            img = img_tensor.numpy()
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        return (255 * img).astype(np.uint8)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, errors, display_id=4):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=display_id)

    def plot_current_curve(self, values, name, display_id=0):
        self.vis.line(
            X=np.arange(len(values)),
            Y=values,
            opts={
                'title': name,
                'xlabel': 'dimension',
                'ylabel': 'intensity'},
            win=display_id)

    def print_current_loss(self, epochs, batch_id, loss, logfile):
        str = '%d/%d ' % (batch_id, epochs)
        for key, value in loss.items():
            str += '%s=%.8f, ' % (key, value.numpy())
        print(str)
        logfile.write(str + '\n')

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_psnr(self, epoch, psnr, ssim):
        message = 'Validation...\n(epoch: %d, psnr: %.2f, ssim: %.2f)' % (epoch, psnr, ssim)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_message(self, message):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        # for now, we take the folder name and the image name
        tokens = image_path[0].split('/')
        short_path = '%s_%s' % (tokens[-2], tokens[-1])
        # short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
