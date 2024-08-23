import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt


####################### load inital display pattern ##########################

def load_rect(dx, display_width, total_resolution):
    x1 = np.arange(-total_resolution / 2, total_resolution / 2) * dx
    X1, Y1 = np.meshgrid(x1, x1)
    in_field_real = rect(X1 / display_width) * rect(Y1 / display_width)
    return in_field_real

def euclideanDistance(instance1,instance2,dimension):
    distance = 0
    for i in range(dimension):
        distance += (instance1[i] - instance2[i])**2

    return np.sqrt(distance)

def pixellayout(DPI):
    if DPI == 150:
        ## pixel pattern size 42 600DPI
        c_red = np.array(([160.642248424002, 56.2408942330144], [73.4902189781020, 144.624005077753]))/4
        c_green = np.array(([29.2085221318613, 21.5112629699054], [116.063498170268, 20.2873745762990], [115.362214562811, 110.014709105244], [29.0329571461657, 108.496497648302]))/4
        c_blue = np.array(([75.5117672812880, 63.5575875458452], [158.190948905110, 142.196833239753]))/4
        r_red = np.array([16, 16])/4
        r_green = np.array([14, 14, 14, 14])/4
        r_blue = np.array([20, 20])/4
        return c_red, c_green, c_blue, r_red, r_green, r_blue
    elif DPI == 75:
        ## pixel pattern size 21 1200DPI
        c_red = np.array(([160.642248424002, 56.2408942330144], [73.4902189781020, 144.624005077753])) / 8
        c_green = np.array(([29.2085221318613, 21.5112629699054], [116.063498170268, 20.2873745762990],
                            [115.362214562811, 110.014709105244], [29.0329571461657, 108.496497648302])) / 8
        c_blue = np.array(([75.5117672812880, 63.5575875458452], [158.190948905110, 142.196833239753])) / 8
        r_red = np.array([16, 16]) / 8
        r_green = np.array([14, 14, 14, 14]) / 8
        r_blue = np.array([20, 20]) / 8
        return c_red, c_green, c_blue, r_red, r_green, r_blue


    # elif DPI == 200:
    #     c_red = (122.394094037335	42.0883003680110; 55.9925477928396	110.189718154479)
    #     c_green = (21.4922073385610	16.3895336913565; 87.6674271773471	15.4570472962278; 86.3712110954752	83.8207307468523; 20.5965387780310	82.6639982082301)
    #     c_blue = (57.5327750714575	48.4248286063583; 120.526437261036	106.816634849335)
    #     r_red = (12.1904761904762, 12.1904761904762)
    #     r_green = (10.6666666666667, 10.6666666666667, 10.6666666666667, 10.6666666666667)
    #     r_blue = (15.2380952380952, 15.2380952380952)
    #     return c_red, c_green, c_blue, r_red, r_green, r_blue
    # elif DPI == 300:
    #     c_red = (80.3211242120010,	27.6204471165072; 36.7451094890510,	72.3120025388766)
    #     c_green = (15.1042610659307, 10.7556314849527; 58.5317490851340	10.1436872881495; 59.6811072814056	53.0073545526218; 16.5164785730829	52.2482488241510)
    #     c_blue = (37.7558836406440	31.7787937729226; 79.0954744525548	71.0984166198764)
    #     r_red = (8, 8)
    #     r_green = (7, 7, 7, 7)
    #     r_blue = (10, 10)
    #     return c_red, c_green, c_blue, r_red, r_green, r_blue
    elif DPI == 400:
        c_red = np.array(([61.1970470186674, 22.5679597078150],[27.9962738964198, 55.0948590772393])) / 1
        c_green = np.array(([10.7461036692805, 8.19476684567825], [43.8337135886735, 7.72852364811391],
                            [43.1856055477376, 41.9103653734261], [10.2982693890155, 41.3319991041151])) / 1
        c_blue = np.array(([28.7663875357288, 24.2124143031791], [60.2632186305179, 54.9321269484773])) / 1
        r_red = np.array([6.09523809523810, 6.09523809523810]) / 1
        r_green = np.array([5.33333333333333, 5.33333333333333, 5.33333333333333, 5.33333333333333]) / 1
        r_blue = np.array([7.61904761904762, 7.61904761904762]) / 1
        return c_red, c_green, c_blue, r_red, r_green, r_blue
    # elif DPI == 600:
    #     c_red = (40.1605621060005	13.8102235582536; 18.3725547445255	36.1560012694383)
    #     c_green = (8.05213053296533	5.37781574247635; 29.7658745425670	5.07184364407475; 30.3405536407028	26.5036772763109; 7.75823928654143	27.1241244120755)
    #     c_blue = (18.8779418203220	15.8893968864613; 39.5477372262774	36.0492083099382)
    #     r_red = (4, 4)
    #     r_green = (3.5, 3.5, 3.5, 3.5)
    #     r_blue = (5, 5)
    #     return c_red, c_green, c_blue, r_red, r_green, r_blue
    # elif DPI == 800:
    #     c_red = (30.5985235093337	11.2839798539075; 13.9981369482099	27.5474295386197)
    #     c_green = (6.13495659654501	4.09738342283912; 22.6787615562415	3.86426182405695; 22.3547075357736	20.9551826867131; 5.91103945641252	20.6659995520575)
    #     c_blue = (14.3831937678644	12.1062071515896; 30.1316093152590	27.4660634742386)
    #     r_red = (3.04761904761905 3.04761904761905)
    #     r_green = (2.66666666666667, 2.66666666666667, 2.66666666666667, 2.66666666666667)
    #     r_blue = (3.80952380952381, 3.80952380952381)
    #     return c_red, c_green, c_blue, r_red, r_green, r_blue
    # elif DPI == 1200:
    #     c_red = (20.0802810530003	6.90511177912680; 9.18627737226275	17.5780006347192)
    #     c_green = (4.02606526648267	2.18890787123818; 14.8829372712835	2.03592182203738; 15.1702768203514	12.7518386381555; 4.37911964327072	12.5620622060378)
    #     c_blue = (9.43897091016100	7.44469844323065; 19.7738686131387	18.5246041549691)
    #     r_red = (2, 2)
    #     r_green = (1.75, 1.75, 1.75, 1.75)
    #     r_blue = (2.5, 2.5)
    #     return c_red, c_green, c_blue, r_red, r_green, r_blue
    # elif DPI == 1600:
    #     c_red = (15.2992617546669	5.26103754600137; 6.99906847410495	13.7737147693098)
    #     c_green = (3.06747829827251	2.04869171141956; 11.3393807781208	1.93213091202848; 11.5583061488392	10.4775913433565; 3.33647210915864	10.3329997760288)
    #     c_blue = (7.19159688393219	6.05310357579478; 15.0658046576295	13.7330317371193)
    #     r_red = (1.52380952380952, 1.52380952380952)
    #     r_green = (1.33333333333333, 1.33333333333333, 1.33333333333333, 1.33333333333333)
    #     r_blue = (1.90476190476190, 1.90476190476190)
    #     return c_red, c_green, c_blue, r_red, r_green, r_blue


def rect(x):
    return (np.abs(x) <= 0.5).astype(np.float32)

# def add_subpixel(pattern_RGB, red_val, green_val, blue_val, DPI):
#     c_red, c_green, c_blue, r_red, r_green, r_blue = pixellayout(DPI)
#     for ii in range(pattern_RGB.shape[0]):
#         for jj in range(pattern_RGB.shape[1]):
#             if euclideanDistance([ii, jj], c_red[0, :], 2) <= r_red[0] or euclideanDistance([ii, jj], c_red[1, :], 2) <= r_red[1]:
#                 if pattern_RGB[ii, jj] != 255:
#                     pattern_RGB[ii, jj] = red_val
#             elif euclideanDistance([ii, jj], c_green[0, :], 2) <= r_green[0] or euclideanDistance([ii, jj], c_green[1, :], 2) <= \
#                     r_green[1] or euclideanDistance([ii, jj], c_green[2, :], 2) <= r_green[2] or euclideanDistance(
#                     [ii, jj], c_green[3, :], 2) <= r_green[3]:
#                 if pattern_RGB[ii, jj] != 255:
#                     pattern_RGB[ii, jj] = green_val
#             elif euclideanDistance([ii, jj], c_blue[0, :], 2) <= r_blue[0] or euclideanDistance([ii, jj], c_blue[1, :], 2) <= r_blue[1]:
#                 if pattern_RGB[ii, jj] != 255:
#                     pattern_RGB[ii, jj] = blue_val
#     pattern_RGB = np.where(pattern_RGB == 255, 0, pattern_RGB)
#     return pattern_RGB

def add_subpixel_np(pattern_RGB, red_val, green_val, blue_val, c_red, c_green, c_blue, r_red, r_green, r_blue):
    for ii in range(pattern_RGB.shape[0]):
        for jj in range(pattern_RGB.shape[1]):
            if euclideanDistance([ii, jj], c_red[0, :], 2) <= r_red[0] or euclideanDistance([ii, jj], c_red[1, :], 2) <= r_red[1]:
                if pattern_RGB[ii, jj] != 1:
                    pattern_RGB[ii, jj] = red_val
            elif euclideanDistance([ii, jj], c_green[0, :], 2) <= r_green[0] or euclideanDistance([ii, jj], c_green[1, :], 2) <= \
                    r_green[1] or euclideanDistance([ii, jj], c_green[2, :], 2) <= r_green[2] or euclideanDistance(
                    [ii, jj], c_green[3, :], 2) <= r_green[3]:
                if pattern_RGB[ii, jj] != 1:
                    pattern_RGB[ii, jj] = green_val
            elif euclideanDistance([ii, jj], c_blue[0, :], 2) <= r_blue[0] or euclideanDistance([ii, jj], c_blue[1, :], 2) <= r_blue[1]:
                if pattern_RGB[ii, jj] != 1:
                    pattern_RGB[ii, jj] = blue_val
    pattern_RGB = np.where(pattern_RGB == 1, 0, pattern_RGB)
    return pattern_RGB

# def add_subpixel(pattern_RGB, red_val, green_val, blue_val, c_red, c_green, c_blue, r_red, r_green, r_blue):
#     for ii in range(pattern_RGB.shape[0]):
#         for jj in range(pattern_RGB.shape[1]):
#             if euclideanDistance([ii, jj], c_red[0, :], 2) <= r_red[0] or euclideanDistance([ii, jj], c_red[1, :], 2) <= r_red[1]:
#                 if pattern_RGB[ii, jj] != 1:
#                     pattern_RGB[ii, jj] = red_val
#             elif euclideanDistance([ii, jj], c_green[0, :], 2) <= r_green[0] or euclideanDistance([ii, jj], c_green[1, :], 2) <= \
#                     r_green[1] or euclideanDistance([ii, jj], c_green[2, :], 2) <= r_green[2] or euclideanDistance(
#                     [ii, jj], c_green[3, :], 2) <= r_green[3]:
#                 if pattern_RGB[ii, jj] != 1:
#                     pattern_RGB[ii, jj] = green_val
#             elif euclideanDistance([ii, jj], c_blue[0, :], 2) <= r_blue[0] or euclideanDistance([ii, jj], c_blue[1, :], 2) <= r_blue[1]:
#                 if pattern_RGB[ii, jj] != 1:
#                     pattern_RGB[ii, jj] = blue_val
#     pattern_RGB = tf.where(pattern_RGB == 1, 0, pattern_RGB)
#     return pattern_RGB


def load_display(pattern, delta, M, s, option=None, option2=None):
    # repeat display patterns to cover aperture plane.
    # pattern: display pattern (tensor.float32)
    # L1: diam of the source plane [m]
    #     (display and padding region)
    # D1: diam of aperture [m]
    # M:  number of samples
    # s:  diam of pattern [m]
    # ds: spacing of pattern [m]

    if option == 'randomRot':
        # local randomness
        # rotate and flip the pattern
        rot90 = tf.transpose(pattern)
        flip = pattern[:, ::-1]
        rot90_flip = rot90[::-1, :]

        N = int(np.ceil(delta * M / s))
        np.random.seed(10)
        order = np.random.randint(4, size=(N, N))

        pattern16 = []
        for ix in range(N):
            pattern4 = []
            for iy in range(N):
                if order[ix, iy] == 0:
                    pattern4.append(pattern)
                elif order[ix, iy] == 1:
                    pattern4.append(rot90)
                elif order[ix, iy] == 2:
                    pattern4.append(flip)
                else:
                    pattern4.append(rot90_flip)
            pattern16.append(tf.concat(pattern4, axis=1))
        display = tf.concat(pattern16, axis=0)

    elif option == 'repeat':
        # figure out times to repeat the pattern
        c = tf.constant([delta * M / s, delta * M / s], dtype=tf.float32)
        c = tf.dtypes.cast(tf.math.ceil(c), tf.int32)

        # pixel_index = np.ones((1, 8)) # 8 subpixles
        c_red, c_green, c_blue, r_red, r_green, r_blue = pixellayout(150)

        pattern_RGB = np.array(np.copy(pattern), dtype='int64')
        pattern_RGB = add_subpixel_np(pattern_RGB, 2, 3, 4, c_red, c_green, c_blue, r_red, r_green, r_blue)

        # plt.imshow(pattern_RGB)
        # plt.show()
        # print(pattern_RGB)

        # for ii in range(pattern_RGB.shape[0]):
        #     for jj in range(pattern_RGB.shape[1]):
        #         if euclideanDistance([ii,jj], c_red[0,:], 2) <= r_red[0] or euclideanDistance([ii,jj], c_red[1,:], 2) <= r_red[1]:
        #             if  pattern_RGB[ii,jj] != 255:
        #                 pattern_RGB[ii,jj] = 2
        #         elif euclideanDistance([ii,jj], c_green[0,:], 2) <= r_green[0] or euclideanDistance([ii,jj], c_green[1,:], 2) <= r_green[1] or euclideanDistance([ii,jj], c_green[2,:], 2) <= r_green[2] or euclideanDistance([ii,jj], c_green[3,:], 2) <= r_green[3]:
        #             if  pattern_RGB[ii,jj] != 255:
        #                 pattern_RGB[ii, jj] = 3
        #         elif euclideanDistance([ii, jj], c_blue[0, :], 2) <= r_blue[0] or euclideanDistance([ii, jj], c_blue[1, :], 2) <= r_blue[1]:
        #             if  pattern_RGB[ii,jj] != 255:
        #                 pattern_RGB[ii,jj] = 4
        # pattern_RGB = np.where(pattern_RGB == 255, 0, pattern_RGB)

        # print(pattern_RGB)
        if option2 == 'preprocessing':
            display = tf.tile(pattern, c)
            display_RGB = np.empty(shape=(np.shape(pattern_RGB)[0]*np.array(c)[0],0))
            for jj in range(np.array(c)[1]):
                pattern_column_temp = np.empty(shape=(0,np.shape(pattern_RGB)[0]))
                for ii in range(np.array(c)[0]):
                    pattern_temp = add_subpixel_np(pattern_RGB, 2+3*ii+3*np.array(c)[1]*jj, 3+3*ii+3*np.array(c)[1]*jj, 4+3*ii+3*np.array(c)[1]*jj, c_red, c_green, c_blue, r_red, r_green, r_blue)
                    pattern_column_temp = np.vstack((pattern_column_temp, pattern_temp))
                display_RGB = np.hstack((display_RGB, pattern_column_temp))

            # plt.imshow(display_RGB)
            # plt.show()
            # print(display_RGB)

        elif option2 == 'running':
            display = tf.tile(pattern, c)
            display_RGB = np.empty(shape=(np.shape(pattern_RGB)[0]*np.array(c)[0],0))#tf.zeros([np.shape(pattern_RGB)[0] * np.array(c)[0], 0])
            for jj in range(np.array(c)[1]):
                pattern_column_temp = np.empty(shape=(0,np.shape(pattern_RGB)[0]))#tf.zeros([0, np.shape(pattern_RGB)[0]])
                for ii in range(np.array(c)[0]):
                    # if ((2+3*ii+3*np.array(c)[1]*jj) in interfer_pixel_index):
                    #     pattern_temp = add_subpixel(pattern_RGB, interfer_pixel[interfer_pixel_index == (2+3*ii+3*np.array(c)[1]*jj)], 3, 4, c_red, c_green, c_blue, r_red, r_green, r_blue)
                    # elif ((3+3*ii+3*np.array(c)[1]*jj) in interfer_pixel_index):
                    #     pattern_temp = add_subpixel(pattern_RGB, 2, interfer_pixel[interfer_pixel_index == (3+3*ii+3*np.array(c)[1]*jj)], 4, c_red, c_green, c_blue, r_red, r_green, r_blue)
                    # elif ((4+3*ii+3*np.array(c)[1]*jj) in interfer_pixel_index):
                    #     pattern_temp = add_subpixel(pattern_RGB, 2, 3, interfer_pixel[interfer_pixel_index == (4+3*ii+3*np.array(c)[1]*jj)], c_red, c_green, c_blue, r_red, r_green, r_blue)
                    # else:
                    pattern_temp = add_subpixel_np(pattern_RGB, 2, 3, 4, c_red, c_green, c_blue, r_red, r_green, r_blue)
                    pattern_column_temp = np.vstack((pattern_column_temp, pattern_temp))#tf.stack((pattern_column_temp, pattern_temp), axis = 0)
                display_RGB = np.hstack((display_RGB, pattern_column_temp))#tf.stack((display_RGB, pattern_column_temp), axis = 1)

            # plt.imshow(display_RGB)
            # plt.show()
            # print(display_RGB)


            # display_RGB = tf.tile(pattern_RGB, c)
            # display_pixel_index = tf.tile(pixel_index, c)
            # print(display_RGB)
        else:
            print('Invalid code run method.')





    else:
        print('Invalid pixel tiling method.')

    # crop display size to the size of aperture plane
    # return tf.cast(display[:M, :M], dtype=tf.float32), tf.cast(display_RGB[:M, :M], dtype=tf.float32)
    return tf.cast(display, dtype=tf.float32), tf.cast(display_RGB, dtype=tf.float32)


def print_opt(file, opt):
    print('=========================================')
    file.write('=========================================\n')
    for arg in vars(opt):
        print('%s: %s' % (arg, getattr(opt, arg)))
        file.write('%s: %s\n' % (arg, getattr(opt, arg)))
    print('=========================================')
    file.write('=========================================\n')

def crop_image(re_img,new_height,new_width):
    # suitable for pillow
    # re_img=Image.fromarray(np.uint8(re_img))
    # width, height = re_img.size
    height, width = np.shape(re_img)
    left = int((width - new_width)/2)
    top = int((height - new_height)/2)
    right = int((width + new_width)/2)
    bottom = int((height + new_height)/2)
    crop_im = re_img[top:bottom, left:right] #Cropping Image
    # crop_im = re_img.crop((left, top, right, bottom)) #Cropping Image
    # crop_im = np.asarray(crop_im)
    return crop_im