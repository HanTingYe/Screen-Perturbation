import os

def file1(dirpath):
    for root, dirs, file1 in os.walk(dirpath):
        return file1
def file2(dirpath):
    for root, dirs, file2 in os.walk(dirpath):
        return file2

file1=file1('/data/volume_2/optimize_display_POLED_400PPI/miniimagenet/images/')
file2=file2('/data/volume_2/optimize_display_POLED_400PPI/logs/test/web/images/cap_allcolorbar_intensity_deblurred_inception_resnetv1_cap_allpixel_1_0_p10_new_v1_all/')

# file1=file1('/home/hye/optimize_display_POLED_400PPI/miniimagenet/images')
# file2=file2('/home/hye/optimize_display_POLED_400PPI/logs/test/web/images/dec_densehalf2')

for i in file1:
    print(i)
    for j in file2:
        if i==j:
            os.remove('/data/volume_2/optimize_display_POLED_400PPI/miniimagenet/images'+'/'+i)

            # os.remove('/home/hye/optimize_display_POLED_400PPI/miniimagenet/images' + '/' + i)
