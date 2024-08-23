import os
import requests
from requests.adapters import HTTPAdapter

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .utils.download import download_url_to_file

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )
    
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )

def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.

    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.

    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
        cached_file = os.path.join(os.getcwd(), os.path.basename(path))
        if not os.path.exists(cached_file):
            download_url_to_file(path, cached_file)
    elif name == 'casia-webface':
        # path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
        # cached_file = os.path.join(os.getcwd(), os.path.basename(path))
        # if not os.path.exists(cached_file):
        #     download_url_to_file(path, cached_file)
        cached_file = './facenet_mobilenet.pth'
    elif name == 'facescrub_dec':
        cached_file = './mobilenet_mtcnn_FaceScrub_dec_allpixel_1_0_webfaceSGD_v4.pth'
    elif name == 'facescrub_cap':
        cached_file = './mobilenet_mtcnn_FaceScrub_cap_allpixel_1_0_webfaceSGD_v4.pth'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    # model_dir = os.path.join(get_torch_home(), 'checkpoints')
    # os.makedirs(model_dir, exist_ok=True)
    #
    # cached_file = os.path.join(model_dir, os.path.basename(path))
    # if not os.path.exists(cached_file):
    #     download_url_to_file(path, cached_file)

    # ------------------------------------------------------#
    #                      Original                         #
    # ------------------------------------------------------#
    # state_dict = torch.load(cached_file)
    # mdl.load_state_dict(state_dict)
    # ------------------------------------------------------#

    # ------------------------------------------------------#
    #   根据预训练权重的Key和模型的Key进行加载
    # ------------------------------------------------------#
    model_dict = mdl.state_dict()
    pretrained_dict = torch.load(cached_file, mdl.device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        k = k.replace('backbone.model.', '')
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    # for k, v in pretrained_dict.items():
    #     if k[15:] in model_dict.keys() and np.shape(model_dict[k[15:]]) == np.shape(v):
    #         temp_dict[k[15:]] = v
    #         load_key.append(k[15:])
    #     else:
    #         no_load_key.append(k[15:])
    model_dict.update(temp_dict)
    mdl.load_state_dict(model_dict)
    # state_dict = torch.load(model_dict, mdl.device)
    # mdl.load_state_dict(state_dict)
    # ------------------------------------------------------#
    #   显示没有匹配上的Key
    # ------------------------------------------------------#
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

class MobileNetV1(nn.Module):
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super(MobileNetV1, self).__init__()
        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            # self.to(device)

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained == 'facescrub_dec':
            tmp_classes = 10575 ## train based on vggface2
        elif pretrained == 'facescrub_cap':
            tmp_classes = 10575 ## train based on vggface2
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')


        self.stage1 = nn.Sequential(
            # 160,160,3 -> 80,80,32
            conv_bn(3, 32, 2), 
            # 80,80,32 -> 80,80,64
            conv_dw(32, 64, 1), 

            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 1000) # self.fc = nn.Linear(1024, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.avg = nn.AdaptiveAvgPool2d(1)
        # self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.Dropout = nn.Dropout(dropout_prob)
        self.Bottleneck = nn.Linear(1024, 128, bias=False)
        self.last_bn = nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.classifier = nn.Linear(128, tmp_classes)
            if pretrained == 'vggface2' or pretrained == 'casia-webface':
                load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.classifier = nn.Linear(128, self.num_classes)
            if pretrained != 'vggface2' and pretrained != 'casia-webface':
                load_weights(self, pretrained)


                
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        # batch_size,1024
        # x = self.fc(x)
        #
        # x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        x = self.last_bn(x)
        if self.classify:
            x = self.classifier(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x
        # before_normalize = self.last_bn(x)
        # if self.classify:
        #     x = F.normalize(before_normalize, p=2, dim=1)
        #     cls = self.classifier(before_normalize)
        # else:
        #     x = F.normalize(before_normalize, p=2, dim=1)
        # return x, cls
