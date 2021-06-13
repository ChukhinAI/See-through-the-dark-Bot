import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
# import dataloader
# import model
# import model_v2 as model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2 as cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 64  # 32->64
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        # 5, number_f, 5 -> [64, 5, 5, 5], expected input[8, 3, 256, 256] to have 5 channels, but got 3 channels instead
        # 5, number_f, 3 -> [64, 5, 3, 3], expected input[8, 3, 256, 256] to have 5 channels, but got 3 channels instead
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, return_indices=False, ceil_mode=False)  # kernel_size=2->3
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        # print('x = ', x)
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r


def lowlight(image_path, epoch):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    image_path = "D:\\учеба\Master\\диплом\\bot_telegram_v3\\algorithm\\data\\input\\imgs\\image.jpg"
    print("image_path = ", image_path)
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    # DCE_net = model.enhance_net_nopool().cuda()
    DCE_net = enhance_net_nopool().cuda()
    # DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))  # original
    # epoch_numb = '163' 	# ========================================================================== epochs
    # for epoch_numb in range(1, 10,  1):
    epoch_numb = epoch
    print(" ================= start with epoch_numb = ", epoch_numb)
    # DCE_net.load_state_dict(torch.load('snapshots/Epoch' + str(epoch_numb) + '.pth'))  # new
    DCE_net.load_state_dict(torch.load('D:\\учеба\\Master\диплом\\bot_telegram_v3\\algorithm\\snapshots\\Epoch3.pth'))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(end_time)
    # image_path = image_path.replace('test_data', 'result') # original
    '''
    image_path = image_path.replace('test_data', 'result_new_v4_' + str(epoch_numb))

    result_path = image_path
    if not os.path.exists(image_path.replace('\\' + image_path.split("\\")[-1], '')):
        # print("in if not")
        os.makedirs(image_path.replace('\\' + image_path.split("\\")[-1], ''))
    '''
    # print("that is ok")
    # torchvision.utils.save_image(enhanced_image, result_path)
    # print("that is NOT ok")
    result_path = "D:\\учеба\Master\\диплом\\bot_telegram_v3\\algorithm\\data\\result\\imgs\\res.jpg"  # new
    torchvision.utils.save_image(enhanced_image, result_path)
    print("================= end with epoch_numb = ", epoch_numb)


if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        # filePath = 'data/test_data/'
        filePath = 'data/input/'

        file_list = os.listdir(filePath)

        for file_name in file_list:
            print("file_name = ", file_name)
            test_list = glob.glob(filePath + file_name + "/*")
            print("test_list = ", test_list)
            for image in test_list:
                # image = image
                print("image name = ", image)

                image_1 = cv2.imread(image)
                height, width, channels = image_1.shape
                print("processing ", image, " with ", height, "x", width, '\n')

                # """
                # while (height or width) >= 1505:
                while height >= 2500 or width >= 2500:
                    image_1 = cv2.resize(image_1, (0, 0), fx=0.2, fy=0.2)
                    # print("resizing ", image, " to ", height, "x", width, '\n')
                    cv2.imwrite(filename=image, img=image_1)
                    image_1 = cv2.imread(image)
                    height, width, channels = image_1.shape
                    print("resized ", image, " to ", height, "x", width, '\n')
                # image = cv2.imread(image)
                # """
                # print(image)
                # epoch = 160
                # for epoch in range(0, 399):
                #     lowlight(image, epoch)
                lowlight(image, epoch=3)
                # print("okay 2")



