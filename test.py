import os
import numpy as np
import torch
from os import listdir
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from data_utils import is_image_file
from model4 import Restormer
import scipy.io as scio
from eval import PSNR, SSIM, SAM
import time


def main():
    input_path = 'D:/study/数据集/CAVE处理后/×4/test/'
    out_path = './result/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    PSNRs = []
    SSIMs = []
    SAMs = []

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = Restormer()

    if opt.cuda:
        model = nn.DataParallel(model).cuda()

    checkpoint = torch.load(opt.model_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    images_name = [x for x in listdir(input_path) if is_image_file(x)]
    T = 0
    for index in range(len(images_name)):
        with torch.no_grad():
            mat = scio.loadmat(input_path + images_name[index])
            hyperLR = mat['LR'].astype(np.float32).transpose(2, 0, 1)
            HR = mat['HR'].astype(np.float32).transpose(2, 0, 1)

            input = Variable(torch.from_numpy(hyperLR).float()).contiguous().view(1, -1, hyperLR.shape[1],
                                                                                  hyperLR.shape[2])
            SR = np.array(HR).astype(np.float32)

            if opt.cuda:
                input = input.cuda()

            localFeats_dec3 = []
            localFeats_dec2 = []
            localFeats_dec1 = []
            for i in range(7):
                if i == 0:
                    x = input[:, 0:7, :, :]
                    # y = input[:,0,:,:]
                    # new_label = label[:, 0:3, :, :]
                elif i == 1:
                    x = input[:, 2:9, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 3:8, :, :]
                elif i == 2:
                    x = input[:, 7:14, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 8:13, :, :]
                elif i == 3:
                    x = input[:, 12:19, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 13:18, :, :]
                elif i == 4:
                    x = input[:, 17:24, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 18:23, :, :]
                elif i == 5:
                    x = input[:, 22:29, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 23:28, :, :]
                elif i == 6:
                    x = input[:, 24:31, :, :]
                    # y = input[:,i,:,:]
                    # new_label = label[:, 28:31, :, :]
                output, local_SR,localFeats_dec3, localFeats_dec2, localFeats_dec1 = model(x, i, localFeats_dec3, localFeats_dec2, localFeats_dec1)
                if i == 0:
                    SR[0:3, :, :] = output.cpu().data[0].numpy()
                elif i == 1:
                    SR[3:8, :, :] = output.cpu().data[0].numpy()
                elif i == 2:
                    SR[8:13, :, :] = output.cpu().data[0].numpy()
                elif i == 3:
                    SR[13:18, :, :] = output.cpu().data[0].numpy()
                elif i == 4:
                    SR[18:23, :, :] = output.cpu().data[0].numpy()
                elif i == 5:
                    SR[23:28, :, :] = output.cpu().data[0].numpy()
                elif i == 6:
                    SR[28:31, :, :] = output.cpu().data[0].numpy()
        SR[SR < 0] = 0
        SR[SR > 1.] = 1.
        psnr = PSNR(SR, HR)
        ssim = SSIM(SR, HR)
        sam = SAM(SR, HR)

        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)

        SR = SR.transpose(1, 2, 0)
        HR = HR.transpose(1, 2, 0)

        scio.savemat(out_path + images_name[index], {'HR': HR, 'SR': SR})
        print("===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}====Name:{}".format(index + 1, psnr,
                                                                                                      ssim, sam,
                                                                                                      images_name[
                                                                                                          index]))
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs),
                                                                               np.mean(SAMs)))
    print(T / len(images_name))


if __name__ == "__main__":
    main()
