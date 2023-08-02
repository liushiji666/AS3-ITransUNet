#coding:utf-8
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np


import time
import pdb
from option import  opt
from model import Restormer
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import PSNR, SSIM, SAM
from torch.optim.lr_scheduler import MultiStepLR


import scipy.io as scio  
psnr = []

   
def main():

    if opt.show:
        global writer
        writer = SummaryWriter(log_dir='logs') 
       
    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
		
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True
    
    # Loading datasets
    train_set = TrainsetFromFolder('D:/study/数据集/样本比例/处理后65/train/')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)    
    val_set = ValsetFromFolder('D:/study/数据集/样本比例/处理后65/test/')
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size= 1, shuffle=False)
        
      
    # Buliding model     
    model = Restormer()
    criterion = nn.L1Loss() 
      
    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()   
    print('# parameters:', sum(param.numel() for param in model.parameters())) 
                   
    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(),  lr=opt.lr,  betas=(0.9, 0.999), eps=1e-08)    

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)         
            opt.start_epoch = checkpoint['epoch'] + 1 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))       

    # Setting learning rate
    scheduler = MultiStepLR(optimizer, milestones=[35,70, 105, 140, 175], gamma=0.5, last_epoch = -1)

    # Training 
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step()
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"])) 
        train(train_loader, optimizer, model, criterion, epoch)         
        val(val_loader, model, epoch)              
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, optimizer, model, criterion, epoch):
    model.train()

    for iteration, batch in enumerate(train_loader, 1):
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()

        localFeats_dec3 = []
        localFeats_dec2 = []
        localFeats_dec1 = []
        for i in range(7):

            if i == 0:
                x = input[:, 0:7, :, :]
                # y = input[:,0,:,:]
                new_label = label[:, 0:3, :, :]
            elif i == 1:
                x = input[:, 2:9, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 3:8, :, :]
            elif i == 2:
                x = input[:, 7:14, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 8:13, :, :]
            elif i == 3:
                x = input[:, 12:19, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 13:18, :, :]
            elif i == 4:
                x = input[:, 17:24, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 18:23, :, :]
            elif i == 5:
                x = input[:, 22:29, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 23:28, :, :]
            elif i == 6:
                x = input[:, 24:31, :, :]
                # y = input[:,i,:,:]
                new_label = label[:, 28:31, :, :]

            SR, local_SR,localFeats_dec3, localFeats_dec2, localFeats_dec1 = model(x, i, localFeats_dec3, localFeats_dec2, localFeats_dec1)
            localFeats_dec3.detach_()
            localFeats_dec3 = localFeats_dec3.detach()
            localFeats_dec3 = Variable(localFeats_dec3.data, requires_grad=False)
            localFeats_dec2.detach_()
            localFeats_dec2 = localFeats_dec2.detach()
            localFeats_dec2 = Variable(localFeats_dec2.data, requires_grad=False)
            localFeats_dec1.detach_()
            localFeats_dec1 = localFeats_dec1.detach()
            localFeats_dec1 = Variable(localFeats_dec1.data, requires_grad=False)
            loss1 = criterion(SR, new_label)
            loss2 = criterion(local_SR, new_label)
            loss = 0.7*loss1 + 0.3*loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if iteration % 100 == 0:
            print("{}===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())), epoch, iteration, len(train_loader),
                loss.item()))

        if opt.show:
            writer.add_scalar('Train/Loss', loss.data[0])


def val(val_loader, model, epoch):
    model.eval()
    val_psnr = 0
    val_ssim = 0
    val_sam = 0

    for iteration, batch in enumerate(val_loader, 1):
        with torch.no_grad():
            input, label = Variable(batch[0]), Variable(batch[1])
            SR = np.ones((label.shape[1], label.shape[2], label.shape[3])).astype(np.float32)

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
                output,pk,localFeats_dec3, localFeats_dec2, localFeats_dec1 = model(x, i, localFeats_dec3, localFeats_dec2, localFeats_dec1)
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
            val_psnr += PSNR(SR, label.data[0].numpy())
            val_ssim += SSIM(SR, label.data[0].numpy())
            val_sam += SAM(SR, label.data[0].numpy())

    val_psnr = val_psnr / len(val_loader)
    val_ssim = val_ssim / len(val_loader)
    val_sam = val_sam / len(val_loader)

    print("PSNR = {:.3f}, SSIM = {:.4f}, SAM = {:.3f}".format(val_psnr, val_ssim, val_sam))
    if opt.show:
        writer.add_scalar('Val/PSNR', val_psnr, epoch)
   
    
def save_checkpoint(epoch, model, optimizer):
    model_out_path = "checkpoint/" + "{}_model_{}_epoch_{}.pth".format(opt.datasetName, opt.upscale_factor, epoch)
    state = {"epoch": epoch , "model": model.state_dict(), "optimizer":optimizer.state_dict()}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")     	
    torch.save(state, model_out_path)
 
          
if __name__ == "__main__":
    main()
