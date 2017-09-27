import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from action40_config import config
import vgg16_model as models
import utils as utils
import fold as imgfolder
import transforms as trans
import shutil

configs = config()
resume = 1

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = configs.learning_rate * (0.1 ** (epoch // 15))
    print(str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(net, testloader,loss_fun):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
        loss = loss_fun(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        '''''''''
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        '''''''''
    print(100.* correct/total)
    return 100.*correct/total

def main():

   ######### config  ###########

   best_metric = 0
   pretrain_vgg16_path = os.path.join(configs.py_dir,'model/vgg16_from_caffe.pth')

   ########  load training data ########
   ######### action 40  ############

   normalize = trans.Normalize(mean=[0.4001, 0.4401, 0.4687],std = [0.229, 0.224, 0.225])
                                    #std=[1, 1, 1])
   train_transform = trans.Compose([
       trans.RandomCrop(224,padding=4),
       trans.RandomHorizontalFlip(),
       trans.ToTensor(),
       normalize,
   ])

   val_transform = trans.Compose([
       trans.Scale((224,224)),
       trans.ToTensor(),
       normalize,
   ])

   train_data = imgfolder.ImageFolder(os.path.join(configs.data_dir,'img/train'),transform=train_transform)
   train_loader = torch.utils.data.DataLoader(
       train_data, batch_size=configs.batch_size, shuffle=True,
       num_workers=4, pin_memory=True)

   val_data = imgfolder.ImageFolder(os.path.join(configs.data_dir,'img/val'),transform=val_transform)
   val_loader = Data.DataLoader(val_data,batch_size=configs.batch_size,
                                shuffle= False, num_workers= 4, pin_memory= True)

   ######### build vgg model ##########

   vgg_cam = models.vgg_cam()
   vgg_pretrain_model = utils.load_pretrain_model(pretrain_vgg16_path)
   vgg_cam.copy_params_from_pretrain_vgg(vgg_pretrain_model,init_fc8=configs.init_random_fc8)
   vgg_cam = vgg_cam.cuda()

   ########  resume  ###########
   if resume:
        checkpoint = torch.load('/media/cheer/2T/train_pytorch/cam/ckpt/model_best.pth')
        vgg_cam.load_state_dict(checkpoint['state_dict'])
   ########## optim  ###########

   optimizer = torch.optim.SGD(vgg_cam.parameters(),lr=configs.learning_rate,momentum=configs.momentum,weight_decay=configs.weight_decay)
   #optimizer = torch.optim.Adam(vgg_cam.parameters(),lr=configs.learning_rate,weight_decay=configs.weight_decay)
   loss_fun = nn.CrossEntropyLoss()

   for epoch in range(200):

       adjust_learning_rate(optimizer, epoch)
       for step, (img_x,label_x) in enumerate(train_loader):

           img,label = Variable(img_x.cuda()), Variable(label_x.cuda())
           predict, _ = vgg_cam(img)
           loss = loss_fun(predict, label)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           if (step) % 20 == 0:
               print("Epoch [%d/%d] Loss: %.4f" % (epoch, step, loss.data[0]))

       current_metric = test(vgg_cam, val_loader, loss_fun)

       if current_metric > best_metric:

           torch.save({'state_dict': vgg_cam.state_dict()}, os.path.join(configs.save_ckpt_dir, 'cam' + str(epoch) + '.pth'))

           shutil.copy(os.path.join(configs.save_ckpt_dir, 'cam' + str(epoch) + '.pth'),
                       os.path.join(configs.save_ckpt_dir, 'model_best.pth'))
           best_metric = current_metric

       if epoch % 10 == 0:

           torch.save({'state_dict': vgg_cam.state_dict()}, os.path.join(configs.save_ckpt_dir, 'cam' + str(epoch) + '.pth'))


if __name__ == '__main__':

    main()
