import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from config import config
import vgg16_model as models
import imagenet as imgnet
import utils as utils

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

def main():

   ######### config  ###########

   configs = config()
   pretrain_vgg16_path = os.path.join(configs.py_dir,'model/vgg16_from_caffe.pth')

   ########  load training data ########
   ######### imagenet ############
   '''''''''
   train_img_dir = os.path.join(configs.data_dir, 'img/train')
   train_label_dir = os.path.join(configs.data_dir,'label/map_clsloc.txt')
   val_img_dir = os.path.join(configs.data_dir, 'img/val')
   val_txt_dir = os.path.join(configs.data_dir, 'img/val.txt')
   val_label_dir = os.path.join(configs.data_dir,'label/ILSVRC2012_validation_ground_truth.txt')

   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

   transform = transforms.Compose([
       transforms.RandomSizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       normalize
   ])

   train_data = imgnet.ImagenetDataset(img_path=train_img_dir,label_path=train_label_dir,
                                       file_name_txt_path='',split_flag='train',transform=transform)

   train_loader = Data.DataLoader(train_data,batch_size=configs.batch_size,
                                  shuffle=True, num_workers=2, pin_memory= True)

   val_data = imgnet.ImagenetDataset(img_path=val_img_dir,label_path=val_label_dir,
                                     file_name_txt_path= val_txt_dir, split_flag='valid', transform=transform)

   val_loader = Data.DataLoader(val_data, batch_size=configs.batch_size,
                                shuffle= False, num_workers= 2, pin_memory= True)
   '''''''''

   transform_train = transforms.Compose([
       transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])

   transform_test = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])

   trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
   train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

   testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
   val_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

   ######### build vgg model ##########

   vgg_cam = models.vgg_cam()
   vgg_pretrain_model = utils.load_pretrain_model(pretrain_vgg16_path)
   vgg_cam.copy_params_from_pretrain_vgg(vgg_pretrain_model,init_fc8=configs.init_random_fc8)
   vgg_cam = vgg_cam.cuda()

   ########## optim  ###########

   #optimizer = torch.optim.SGD(vgg_cam.parameters(),lr=configs.learning_rate,momentum=configs.momentum)
   optimizer = torch.optim.Adam(vgg_cam.parameters(),lr=configs.learning_rate,weight_decay=configs.weight_decay)
   loss_fun = nn.CrossEntropyLoss()

   for epoch in range(20):

       for step, (img_x,label_x) in enumerate(train_loader):

           img,label = Variable(img_x.cuda()), Variable(label_x.cuda())
           predict, _ = vgg_cam(img)
           loss = loss_fun(predict, label)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           if (step) % 20 == 0:
               print("Epoch [%d/%d] Loss: %.4f" % (epoch, step, loss.data[0]))

           if (step) % configs.save_ckpoints_iter_number == 0:

               torch.save(vgg_cam,os.path.join(configs.save_ckpt_dir, 'cam' + str(step) + '.pkl'))

           if step % configs.validate_iter_number == 0:

               test(vgg_cam,val_loader,loss_fun)

if __name__ == '__main__':

    main()
