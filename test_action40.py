import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.nn import functional as F
from action40_config import config
import vgg16_model as models
import utils as utils
import fold as imgfolder
import transforms as trans
import shutil
import cv2
import json
import matplotlib.pyplot as plt
import collections

configs = config()
resume = 1

def parse_json(file_path):
    import json
    json_file = file(file_path)
    j = json.load(json_file)
    return j

######## source code from offical code ###############
def returnCAM(feature_conv, weight_softmax, class_idx,probs):
    # generate the class activation maps upsample to 256x256
    top_number = len(class_idx)
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = {}
    output_cam_imgs = []
    output_cam_prob = {}
    #out = collections.OrderedDict()
    for idx,prob in zip(class_idx,probs):
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        #out.setdefault(str(idx),[cv2.resize(cam_img, size_upsample),prob])
        output_cam.setdefault(idx,[cv2.resize(cam_img, size_upsample),prob])
        output_cam_prob.setdefault(prob,cv2.resize(cam_img,size_upsample))
        output_cam_imgs.append(cv2.resize(cam_img,size_upsample))

    return output_cam_imgs

def untransform(transform_img):

    transform_img = transform_img.transpose(1,2,0)
    transform_img *= [0.229, 0.224, 0.225]
    transform_img += [0.4001, 0.4401, 0.4687]
    transform_img = transform_img * 255
    transform_img = transform_img.astype(np.uint8)
    transform_img = transform_img[:,:,::-1]
    return transform_img

def test(net, testloader):

    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
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

   ########  load training data ########
   ######### action 40  ############

   normalize = trans.Normalize(mean=[0.4001, 0.4401, 0.4687],
                                    std=[0.229, 0.224, 0.225])
   transform = trans.Compose([
       trans.Scale((224,224)),
       trans.ToTensor(),
       normalize,
   ])

   test_data = imgfolder.ImageFolder(os.path.join(configs.data_dir,'img/test'),transform=transform)
   test_loader = Data.DataLoader(test_data,batch_size=configs.batch_size,
                                shuffle= False, num_workers= 4, pin_memory= True)

   classes = {int(key): value for (key, value)
              in parse_json(configs.class_info_dir).items()}

   ######### build vgg model ##########

   vgg_cam = models.vgg_cam()
   vgg_cam = vgg_cam.cuda()
   checkpoint = torch.load(configs.best_ckpt_dir)
   vgg_cam.load_state_dict(checkpoint['state_dict'])

   # hook the feature extractor
   features_blobs = []

   def hook_feature(module, input, output):
       features_blobs.append(output.data.cpu().numpy())

   finalconv_name = 'classifier'  # this is the last conv layer of the network
   vgg_cam._modules.get(finalconv_name).register_forward_hook(hook_feature)

   # get the softmax weight
   params = list(vgg_cam.parameters())
   weight_softmax = np.squeeze(params[-1].data.cpu().numpy())

   save_cam_dir = os.path.join(configs.py_dir,'predict')
   if not os.path.exists(save_cam_dir):
      os.mkdir(save_cam_dir)
   top_number = 5
   correct = 0
   total = 0

   for batch_idx, (inputs, targets) in enumerate(test_loader):

       inputs, targets = inputs.cuda(), targets.cuda()
       transformed_img = inputs.cpu().numpy()[0]
       target_name = classes[targets.cpu().numpy()[0]]
       transformed_img = untransform(transformed_img)
       inputs, targets = Variable(inputs, volatile=True), Variable(targets)
       outputs, _ = vgg_cam(inputs)

       _, predicted = torch.max(outputs.data, 1)
       total += targets.size(0)
       correct += predicted.eq(targets.data).cpu().sum()

       h_x = F.softmax(outputs).data.squeeze()
       probs, idx = h_x.sort(0, True)
       prob = probs.cpu().numpy()[:top_number]
       idx_ =  idx.cpu().numpy()[:top_number]
       OUT_CAM = returnCAM(features_blobs[-1],weight_softmax,idx_,prob)

       save_fig_dir = os.path.join(save_cam_dir, 'cam_' + str(batch_idx) + '.jpg')
       plt.figure(1, figsize=(8, 6))
       ax =  plt.subplot(231)
       img1 = transformed_img[:, :, (2, 1, 0)]
       ax.set_title(('{}').format(target_name),fontsize=14)
       ax.imshow(img1)

       for b_index, (idx,prob_in,cam) in enumerate(zip(idx_,prob,OUT_CAM)):

           cl = str(classes[idx])
           #save_fig_dir1 = os.path.join(save_cam_dir, 'cam_cv_' + str(batch_idx) + '_' + cl + '.jpg')
           height, width, _ = transformed_img.shape
           heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
           result = heatmap * 0.3 + transformed_img * 0.7
           ax = plt.subplot(2,3,b_index+2)
           ax.imshow(result.astype(np.uint8)[:,:,(2,1,0)])
           ax.set_title(('{}:{}').format(cl,('%.3f' % prob_in)), fontsize=8)

       plt.savefig(save_fig_dir)

       print batch_idx

   print(100.* correct/total)


if __name__ == '__main__':

    main()
