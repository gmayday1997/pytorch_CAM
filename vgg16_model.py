import torch
import torch.nn as nn
import torch.nn.init as init

###### orginal vgg-cam  ######
class vgg_cam(nn.Module):

    def __init__(self):
        super(vgg_cam,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512,512,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2,2),
        )
        #self.cam = nn.AvgPool2d(7,7)
        #self.fc = nn.Linear(512,40,bias=False)

        self.classifier = nn.Sequential(
            nn.Conv2d(512,1024,3,1,padding=1),
            nn.ReLU(),
        )
        '''''''''
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7,padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True)
        )
        '''''''''
        self.cam = nn.AvgPool2d(14,14)
        self.fc = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(1024,40,bias=False)
        )

    def forward(self,x):

        x = self.conv1(x)
        #print x.data.cuda().numpy()
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x1 = x.data.cpu().numpy()
        #print(x.size)
        x = self.classifier(x)
        cam = self.cam(x)
        #x2 = cam.data.cpu().numpy()
        #print(cam.size(0))
        cam_x = cam.view(cam.size(0),-1)
        #s = cam_x.data.cpu().numpy()
        out = self.fc(cam_x)
        return out,cam_x

    def copy_params_from_pretrain_vgg(self,pretrain_vgg16, init_fc8 = True):

        conv_blocks = [self.conv1,
                       self.conv2,
                       self.conv3,
                       self.conv4,
                       self.conv5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(pretrain_vgg16.features.children())

        for idx, conv_block in enumerate(conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        '''''''''
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = pretrain_vgg16.classifier[i1]
            l2 = self.classifier[i2]
            # print type(l1), dir(l1),
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        '''''''''

        '''''''''
        n_class = self.fc.weight.size()[0]
        if init_fc8:
            l1 = self.fc
            init.kaiming_normal(l1.weight)
        else:
            l1 = pretrain_vgg16.classifier[7]
            l2 = self.fc
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
        '''''''''

    ###########source code from
    ########### https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
    def copy_params_from_vgg16(self, vgg16):
        for l1, l2 in zip(vgg16.features, self.features):
            if (isinstance(l1, nn.Conv2d) and
                    isinstance(l2, nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i in [0, 3]:
            l1 = vgg16.classifier[i]
            l2 = self.classifier[i]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())