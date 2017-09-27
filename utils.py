import torch
import torchvision 

def load_pretrain_model(model_file):

   model = torchvision.models.vgg16(pretrained=False)
   state_dict = torch.load(model_file)
   model.load_state_dict(state_dict)
   print('model has been load')
   return model