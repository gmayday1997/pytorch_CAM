import os

class config():
    def __init__(self):

        self.py_dir = os.getcwd()
        self.data_dir = os.path.join(self.py_dir,'data')
        self.save_ckpt_dir = os.path.join(self.py_dir,'ckpt')
        self.class_info_dir = os.path.join(self.data_dir,'class.json')
        self.best_ckpt_dir = os.path.join(self.save_ckpt_dir,'model_best.pth')
        self.height = 224
        self.width = 224
        self.init_random_fc8 = True
        self.learning_rate = 1e-5
        self.max_iter_number = 100000
        self.validate_iter_number = 500
        self.save_ckpoints_iter_number = 10000
        self.weight_decay = 5e-5
        self.momentum = 0.99
        self.train = False
        if self.train:
            self.batch_size = 8
        else:
            self.batch_size = 1
