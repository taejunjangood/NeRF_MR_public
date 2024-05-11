import yaml
import numpy as np
import pyCT

def getHeader():
    header = pyCT.getParameters()
    header.mode = False
    return header

def getConfig(config_name):
    with open('./nerf2/configs/{}.yml'.format(config_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return _Config(config_name, config)

class _Config():
    def __init__(self, config_name, config):
        self.name = config_name
        self.device = None
        self.step = None
        self.batch = Struct(size=None)
        self.ray_sampling = Struct(step=None, mode=None)
        self.model = Struct(depth=None, width=None, encoding_dim=None, activation=None)
        self.optimizer = Struct(learning_rate=None, learning_rate_decay=None)
        self.freq = Struct(print_loss=None, save_weight=None, run_test=None)
        self.__set(config)

    def __str__(self):
        output = ''
        output += 'step               : {}\n'.format(self.step)
        output += 'batch_size         : {}\n'.format(self.batch.size)
        output += 'ray_sampling_mode  : {}\n'.format(self.ray_sampling.mode)
        output += 'ray_sampling_step  : {} mm\n'.format(self.ray_sampling.step)
        output += 'model_depth        : {}\n'.format(self.model.depth)
        output += 'model_width        : {}\n'.format(self.model.width)
        output += 'encoding_dimension : {}\n'.format(self.model.encoding_dim)
        output += 'activation function: {}\n'.format(self.model.activation)

        return output
        
    def __set(self, config):
        # step
        self.step = config['step']
        # batch
        self.batch.size = config['batch_size']
        # ray_sampling
        self.ray_sampling.step = config['ray_sampling_step']
        self.ray_sampling.mode = config['ray_sampling_mode']
        # model
        self.model.depth = config['model_depth']
        self.model.width = config['model_width']
        self.model.encoding_dim = config['encoding_dim']
        self.model.activation = config['activation']
        # optimizer
        self.optimizer.learning_rate = config['learning_rate']
        self.optimizer.learning_rate_decay = config['learning_rate_decay']
        # freq
        self.freq.print_loss = config['freq_print_loss']
        self.freq.save_weight = config['freq_save_weight']
        self.freq.save_image = config['freq_save_image']




class Struct():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
