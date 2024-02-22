import yaml
import numpy as np

def getHeader(image_shape, is_incribed):
    header = _Header()

    if len(image_shape)==2:
        nz = 1
        ny, nx = image_shape
    elif len(image_shape)==3:
        nz, ny, nx = image_shape
    header.object.size.x = header.object.length.x = nx
    header.object.size.y = header.object.length.y = ny
    header.object.size.z = header.object.length.z = nz

    header.detector.spacing.u = 1.
    header.detector.spacing.v = 1.
    header.detector.size.u = int(max(nx,ny)/header.detector.spacing.u) if is_incribed else int(np.ceil((nx**2 + ny**2)**.5)/header.detector.spacing.u)
    header.detector.size.v = nz
    header.detector.length.u = header.detector.size.u * header.detector.spacing.u
    header.detector.length.v = header.detector.size.v * header.detector.spacing.v

    header.distance.source2object = max(nx,ny) / 2 if is_incribed else (nx**2 + ny**2)**.5 / 2
    header.distance.source2detector = max(nx,ny) if is_incribed else (nx**2 + ny**2)**.5
    header.distance.near = 0
    header.distance.far = max(nx,ny) if is_incribed else (nx**2 + ny**2)**.5
    
    return header

class _Header():
    def __init__(self):
        # set Mode
        self.mode = 0 #parallel
        # set Object
        self.object = Struct(size    = Struct(x=None, y=None, z=None), 
                             spacing = Struct(x=1, y=1, z=1), 
                             length  = Struct(x=None, y=None, z=None), 
                             offset  = Struct(x=0, y=0, z=0),
                             rotation = Struct(azi=None, ele=None))
        self.detector = Struct(size     = Struct(u=None, v=None), 
                               spacing  = Struct(u=1, v=1), 
                               length   = Struct(u=None, v=None), 
                               offset   = Struct(u=0, v=0))
        self.distance = Struct(source2object=None, source2detector=None, near=None, far=None)
    def __str__(self):
        output = ''
        if self.mode:
            output += 'mode                    : cone\n'
        else:
            output += 'mode                    : parallel\n'
        output +=     'object size     (voxel) : ({}, {} ,{})\n'.format(self.object.size.x, self.object.size.y, self.object.size.z)
        output +=     'object spacing     (mm) : ({}, {} ,{})\n'.format(self.object.spacing.x, self.object.spacing.y, self.object.spacing.z)
        output +=     'object length      (mm) : ({}, {} ,{})\n'.format(self.object.length.x, self.object.length.y, self.object.length.z)
        output +=     'detector size   (pixel) : ({}, {})\n'.format(self.detector.size.u, self.detector.size.v)
        output +=     'detector spacing   (mm) : ({}, {})\n'.format(self.detector.spacing.u, self.detector.spacing.v)
        output +=     'detector length    (mm) : ({}, {})\n'.format(self.detector.length.u, self.detector.length.v)
        output +=     'source to detector (mm) : {}\n'.format(self.distance.source2object)
        output +=     'source to detector (mm) : {}\n'.format(self.distance.source2detector)
        return output
        

def getConfig(name_config):
    with open('./nerf/configs/{}.yml'.format(name_config), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return _Config(name_config, config)

class _Config():
    def __init__(self, name_config, config):
        self.name = name_config
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
        output += 'step              : {}\n'.format(self.step)
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
