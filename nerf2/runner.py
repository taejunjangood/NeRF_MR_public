import os, glob, datetime, imageio
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange
from IPython import display
from ipywidgets import Output
out = Output()
from .network import *
from .parameter import *
from .dataloader import *

class ImageInfo():
    def __init__(self, size, scale):
        self.size = size
        self.scale = scale
        
class DataInfo():
    def __init__(self, name, data, angles, af, mode):
        self.name = name
        self.data = data
        self.na, self.nv, self.nu = data.shape
        self.angles = angles
        self.af = af
        self.mode = mode

class Runner():
    def __init__(self, data_info:DataInfo, config_name:str, device=0):
        self.data_info = data_info
        self.config = getConfig(config_name)
        self.model = getModel(self.config)
        self.encoder = getPositionalEncoder(self.config)
        if device == -1:
            device = 'cpu'
        self.device = device
    
    def train(self, notebook=False, loss_threshold=0):
        # set save path
        path = './nerf2/results'
        S = [self.config.name, 
             self.data_info.name, 
             'af_{0:02d}'.format(self.data_info.af), 
             self.data_info.mode,
             datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")]    
        for s in S:
            if type(s) is int:
                path = path + '/{0:02d}'.format(s)
            else:
                path = path + '/' + s
            if not os.path.isdir(path):
                os.mkdir(path)
        print('training')
        print('name     : {}'.format(self.data_info.name))
        print('AF       : {0:02d}'.format(self.data_info.af))
        print('# angles : {0:03d}'.format(len(self.data_info.angles)))
        print('mode     : {}'.format(self.data_info.mode))

        nu, nv = self.data_info.nu, self.data_info.nv
        header = getHeader()
        header.detector.spacing.set([1., 1.])
        header.detector.size.set([nu, nv])
        header.detector.length.set([nu, nv])
        header.source.distance.source2object = nu/2
        header.source.distance.source2detector = nu
        header.source.distance.near = 0
        header.source.distance.far = nu

        # get dataloader
        dataloader = getDataloader('train', self.config, header, self.data_info)
        # set total step
        total_step = self.config.step
        # set training optimizer
        optimizer = torch.optim.Adam(params=list(self.model.parameters()), 
                                     lr=self.config.optimizer.learning_rate, 
                                     betas=(0.9, 0.999))
        # set loss fuction MSE
        loss_func = lambda x,y: torch.mean((x-y)**2)
        # train
        self.model.to(self.device)
        self.model.train()
        list_loss = np.zeros(total_step)
        trng = trange(1,total_step+1)
        for step in trng:
            # get batches in one step
            minibatches = dataloader.generateBatch(shuffle=True)
            for minibatch in minibatches:
                # get minibatch
                ray_samples, ray_sample_intervals, label = minibatch
                # ray_samples /= (header.source.distance.far - header.source.distance.near)
                # convert to tensor
                ray_samples = self.encoder(torch.tensor(ray_samples, dtype=torch.float, device=self.device))
                ray_sample_intervals = torch.tensor(ray_sample_intervals, dtype=torch.float, device=self.device)
                label = torch.tensor(label, dtype=torch.float, device=self.device)
                # forward
                output = self.model(ray_samples)
                # summation
                predict = torch.sum(ray_sample_intervals * output[...,0], axis=-1)
                # backward
                loss = loss_func(predict, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if loss.item() > loss_threshold:
                os.rmdir(path)
                print('[{0}] ... loss  : {1}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), loss.item()))
                print('retry training')
                return True
            # update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.config.optimizer.learning_rate_decay
            
            # print loss
            trng.set_postfix(loss = loss.item())
            list_loss[step-1] = loss.item()
            
            
            # save weigth
            if step % self.config.freq.save_weight == 0:
                torch.save(self.model.state_dict(), os.path.join(path, 'step_{0:05d}.pt'.format(step)))
            if step % self.config.freq.save_image == 0:
                imageinfo = ImageInfo([nu,nu,1], [1,1,1])
                recon = self.infer(imageinfo)[0]
                recon = np.clip(recon, 0, 1)*255
                recon = recon[...,None].repeat(3, -1).astype(np.uint8)
                imageio.imwrite(path + '/step_{0:05d}.png'.format(step), recon)
                if notebook is True:
                    if step == self.config.freq.save_image:
                        display.display(out)
                    with out:
                        display.clear_output(wait=True)
                        fig, ax = plt.subplots(1,2)
                        fig.set_size_inches([10,5])
                        ax[0].imshow(recon)
                        ax[0].set_title('{0:05d}'.format(step))
                        # ax[0].axis('off')
                        ax[1].plot(np.arange(1,step+1), list_loss[:step])
                        ax[1].plot(step,list_loss[step-1], 'ro')
                        ax[1].set_title('Loss')
                        ax[1].text(step, list_loss[step-1], '{0:.4f}'.format(list_loss[step-1]))
                        ax[1].set_xlim([0,total_step+1])
                        plt.show()

        return False


        

    def infer(self, image_info:ImageInfo, ckpt_path=None):
        nx, ny, nz = image_info.size
        dx, dy, dz = image_info.scale
        sx, sy, sz = nx*dx, ny*dy, nz*dz
        nu = self.data_info.nu
        header = getHeader()
        header.object.size.set([nx, ny, nz])
        header.object.spacing.set([dx, dy, dz])
        header.object.length.set([sx, sy, sz])
        header.source.distance.source2detector = nu
        header.source.distance.source2object = nu / 2
        header.source.distance.near = 0
        header.source.distance.far = nu

        dataloader = getDataloader('infer', self.config, header)
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))
        self.model.to(self.device)
        self.model.eval()
        minibatches = dataloader.generateBatch()
        array = []
        with torch.no_grad():
            for minibatch in minibatches:
                inputs = self.encoder(torch.tensor(minibatch, device=self.device, dtype=torch.float))
                outputs = self.model(inputs)
                array.append(outputs.detach().cpu().numpy())

            return np.concatenate(array).reshape(nx,ny,nz).T

# class Runner():
#     def __init__(self, image_info, config_name, device=0):
#         self.header = getHeader(image_shape=image_info[0], is_incribed=image_info[1])
#         self.name_data = image_info[2]
#         self.z_slice = image_info[3]
#         self.config = getConfig(config_name)
#         self.model = getModel(self.config)
#         self.encoder = getPositionalEncoder(self.config)
#         if device == -1:
#             device = 'cpu'
#         self.device = device
        
#     def train(self, data, data_info, notebook=False):
#         # set save path
#         angles, AF, mode = data_info
#         path = './nerf/results'
#         S = [self.config.name, 
#              self.name_data, 
#              'slice_{0:02d}'.format(self.z_slice), 
#              'af_{0:02d}'.format(AF), 
#              mode, 
#              datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")]    
#         for s in S:
#             if type(s) is int:
#                 path = path + '/{0:02d}'.format(s)
#             else:
#                 path = path + '/' + s
#             if not os.path.isdir(path):
#                 os.mkdir(path)
#         print('training')
#         print('name     : {}'.format(self.name_data))
#         print('slice    : {0:02d}'.format(self.z_slice))
#         print('AF       : {0:02d}'.format(AF))
#         print('# angles : {0:03d}'.format(len(angles)))
#         print('mode     : {}'.format(mode))
#         # get dataloader
#         dataloader = getDataloader('train', self.header, self.config, data, angles)
#         # set total step
#         total_step = self.config.step
#         # set training optimizer
#         optimizer = torch.optim.Adam(params=list(self.model.parameters()), 
#                                      lr=self.config.optimizer.learning_rate, 
#                                      betas=(0.9, 0.999))
#         # set loss fuction MSE
#         loss_func = lambda x,y: torch.mean((x-y)**2)
#         # train
#         self.model.to(self.device)
#         self.model.train()
#         list_loss = np.zeros(total_step)
#         trng = trange(1,total_step+1)
#         for step in trng:
#             # get batches in one step
#             minibatches = dataloader.generateBatch(shuffle=True)
#             for minibatch in minibatches:
#                 # get minibatch
#                 ray_samples, ray_sample_intervals, label = minibatch
#                 ray_samples /= (self.header.source.distance.far - self.header.source.distance.near)
#                 # convert to tensor
#                 ray_samples = self.encoder(torch.tensor(ray_samples, dtype=torch.float, device=self.device))
#                 ray_sample_intervals = torch.tensor(ray_sample_intervals, dtype=torch.float, device=self.device)
#                 label = torch.tensor(label, dtype=torch.float, device=self.device)
#                 # forward
#                 output = self.model(ray_samples)
#                 # summation
#                 predict = torch.sum(ray_sample_intervals * output[...,0], axis=-1)
#                 # backward
#                 loss = loss_func(predict, label)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             if loss.item() > 20000:
#                 os.rmdir(path)
#                 print('[{0}] ... loss  : {1}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), loss.item()))
#                 print('retry training')
#                 return True
#             # update learning rate
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] *= self.config.optimizer.learning_rate_decay
            
#             # print loss
#             trng.set_postfix(loss = loss.item())
#             list_loss[step-1] = loss.item()
            
            
#             # save weigth
#             if step % self.config.freq.save_weight == 0:
#                 torch.save(self.model.state_dict(), os.path.join(path, 'step_{0:05d}.pt'.format(step)))
#             if step % self.config.freq.save_image == 0:
#                 recon = self.infer(data_info)[0]
#                 recon = np.clip(recon, 0, 1)*255
#                 recon = recon[...,None].repeat(3, -1).astype(np.uint8)
#                 imageio.imwrite(path + '/step_{0:05d}.png'.format(step), recon)
#                 if notebook is True:
#                     if step == self.config.freq.save_image:
#                         display.display(out)
#                     with out:
#                         display.clear_output(wait=True)
#                         fig, ax = plt.subplots(1,2)
#                         fig.set_size_inches([10,5])
#                         ax[0].imshow(recon)
#                         ax[0].set_title('{0:05d}'.format(step))
#                         # ax[0].axis('off')
#                         ax[1].plot(np.arange(1,step+1), list_loss[:step])
#                         ax[1].plot(step,list_loss[step-1], 'ro')
#                         ax[1].set_title('Loss')
#                         ax[1].text(step, list_loss[step-1], '{0:.4f}'.format(list_loss[step-1]))
#                         ax[1].set_xlim([0,total_step+1])
#                         plt.show()

#         return False

#     def infer(self, data_info, exam=None):
#         # check ckpt
#         if exam is None:
#             pass
#         else:
#             exam_No, ckpt_No = exam
#             _, AF, mode = data_info
#             S = [self.config.name, 
#                 self.name_data, 
#                 'slice_{0:02d}'.format(self.z_slice), 
#                 'af_{0:02d}'.format(AF), 
#                 mode
#                 ]    
#             path = './nerf/results'
#             for s in S:
#                 path = os.path.join(path, s)
#                 if not os.path.isdir(path):
#                     raise ValueError("There is no path \"{}\".".format(path))
#             path = sorted(glob.glob(path + '/*'))[exam_No]
#             # load ckpt
#             if type(ckpt_No) is int:
#                 ckpt_files = sorted(glob.glob(path + '/*.pt'))
#                 self.model.load_state_dict(torch.load(ckpt_files[ckpt_No]))
#             elif type(ckpt_No) is str:
#                 ckpt_files = sorted(glob.glob(path + '/*{}*.pt'.format(ckpt_No)))
#                 if len(ckpt_files) > 1:
#                     print("There are many ckpt files.")
#                     for name in ckpt_files:
#                         print(name)
#                     raise ValueError("Please choose a specific ckpt name")
#                 self.model.load_state_dict(torch.load(ckpt_files[0]))

#         # get dataloader
#         dataloader = getDataloader('infer', self.header, self.config)
        
#         self.model.to(self.device)
#         self.model.eval()
#         minibatches = dataloader.generateBatch()
#         array = []
#         with torch.no_grad():
#             for minibatch in minibatches:
#                 inputs = self.encoder(torch.tensor(minibatch, device=self.device, dtype=torch.float))
#                 outputs = self.model(inputs)
#                 array.append(outputs.detach().cpu().numpy())
#             nx = self.header.object.size.x
#             ny = self.header.object.size.y
#             nz = self.header.object.size.z
            
#             return np.concatenate(array).reshape(nx,ny,nz).T
