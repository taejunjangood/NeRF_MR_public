import numpy as np
import sigpy as sp
import pyCT.forward as forward
from nerf2.parameter import getHeader
from nerf2.parameter import *
import pyCT

def makeMeasuredData(image, angles, is_incribed):
    nz, ny, nx = image.shape
    if nz != 1:
        raise ValueError("Image is not 2D.")

    header = getHeader()
    header.object.size.set([nx, ny, 1])
    header.object.length.set([float(nx), float(ny), 1.])
    header.object.spacing.set([1., 1., 1.])
    if is_incribed:
        nu = max(nx, ny)
    else:
        nu = int((nx**2 + ny**2)**.5)
        nu = nu if nu%2 == 0 else nu+1
    header.detector.size.set([nu, 1])
    header.detector.length.set([float(nu), 1.])
    header.detector.spacing.set([1., 1.])
    header.source.distance.source2object = nu/2
    header.source.distance.source2detector = float(nu)
    header.source.distance.near = 0.
    header.source.distance.far = float(nu)

    header.source.motion.set(angles)
    return pyCT.project(image, header)


def getRadialSampledImage(image, angles, num_readout=None, is_incribed=False, is_tight=True, **kwargs):
    if len(image.shape) == 3:
        nz, ny, nx = image.shape
        if nz != 1:
            raise ValueError('Image is not 2D.')
    elif len(image.shape) == 2:
        ny, nx = image.shape
    else:
        raise ValueError('Image is not 2D.')
    
    if 'return_raw' in kwargs.keys():
        return_raw = kwargs['return_raw']
    else:
        return_raw = False
    if 'return_coordinates' in kwargs.keys():
        return_coordinates = kwargs['return_coordinates']
    else:
        return_coordinates = False

    if is_incribed:
        diameter = max(nx, ny)
        diameter = diameter+1 if (diameter % 2) == 1 else diameter
        theta = np.arctan(ny/nx)
        my, mx = int(diameter * (1-np.sin(theta)))//2, int(diameter * (1-np.cos(theta)))//2
    else:
        diameter = int((ny**2 + nx**2)**.5)
        diameter = diameter+1 if (diameter % 2) == 1 else diameter
        my, mx = (diameter - np.array(image.shape)) // 2
        
    if num_readout is None:
        num_readout = diameter*2
    if num_readout % 2 == 1:
        r = np.linspace(-.5, .5, num_readout+2)[1:-1]
    else:
        r = np.linspace(-.5, .5, num_readout+1)[1:-1]
        num_readout -= 1
    
    r, theta = np.meshgrid(r, angles)
    num_trajectory = len(angles)

    coordinates = np.zeros((num_trajectory, num_readout, 2))
    coordinates[:, :, -1] = r * np.cos(theta)
    coordinates[:, :, -2] = r * np.sin(theta)
    coordinates = coordinates * [diameter, diameter]

    dcf = (coordinates[...,0]**2 + coordinates[...,1]**2) ** .5
    if is_incribed:
        undersampled_raw = sp.gridding(sp.nufft(image, coordinates)*dcf,
                                        coordinates,
                                        [diameter, diameter])
    else:
        undersampled_raw = sp.gridding(sp.nufft(np.pad(image, pad_width=((my,),(mx,))), coordinates)*dcf, 
                                    coordinates, 
                                    [diameter, diameter])
    undersampled_image = np.abs(np.fft.fftshift(np.fft.ifft2(undersampled_raw)))
    if is_tight:
        undersampled_image = undersampled_image[my:-my, mx:-mx]
        undersampled_raw = np.fft.fftshift(np.fft.fftshift(undersampled_raw)[my:-my, mx:-mx])
    else:
        undersampled_image = undersampled_image
        undersampled_raw = undersampled_raw

    if return_raw or return_coordinates:
        output = [undersampled_image]
        if return_raw:
            output.append(undersampled_raw)
        if return_coordinates:
            output.append(coordinates)
        return tuple(output)
    else:
        return undersampled_image