import numpy as np

def getMinimalSpokes(image_shape):
    if len(image_shape) == 3:
        nz, ny, nx = image_shape
        if nz == 1:
            image_shape = [ny, nx]
        else:
            raise ValueError('Image is not 2D.')
    return int(np.pi/2 * max(image_shape))

def getAngles(num, mode, start_angle=None, end_angle=None):
    if mode not in ['uniform', 'golden', 'stratified', 'random', 'limited']:
        raise ValueError('{} is not supported.'.format(mode))
    if mode == 'golden':
        GR = (1+5**.5)/2
        return np.arange(num) * np.pi/GR
    else:
        if (type(start_angle) or type(end_angle)) is None:
            raise ValueError('Enter start angle or end angle.')
        if mode == 'uniform' or mode == 'limited':
            return np.linspace(start_angle, end_angle, num+1)[:-1]
        elif mode == 'stratified':
            uniform = np.linspace(0,1,num+1)[:-1]
            perturb = np.random.rand(num) / num
            return uniform + perturb
        elif mode == 'random':
            return np.sort(np.random.rand(num)*(end_angle-start_angle)) + start_angle