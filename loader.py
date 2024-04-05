import h5py
import numpy as np
from sigpy import sim

def loadNYU(filename):
    with h5py.File(filename, 'r') as meta:
        if 'reconstruction_rss' in meta.keys():
            image = np.array(meta['reconstruction_rss'])
        else:
            raw = np.array(meta['kspace'])
            image = np.sum(np.abs(np.fft.ifftn(raw, axes=[-1,-2]))**2, axis=-3)**.5
    image = (image-image.min()) / (image.max()-image.min())
    return image

def getPhantom(shape):
    return sim.shepp_logan(shape, np.float32)