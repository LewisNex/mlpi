import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import math

from scipy.optimize import minimize_scalar
import skimage.transform as ski

from matplotlib.colors import LogNorm

def show_images(images, log_color=False):
    fig = plt.figure()
    num_imgs = len(images)
#     axes = [fig.add_subplot(101 + n + 10*num_imgs, label=label) for n, label in enumerate(["Hadrons", "Leptons", "Total"])]
    axes = [fig.add_subplot(101 + n + 10*num_imgs) for n in range(num_imgs)]
    for image, axis in zip(images, axes):
        image = image + 0.001 # Fix logging of zeros
        if log_color:
            axis.imshow(image, norm=LogNorm(vmin=image.min(), vmax=image.max()))
        else:
            axis.imshow(image)
    plt.show()

def variance_angle(img, beta=1):
    # Turn the his,  togram into a set of points centred on the middle pixel
    x_max, y_max = img.shape
    points = np.asarray([[x-x_max/2 , y-y_max/2, img[int(x), int(y)]] 
                         for x in range(x_max)
                         for y in range(y_max)
                         if img[int(x), int(y)] > 0])
    mean_alpha = np.mean(np.arctan2(points[:,1] , (points[:,0])))
    return mean_alpha
    
def img_centre(img, ptmean=True):
    # Turn the histogram into a set of points centred on the middle pixel
    x_max, y_max = img.shape
    if ptmean:
        tot = np.sum(img)
        points = np.asarray([[(x-x_max/2) * img[int(x), int(y)] / tot,(y-y_max/2) * img[int(x), int(y)] / tot]
                             for x in range(x_max)
                             for y in range(y_max)])
        c_x, c_y = np.sum(points, axis=0)
        return c_x, c_y
    else:
        m_x, m_y = np.argwhere(img == np.max(img))[0]
        return math.ceil(m_x - x_max/2), math.ceil(m_y - y_max/2)
    
    
def correct_image(img, ptmean=True, rotate=True, transform_layers_together=True):
    # split image into layers
    if len(img.shape) > 2:
        layers = [img[:,:,i] for i in range(len(img[0,0,:]))]
    else:
        layers = [img]
        
    corrected = np.zeros(img.shape)
    
    if transform_layers_together:
        xc, yc = img_centre(img.sum(axis=2), ptmean)
        for i, layer in enumerate(layers):
            layers[i] = ski.warp(layer, ski.AffineTransform(translation=(1*yc, 1*xc)), order=0)
        if rotate:
            angle = (variance_angle(np.asarray(layers).sum(axis=0)) * 180/np.pi)
            for i, layer in enumerate(layers):
                layers[i] = ski.rotate(layer, -angle, order=0)
        for i, layer in enumerate(layers):
            corrected[:,:,i] = layer
        return corrected / np.sum(corrected)
    else:
        for i, layer in enumerate(layers):
            xc, yc = img_centre(layer, ptmean)
            layer = ski.warp(layer, ski.AffineTransform(translation=(1*yc, 1*xc)), order=0)
            if rotate:
                angle = (variance_angle(layer) * 180/np.pi)
                layer = ski.rotate(layer, -angle, order=0)
            corrected[:,:,i] = layer
        return corrected / np.sum(corrected)