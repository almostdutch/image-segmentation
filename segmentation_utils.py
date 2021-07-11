#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segmentation_utils.py

"""

import matplotlib
import numpy as np
from scipy.signal import correlate
import random
from sklearn.cluster import KMeans

def GaussianKernel1D(sigma):
    '''Returns 1D Gaussian kernel
    
    sigma: standard deviation of normal distribution
    '''
    
    kernel_size = 6 * sigma + 1;    
    kn = int((kernel_size - 1) / 2);
    X = np.arange(-kn, kn + 1, 1);
    
    kernel = np.exp(-(np.power(X, 2)) / (2 * sigma ** 2));
    kernel = kernel / kernel.sum();
    kernel = kernel.reshape(len(kernel), 1);
    
    return kernel;

def Denoising(img_in, sigma):
    '''Returns an image filtered with a Gaussian kernel
    
    sigma: standard deviation of normal distribution
    '''
       
    kernel = GaussianKernel1D(sigma);
    kernel_x = kernel;
    kernel_y = kernel.T;   
    
    img_out =  correlate(img_in, kernel_x, mode = 'same', method = 'auto');
    img_out =  correlate(img_out, kernel_y, mode = 'same', method = 'auto');
    
    return img_out;

def cmap_discretize(N):
    """Returns a discrete colormap from the continuous colormap cmap (jet).
    
    N: number of colors.
    """
    
    cmap = matplotlib.cm.jet;
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024);

def CountPixels(image, N_levels): 
    """Returns a tupil (pixel count, normalized pixel count)
    
    N_levels: number of intensity levels, N_levels = 2 ** bpp (bits per pixel)
    """       
    
    pixel_count = np.zeros((N_levels, 1));
    for i in range(N_levels):
        pixel_count[i] = np.sum(image == i);
        
    pixel_count_normalized = pixel_count / (image.shape[0] * image.shape[1]);
    
    return pixel_count, pixel_count_normalized;

def GlobalThresholding(image, kernel_sigma, N_levels, N_classes, step = 1):
    """Returns a segmentation mask with N_classes classes
    
    image: original image
    kernel_sigma: standard deviation of normal distribution
    N_levels: number of intensity levels, N_levels = 2 ** bpp (bits per pixel)
    N_classes: number of classes [2, 4]
    step: iteration step, controls segmentation accuracy [0, N]
    """       
    
    if kernel_sigma >= 1:
        image = Denoising(image, kernel_sigma);
        
    pixel_count, pixel_count_normalized = CountPixels(image, N_levels);
    mean_g = image.mean(); # global mean

    if N_classes == 2:   
        interclass_var = np.zeros((N_levels)); # inter-class variance
        range_array = np.arange(0, N_levels, 1).reshape(N_levels, 1);
        for ii in range(0, N_levels - 1, step): 

            threshold = ii;
            
            mask_1 = range_array <= threshold;
            mask_2 = range_array > threshold;
            
            p_1 = pixel_count_normalized[mask_1].sum(); # probability of class 1
            p_2 = 1 - p_1; # probability of class 2
            
            mean_1 = 1 / p_1 * np.sum(range_array[mask_1] * pixel_count_normalized[mask_1]); # mean of class 1
            mean_2 = 1 / p_2 * np.sum(range_array[mask_2] * pixel_count_normalized[mask_2]); # mean of class 2
            
            temp = p_1 * (mean_1 - mean_g) ** 2 + p_2 * (mean_2 - mean_g) ** 2;
            interclass_var[ii] = np.nan_to_num(temp);
            
        threshold = np.argmax(interclass_var);
        mask_1 = image <= threshold;
        mask_2 = image > threshold;
        mask = np.zeros(image.shape);
        mask[mask_1] = 0;
        mask[mask_2] = 1;
        return mask;
    elif N_classes == 3:
        interclass_var = np.zeros((N_levels, N_levels)); # inter-class variance
        range_array = np.arange(0, N_levels, 1).reshape(N_levels, 1);
        for ii in range(0, N_levels - 2, step): 
            for jj in range(ii + 1, N_levels - 1, step):

                threshold1 = ii;
                threshold2 = jj;
                
                mask_1 = range_array <= threshold1;
                mask_2 = (range_array > threshold1) * (range_array <= threshold2);
                mask_3 = range_array > threshold2;
                
                p_1 = pixel_count_normalized[mask_1].sum(); # probability of class 1
                p_2 = pixel_count_normalized[mask_2].sum(); # probability of class 2
                p_3 = 1 - (p_1 + p_2); # probability of class 3
                
                mean_1 = 1 / p_1 * np.sum(range_array[mask_1] * pixel_count_normalized[mask_1]); # mean of class 1
                mean_2 = 1 / p_2 * np.sum(range_array[mask_2] * pixel_count_normalized[mask_2]); # mean of class 2
                mean_3 = 1 / p_3 * np.sum(range_array[mask_3] * pixel_count_normalized[mask_3]); # mean of class 3
                
                temp = p_1 * (mean_1 - mean_g) ** 2 + p_2 * (mean_2 - mean_g) ** 2 + p_3 * (mean_3 - mean_g) ** 2;
                interclass_var[ii, jj] = np.nan_to_num(temp);
            
        threshold = np.unravel_index(np.argmax(interclass_var, axis=None), interclass_var.shape);
        threshold1 = threshold[0];
        threshold2 = threshold[1];
        
        mask_1 = image <= threshold1;
        mask_2 = (image > threshold1) * (image <= threshold2);
        mask_3 = image > threshold2;
        mask = np.zeros(image.shape);
        mask[mask_1] = 0;
        mask[mask_2] = 1;
        mask[mask_3] = 2;
        return mask;
    elif N_classes == 4:
        interclass_var = np.zeros((N_levels, N_levels, N_levels)); # inter-class variance
        range_array = np.arange(0, N_levels, 1).reshape(N_levels, 1);
        for ii in range(0, N_levels - 3, step): 
            for jj in range(ii + 1, N_levels - 2, step):
                for kk in range(jj + 1, N_levels - 1, step):   
                        
                    threshold1 = ii;
                    threshold2 = jj;
                    threshold3 = kk;
                    
                    mask_1 = range_array <= threshold1;
                    mask_2 = (range_array > threshold1) * (range_array <= threshold2);
                    mask_3 = (range_array > threshold2) * (range_array <= threshold3);        
                    mask_4 = range_array > threshold3;
                    
                    p_1 = pixel_count_normalized[mask_1].sum(); # probability of class 1
                    p_2 = pixel_count_normalized[mask_2].sum(); # probability of class 2
                    p_3 = pixel_count_normalized[mask_3].sum(); # probability of class 3
                    p_4 = 1 - (p_1 + p_2 + p_3); # probability of class 4
                    
                    mean_1 = 1 / p_1 * np.sum(range_array[mask_1] * pixel_count_normalized[mask_1]); # mean of class 1
                    mean_2 = 1 / p_2 * np.sum(range_array[mask_2] * pixel_count_normalized[mask_2]); # mean of class 2
                    mean_3 = 1 / p_3 * np.sum(range_array[mask_3] * pixel_count_normalized[mask_3]); # mean of class 3
                    mean_4 = 1 / p_4 * np.sum(range_array[mask_4] * pixel_count_normalized[mask_4]); # mean of class 4
                    
                    temp = p_1 * (mean_1 - mean_g) ** 2 + p_2 * (mean_2 - mean_g) ** 2 + \
                        p_3 * (mean_3 - mean_g) ** 2 + p_4 * (mean_4 - mean_g) ** 2;
                    interclass_var[ii, jj, kk] = np.nan_to_num(temp);
            
        threshold = np.unravel_index(np.argmax(interclass_var, axis=None), interclass_var.shape);
        threshold1 = threshold[0];
        threshold2 = threshold[1];
        threshold3 = threshold[2];
        
        mask_1 = image <= threshold1;
        mask_2 = (image > threshold1) * (image <= threshold2);
        mask_3 = (image > threshold2) * (image <= threshold3);
        mask_4 = image > threshold3;
        mask = np.zeros(image.shape);
        mask[mask_1] = 0;
        mask[mask_2] = 1;
        mask[mask_3] = 2;
        mask[mask_4] = 3;
        return mask;
    else:
        print('max supported N_class == 4. Abort..\n')
        return None;
    
def SpatiallyAdaptiveBinaryThresholding(image, kernel_sigma, a, b):
    """Returns a segmentation mask with 2 classes
    
    image: original image
    kernel_sigma: standard deviation of normal distribution    
    a: coefficient for local standard deviation
    b: coefficient for local mean
    """       

    if kernel_sigma >= 1:
        image = Denoising(image, kernel_sigma);  
        
    nr, nc = image.shape;
    mask = np.zeros(image.shape);
    
    for ii in range(1, nr - 1):
        for jj in range(1, nc - 1):
            nb = image[ii - 1:ii + 2, jj - 1:jj + 2];
            nb = nb.reshape(3, 3);
            nb_std = np.std(nb);
            nb_mean = np.mean(nb);
            
            if image[ii, jj] > a * nb_std and image[ii, jj] > b * nb_mean:
                mask[ii, jj] = 1;

    return mask;
    
def MovigAverageBinaryThresholding(image, kernel_sigma, n, b):
    """Returns a segmentation mask with 2 classes

    image: original image
    kernel_sigma: standard deviation of normal distribution    
    n: size of averaging kernel
    b: coefficient for local mean
    """       

    if kernel_sigma >= 1:
        image = Denoising(image, kernel_sigma);
        
    nr, nc = image.shape;
    mask = np.zeros(image.shape);
    
    for ii in range(0, nr):
        for jj in range(0, nc):
            if jj < n:
                nb_mean = image[ii, 0:jj + 1].sum() / n;
            else:
                nb = image[ii, jj - n + 1:jj + 1];
                nb_mean = np.mean(nb);
            if image[ii, jj] > b * nb_mean:
                mask[ii, jj] = 1;        
    
    return mask;

def EstimateIlluminationProfile(image, kernel_sigma):
    """Returns an estimated image illumination profile
    
    image: image with an inhomogeneous illumination profile
    kernel_sigma: standard deviation of normal distribution
    """ 
    
    illumination_profile = Denoising(image, kernel_sigma);
    
    map_corr = np.ones(image.shape);
    map_corr = Denoising(map_corr, kernel_sigma);
    
    illumination_profile = illumination_profile / map_corr;
    
    return illumination_profile;

def RegionGrowingSegmentation(image, kernel_sigma, seed_row, seed_col, Niter, sim_threshold):
    """Returns a segmentation mask with 2 classes
    
    image: original image
    kernel_sigma: standard deviation of normal distribution    
    seed_row: row position of a seed pixel
    seed_col: column position of a seed pixel
    Niter: number of iterations
    sim_threshold: similarity threshold [0, 1]
    """ 
    
    if kernel_sigma >= 1:
        image = Denoising(image, kernel_sigma);
        
    image = image / image.max();
    nr, nc = image.shape;
    mask_old = np.zeros(image.shape, dtype = bool);
    mask_old[seed_row, seed_col] = True;
        
    sim_pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)];
    nb_pos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)];
    
    for iter_no in range(Niter):
        mask = np.zeros(image.shape, dtype = bool);
        for row_no in range(1, nr - 1):
            for col_no in range(1, nc - 1):
                if mask_old[row_no, col_no] != 1:
                    continue;
                
                nb = image[row_no - 1:row_no + 2, col_no - 1:col_no + 2];                
                region_mean = image[mask_old].mean();
                if region_mean == 1:
                    mask[row_no - 1:row_no + 2, col_no - 1:col_no + 2] = 1;   
                    continue;
                
                similarity = np.exp(-(nb - region_mean) ** 2);            
                for el_no in range(len(nb_pos)):
                    delta_row, delta_col = nb_pos[el_no];
                    row_sim, col_sim = sim_pos[el_no];
                    if similarity[row_sim, col_sim] < sim_threshold:
                        continue;
                        
                    mask[row_no + delta_row, col_no + delta_col] = 1;
                    
        if mask_old.sum() == mask.sum():
            break; 
            
        mask_old = mask;
        
    return mask;
        
def ColorSegmentation(image, kernel_sigma, color_seg, sim_threshold):
    """Returns a segmentation mask with 2 classes
    
    image: original image
    kernel_sigma: standard deviation of normal distribution    
    color_seg: a tupil of three RGB values
    Niter: number of iterations
    sim_threshold: similarity threshold [0.3, 1]
    """ 
    
    color_seg = np.array(color_seg) / 255;
    
    if kernel_sigma >= 1:
        for cha_no in range(image.shape[2]):
            image[:, :, cha_no] = Denoising(image[:, :, cha_no], kernel_sigma);
        
    image = image / 255;
    mask = np.zeros((image.shape[0], image.shape[1]), dtype = bool);
        
    similarity = np.exp(-np.sum((image - color_seg) ** 2, axis = 2));
    mask[similarity > sim_threshold] = 1;

    return mask;

def KmeansSegmentation(image, kernel_sigma, N_classes, N_iter = 1, tol = 10e-6):
    """Returns a segmentation mask with N_classes classes
    
    image: original image
    kernel_sigma: standard deviation of normal distribution    
    N_classes: number of classes
    Niter: number of iterations
    tol: tolerance
    """ 

    if kernel_sigma >= 1:
        image = Denoising(image, kernel_sigma);
    
    nr, nc = image.shape;
    image_vec = image.reshape(nr * nc, 1);
    mask_pos = image_vec > 0;
    X = image_vec[mask_pos].reshape(mask_pos.sum(), 1);
    kmeans = KMeans(n_clusters = N_classes, random_state=0, max_iter = N_iter, tol = tol).fit(X);
    labels = kmeans.labels_;    
    
    mask = np.zeros((nr * nc, 1));    
    mask[mask_pos] = labels;
    mask = mask.reshape(nr, nc);
    
    return mask;