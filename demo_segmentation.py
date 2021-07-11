"""
demo_segmentation.py

    
    (1) Global thresholding
        
    (2) Spatially adaptive binary thresholding
        
    (3) Moving average binary thresholding
    
    (4) Region growing segmentation
    
    (5) color segmentation
    
    (6) kmeans segmentation

"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg 
from segmentation_utils import cmap_discretize, GlobalThresholding, \
    SpatiallyAdaptiveBinaryThresholding, MovigAverageBinaryThresholding, \
    EstimateIlluminationProfile, RegionGrowingSegmentation, ColorSegmentation, KmeansSegmentation

fontsize = 32;

# load sample data
bpp = 8; # bits per pixel
N_levels = 2 ** bpp; # number of intensity levels

# cells
image = np.array(mpimg.imread('sample_data/cells.tif'));
image_cells = image.astype(np.uint8);

# coins
image = np.array(mpimg.imread('sample_data/coins.tif'));
image_coins = image.astype(np.uint8);

# text
image = np.array(mpimg.imread('sample_data/text.tif'));
image_text = image.astype(np.uint8);
intensity_grad = np.linspace(1,250, image_text.shape[1]);
intensity_grad = intensity_grad / intensity_grad.max();
image_text_grad = image_text * intensity_grad; # adding intensity gradient to image
image_text_grad[image_text_grad < 0] = 0;
image_text_grad[image_text_grad > N_levels - 1] = N_levels - 1;

# nebula
image = np.array(mpimg.imread('sample_data/nebula.jpeg'));
image_nebula = image.astype(np.uint8);

# brain
image = np.array(mpimg.imread('sample_data/brain.tif'));
image_brain = image.astype(np.uint8);


# Global thresholding
image = image_cells;
kernel_sigma = 0;
N_classes = 2;
step = 1;
mask_class2 = GlobalThresholding(image, kernel_sigma, N_levels, N_classes, step);

image = image_cells;
N_classes = 3;
step = 1;
mask_class3 = GlobalThresholding(image, kernel_sigma, N_levels, N_classes, step);

fig_width, fig_height = 40, 20;
fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, fig_height));
ax1.imshow(image, cmap='gray')
ax1.set_title("image", fontsize = fontsize)
ax1.set_axis_off()
discretized_jet = cmap_discretize(2);
img = ax2.imshow(mask_class2, cmap=discretized_jet);
plt.colorbar(img, ax=ax2)
ax2.set_title("segmentation", fontsize = fontsize)
ax2.set_axis_off()
discretized_jet = cmap_discretize(3);
img = ax3.imshow(mask_class3, cmap=discretized_jet);
plt.colorbar(img, ax=ax3)
ax3.set_title("segmentation", fontsize = fontsize)
ax3.set_axis_off()
plt.tight_layout()


# Spatially adaptive binary thresholding
image = image_cells;
kernel_sigma = 0;
a = 4;
b = 0.5;
mask = SpatiallyAdaptiveBinaryThresholding(image, kernel_sigma, a, b);

fig_width, fig_height = 30, 20;
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height));
ax1.imshow(image, cmap='gray')
ax1.set_title("image", fontsize = fontsize)
ax1.set_axis_off()
cmap = colors.ListedColormap(['black', 'white']);
img = ax2.imshow(mask, cmap=cmap);
ax2.set_title("segmentation", fontsize = fontsize)
ax2.set_axis_off()
plt.tight_layout()


# Moving average binary thresholding
image = image_text_grad;
kernel_sigma = 31;
illumination_profile = EstimateIlluminationProfile(image, kernel_sigma);

fig_width, fig_height = 30, 20;
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height));
ax1.imshow(image, cmap='gray')
ax1.set_title("image", fontsize = fontsize)
ax1.set_axis_off()
ax2.imshow(illumination_profile, cmap='gray')
ax2.set_title("illumination image", fontsize = fontsize)
ax2.set_axis_off()
ax3.imshow(image / illumination_profile, cmap='gray')
ax3.set_title("image illumination correction", fontsize = fontsize)
ax3.set_axis_off()

image = image_text_grad;
kernel_sigma = 0;
n = 40;
b = 0.8;
mask =  MovigAverageBinaryThresholding(image / illumination_profile, kernel_sigma, n, b);

cmap = colors.ListedColormap(['black', 'white']);
img = ax4.imshow(mask, cmap=cmap);
ax4.set_title("segmentation", fontsize = fontsize)
ax4.set_axis_off()
plt.tight_layout()


# Region growing segmentation
image = image_coins;
kernel_sigma = 0;
seed_row = 120;
seed_col = 150;
Niter = 100;
sim_threshold = 0.92;
mask = RegionGrowingSegmentation(image, kernel_sigma, seed_row, seed_col, Niter, sim_threshold);

fig_width, fig_height = 30, 20;
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, fig_height));
ax1.imshow(image, cmap='gray')
ax1.set_title("image", fontsize = fontsize)
ax1.set_axis_off()
cmap = colors.ListedColormap(['black', 'white']);
img = ax2.imshow(mask, cmap=cmap);
ax2.set_title("segmentation", fontsize = fontsize)
ax2.set_axis_off()
ax3.imshow(image * mask, cmap='gray')
ax3.set_title("image masked", fontsize = fontsize)
ax3.set_axis_off()
plt.tight_layout()


# color segmentation
image = image_nebula;
kernel_sigma = 0;
# color_seg = (243, 92, 231); # pink
color_seg = (8, 45, 134); # blue
sim_threshold = 0.9;
mask = ColorSegmentation(image, kernel_sigma, color_seg, sim_threshold);

fig_width, fig_height = 30, 20;
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(fig_width, fig_height));
ax1.imshow(image)
ax1.set_title("image", fontsize = fontsize)
ax1.set_axis_off()
cmap = colors.ListedColormap(['black', 'white']);
img = ax2.imshow(mask, cmap=cmap);
ax2.set_title("segmentation", fontsize = fontsize)
ax2.set_axis_off()
ax3.imshow(image * mask[:, :, None])
ax3.set_title("image masked", fontsize = fontsize)
ax3.set_axis_off()
plt.tight_layout()


# kmeans segmentation
image = image_brain;
kernel_sigma = 1;
N_classes = 3;
N_iter = 10;
tol = 10e-6;
mask = KmeansSegmentation(image, kernel_sigma, N_classes, N_iter, tol);

fig_width, fig_height = 30, 20;
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height));
ax1.imshow(image)
ax1.set_title("image", fontsize = fontsize)
ax1.set_axis_off()
discretized_jet = cmap_discretize(N_classes);
img = ax2.imshow(mask, cmap=discretized_jet);
plt.colorbar(img, ax=ax2)
ax2.set_title("segmentation", fontsize = fontsize)
ax2.set_axis_off()
plt.tight_layout()

