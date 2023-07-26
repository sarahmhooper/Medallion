# Copyright (c) 2021 Sen Wu. All Rights Reserved.

import csv
import logging
import random

import numpy as np
from skimage import transform

from dauphin.image_segmentation.config import KEY_DELIMITER

logger = logging.getLogger(__name__)


def hist_scaled(img, brks=None):
    """ HIST EQ CODE FROM: https://github.com/fastai/fastai/blob/master/fastai/medical/imaging.py """
    
    def array_freqhist_bins(self, n_bins=100):
        imsd = np.sort(self.flatten())
        t = np.array([0.001])
        t = np.append(t, np.arange(n_bins) / n_bins + (1 / 2 / n_bins))
        t = np.append(t, 0.999)
        t = (len(imsd) * t + 0.5).astype(int)
        return np.unique(imsd[t])

    if brks is None:
        brks = array_freqhist_bins(img)
    ys = np.linspace(0.0, 1.0, len(brks))
    x = img.flatten()
    x = np.interp(x, brks, ys)
    return x.reshape(img.shape).clip(0.0, 1.0)


def custom_preproc_op(img, crop_dim1, crop_dim2, resize=True, hist=False, order=1):
    """
    Custom preprocessing operation to apply histogram equalization and center cropping or resizing in sensible order
    """
    img = img.copy()
    dim_1, dim_2, dim_sl = img.shape[0], img.shape[1], img.shape[2]
    dim_extra = len(img.shape)-2

    if crop_dim1 == dim_1 and crop_dim2 == dim_2:
        
        if not hist:
            return img
        else:
            for sl in range(dim_sl):
                img[:, :, sl] = hist_scaled(img[:, :, sl])
            return img

    elif resize:
        
        if dim_extra>1: 
            resize_size = (crop_dim1,crop_dim2,1,*img.shape[3:])
        else: 
            resize_size = (crop_dim1,crop_dim2,1)

        new_img = np.zeros((crop_dim1,crop_dim2,dim_sl,*img.shape[3:]))
        for sl in range(dim_sl): 
            if not hist:
                new_img[:, :, sl:sl+1] = transform.resize(img[:, :, sl:sl+1],resize_size,order=order)
            else:
                new_img[:, :, sl:sl+1] = hist_scaled(transform.resize(img[:, :, sl:sl+1],resize_size,order=order))
        return new_img
        
        
    else:
    
        start_dim1 = dim_1 // 2 - (crop_dim1 // 2)
        start_dim2 = dim_2 // 2 - (crop_dim2 // 2)

        if start_dim1 > 0 and start_dim2 > 0:
            img = img[start_dim1 : start_dim1 + crop_dim1, start_dim2 : start_dim2 + crop_dim2]
        else:
            if start_dim1 > 0:
                img = img[start_dim1 : start_dim1 + crop_dim1]
            if start_dim2 > 0:
                img = img[:, start_dim2 : start_dim2 + crop_dim2]

        if hist:
            for sl in range(dim_sl):
                img[:, :, sl] = hist_scaled(img[:, :, sl])

        if start_dim1 < 0 and start_dim2 < 0:

            img = np.pad(
                img,
                (
                    ((-1 * start_dim1, crop_dim1 + start_dim1 - dim_1),
                    (-1 * start_dim2, crop_dim2 + start_dim2 - dim_2)) + 
                    ((0, 0),)*dim_extra
                ),
            )


        else:

            if start_dim1 < 0:
                img = np.pad(
                    img,
                    (
                        ((-1 * start_dim1, crop_dim1 + start_dim1 - dim_1), 
                        (0,0))+
                        ((0, 0),)*dim_extra
                    ),
                )
            elif start_dim1 == 0 and dim_1 != crop_dim1:  # Crop differs by 1 from dimension
                if dim_1 == crop_dim1 - 1:
                    img = np.pad(img, (((0, 1), (0, 0)) + ((0, 0),)*dim_extra))
                elif dim_1 == crop_dim1 + 1:
                    img = img[:-1]

            if start_dim2 < 0:
                img = np.pad(
                    img,
                    (
                        ((0, 0),
                        (-1 * start_dim2, crop_dim2 + start_dim2 - dim_2))+
                        ((0, 0),)*dim_extra
                    ),
                )
            elif start_dim2 == 0 and dim_2 != crop_dim2:  # Crop differs by 1 from dimension
                if dim_2 == crop_dim2 - 1:
                    img = np.pad(img, (((0, 0), (0, 1)) + ((0, 0),)*dim_extra))
                elif dim_2 == crop_dim2 + 1:
                    img = img[:, :-1]


        return img

