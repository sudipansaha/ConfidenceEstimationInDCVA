# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sudipan Saha
"""
import os
import glob
import sys
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
import pickle as pickle
from skimage.transform import resize
from skimage import filters
from skimage import morphology

import PIL

from scipy.spatial.distance import cdist
import scipy.stats as sistats
from preOrPostprocessing import saturateImage
from options import optionsDCVA
import rasterio
from rasterio.enums import Resampling

from dcvaFunctionResnet18 import dcva

from dataLoaderOSCD import multiCD

opt = optionsDCVA().parseOptions()
thresholdingStrategy = opt.thresholding
otsuScalingFactor = opt.otsuScalingFactor
objectMinSize = opt.objectMinSize

opt.topPercentSaturationOfImageOk = False ## No need to do normalization in the dcva function

resultDirectory = './resultWithModelResnet18/'



PATH_TO_DATASET = "../../../datasets/ONERA_s1_s2/"
CHANNEL_TYPE = 1  # 0-BGR | 1-BGRIr | 2-All bands s.t. resolution <= 20m | 3-All bands | 4-All bands Sentinel-2 & Sentinel-1 | 5-RGB bands Sentinel-2 & Sentinel-1
testDataset = multiCD(PATH_TO_DATASET, split="test",s2_channel_type=CHANNEL_TYPE)

dset = testDataset
for imgIndex in dset.names:
        print(f"Testing for ROI {imgIndex}")
        # inserted multi-modality here
        #S2_1_full, S2_2_full, cm_full = dset.get_img(imgIndex)
        fullImgs = dset.get_img_np_instead_of_tensor(imgIndex)
        S2_1_full, S2_2_full, cm_full = fullImgs['time_1']['S2'], fullImgs['time_2']['S2'], fullImgs['label']
        
        
        preChangeImage = S2_1_full
        preChangeImage = np.transpose(preChangeImage,(1,2,0)) 
        postChangeImage = S2_2_full
        postChangeImage = np.transpose(postChangeImage,(1,2,0))       
        
        
        ## Dataloader for OSCD is loading in B-G-R-NIR formal
        ## S2 Trained Model is in format B4 (R) - B3 (G) - B2 (B) 
        ## As they are not in same format we do need to rearrange the bands
        
        preChangeImage = preChangeImage[:,:,(2,1,0)]
        postChangeImage = postChangeImage[:,:,(2,1,0)]
        
        
        ## Normalizing Sentinel-2 bands with mean and std given derived from EuroSAT-RGB
        preChangeImage = saturateImage().standardizeSentinel2RGBWithEuroSATValues(preChangeImage)
        postChangeImage = saturateImage().standardizeSentinel2RGBWithEuroSATValues(postChangeImage)
        
              
        preChangeImageOriginalShape = preChangeImage.shape
#        if preChangeImageOriginalShape[0]<preChangeImageOriginalShape[1]: ##code is written in a way s.t. it expects row>col
#            preChangeImage = np.swapaxes(preChangeImage,0,1)
#            postChangeImage = np.swapaxes(postChangeImage,0,1)
        
        stepSizeForLargeAreaAnalysis = 1024
        
        for rowIter in range(0,preChangeImage.shape[0],stepSizeForLargeAreaAnalysis):
            for colIter in range(0,preChangeImage.shape[1],stepSizeForLargeAreaAnalysis):
                rowEnd = min((rowIter+stepSizeForLargeAreaAnalysis),preChangeImage.shape[0])
                colEnd = min((colIter+stepSizeForLargeAreaAnalysis),preChangeImage.shape[1])
                preChangeSubImage = preChangeImage[rowIter:rowEnd,colIter:colEnd,:]
                postChangeSubImage = postChangeImage[rowIter:rowEnd,colIter:colEnd,:]
                dcva(preChangeSubImage,postChangeSubImage,opt,resultDirectory+'temp/tempResult_'+str(rowIter)+'_'+str(colIter)+'_'+'.mat')
        
        
        detectedChangeMap = np.zeros((preChangeImage.shape[0],preChangeImage.shape[1])) 
        for rowIter in range(0,preChangeImage.shape[0],stepSizeForLargeAreaAnalysis):
            for colIter in range(0,preChangeImage.shape[1],stepSizeForLargeAreaAnalysis): 
                fileNameToRead = resultDirectory+'temp/tempResult_'+str(rowIter)+'_'+str(colIter)+'_'+'.mat'
                detectedChangeMapThisSubArea = sio.loadmat(fileNameToRead)['detectedChangeMapThisSubArea']
                rowEnd = min((rowIter+stepSizeForLargeAreaAnalysis),preChangeImage.shape[0])
                colEnd = min((colIter+stepSizeForLargeAreaAnalysis),preChangeImage.shape[1])
                detectedChangeMap[rowIter:rowEnd,colIter:colEnd] = detectedChangeMapThisSubArea
        
        

        
        
        
        ##Normalizing the detected Change Map        
        detectedChangeMapNormalized=(detectedChangeMap-np.amin(detectedChangeMap))/(np.amax(detectedChangeMap)-np.amin(detectedChangeMap))
#        
#        #hist, binEdges = np.histogram(detectedChangeMapNormalized)
#        #histNormalized = hist/sum(hist)
#        #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#        #print(histNormalized)
#        #print(binEdges)
#        
#          
        #detectedChangeMapNormalized=filters.gaussian(detectedChangeMapNormalized,3) #this one is with constant sigma
        cdMap=np.zeros(detectedChangeMapNormalized.shape, dtype=bool)
        if thresholdingStrategy == 'adaptive':
            for sigma in range(101,202,50):
                adaptiveThreshold=2*filters.gaussian(detectedChangeMapNormalized,sigma)
                cdMapTemp=(detectedChangeMapNormalized>adaptiveThreshold) 
                cdMapTemp=morphology.remove_small_objects(cdMapTemp,min_size=objectMinSize)
                cdMap=cdMap | cdMapTemp
        elif thresholdingStrategy == 'otsu':
            otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
            cdMap = (detectedChangeMapNormalized>otsuThreshold) 
            cdMap=morphology.remove_small_objects(cdMap,min_size=objectMinSize)
        elif thresholdingStrategy == 'scaledOtsu':
            otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
            cdMap = (detectedChangeMapNormalized>otsuScalingFactor*otsuThreshold) 
            cdMap=morphology.remove_small_objects(cdMap,min_size=objectMinSize)
        else: 
            sys.exit('Unknown thresholding strategy')
        cdMap=morphology.binary_closing(cdMap,morphology.disk(3))
        
#        if preChangeImageOriginalShape[0]<preChangeImageOriginalShape[1]: ##Conformity to row>col
#            cdMap = np.swapaxes(cdMap,0,1)
        
        ##Creating directory to save result
        if not os.path.exists(resultDirectory):
            os.makedirs(resultDirectory)
        
        #Saving the result
        #sio.savemat(resultDirectory+'binaryCdResult.mat', mdict={'cdMap': cdMap})
        plt.imsave(resultDirectory+'binaryCdResult_'+imgIndex+'.png',np.repeat(np.expand_dims((cdMap),2),3,2).astype(float))
        #plt.imsave(resultDirectory+'normalizedChangeMap_'+imgIndex+'.png',np.repeat(np.expand_dims(detectedChangeMapNormalized,2),3,2).astype(float))
  
  
  
  
  
