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
import cv2 as cv
import PIL
import cv2
from scipy.spatial.distance import cdist
import scipy.stats as sistats
from preOrPostprocessing import saturateImage
from options import optionsDCVAAndConfidenceEvaluation
import rasterio
from rasterio.enums import Resampling

from dcvaFunctionResnet18 import dcva

from dataLoaderOSCD import multiCD

opt = optionsDCVAAndConfidenceEvaluation().parseOptions()

otsuScalingFactor = opt.otsuScalingFactor
smoothingIterNum = opt.smoothingIterNum
smoothingNoiseStd = opt.smoothingNoiseStd

opt.topPercentSaturationOfImageOk = False ## No need to do normalization in the dcva function

resultDirectory = './results/'



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
#                preChangeImage = np.swapaxes(preChangeImage,0,1)
#                postChangeImage = np.swapaxes(postChangeImage,0,1)

        
              
        
        cdMapSmoothedSumOverIterations = np.zeros((preChangeImage.shape[0],preChangeImage.shape[1]), dtype=int)
        
        preChangeImageBeforeNoise = np.copy(preChangeImage)
        postChangeImageBeforeNoise = np.copy(postChangeImage)
        
        for smoothingIter in range(smoothingIterNum):
        
            ## Creating smoothed version of the input images
            
#            for channelIter in range(preChangeImage.shape[2]):
#                gaussianPreChangeBand = np.random.normal(0,0.1,(preChangeImage.shape[0],preChangeImage.shape[1]))
#                gaussianPostChangeBand = np.random.normal(0,0.1,(preChangeImage.shape[0],preChangeImage.shape[1]))
#                preChangeImage[:,:,channelIter] = preChangeImage[:,:,channelIter]+gaussianPreChangeBand
#                postChangeImage[:,:,channelIter] = postChangeImage[:,:,channelIter]+gaussianPostChangeBand
                
            
            preChangeImage=torch.from_numpy(preChangeImageBeforeNoise)   
            postChangeImage=torch.from_numpy(postChangeImageBeforeNoise)
            preChangeImage = preChangeImage + torch.randn_like(preChangeImage) * smoothingNoiseStd
            postChangeImage = postChangeImage + torch.randn_like(postChangeImage) * smoothingNoiseStd
            preChangeImage = preChangeImage.numpy()
            postChangeImage = postChangeImage.numpy()           

        
            
            
        
            
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
            
            ## Fixed threshold strategy 
            otsuThreshold=filters.threshold_otsu(detectedChangeMapNormalized)
            cdMapThisIter = (detectedChangeMapNormalized>otsuScalingFactor*otsuThreshold)
            
            
            cdMapSmoothedSumOverIterations = cdMapSmoothedSumOverIterations+cdMapThisIter.astype(int)
            
        
        

#        if preChangeImageOriginalShape[0]<preChangeImageOriginalShape[1]: ##Conformity to row>col
#            detectedChangeMapNormalized = np.swapaxes(detectedChangeMapNormalized,0,1)



        
        ## Saving to .npy file
        np.save(resultDirectory+'smoothingIter_'+str(smoothingIterNum)+'/noise_'+str(smoothingNoiseStd)+'/smoothedCdMaps/'+imgIndex+'.npy', cdMapSmoothedSumOverIterations)
        
        



#      
  
  
  
  
  
