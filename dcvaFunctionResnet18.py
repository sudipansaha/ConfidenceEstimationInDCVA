# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Sudipan Saha
"""
import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models, transforms
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
from options import optionsDCVA
import rasterio



def dcva(preChangeImage, postChangeImage, opt, saveNormalizedChangeMapPath):
  outputLayerNumbers = np.array(opt.layersToProcess.split(','),dtype=int)
  thresholdingStrategy = opt.thresholding
  otsuScalingFactor = opt.otsuScalingFactor
  objectMinSize = opt.objectMinSize
  topPercentSaturationOfImageOk=opt.topPercentSaturationOfImageOk
  multipleCDBool=opt.multipleCDBool
  changeVectorBinarizationStrategy=opt.changeVectorBinarizationStrategy
  clusteringStrategy=opt.clusteringStrategy
  clusterNumber=opt.clusterNumber
  hierarchicalDistanceStrategy=opt.hierarchicalDistanceStrategy
  
  
  
  nanVar=float('nan')
  
  #Defining parameters related to the CNN
  sizeReductionTable=[nanVar,4,8,16,32] 
  featurePercentileToDiscardTable=[nanVar,90,90,96,98]
  ##filterNumberTable=[nanVar,256,512,1024,2048]
  filterNumberTable=[nanVar,64,128,256,512]  ###For Resnet-18
  
  #When operations like filterNumberForOutputLayer=filterNumberTable[outputLayerNumber] are taken, it works, as 0 is dummy and indexing starts from 1
  
  
      
  
  
  #Pre-change and post-change image normalization
  if topPercentSaturationOfImageOk:
      preChangeImageNormalized=saturateImage().saturateSomePercentileMultispectral(preChangeImage, topPercentToSaturate)
      postChangeImageNormalized=saturateImage().saturateSomePercentileMultispectral(postChangeImage, topPercentToSaturate)
  else:
      preChangeImageNormalized = preChangeImage.copy()
      postChangeImageNormalized = postChangeImage.copy()
      
  
  
  #Reassigning pre-change and post-change image to normalized values
  data1=np.copy(preChangeImageNormalized)
  data2=np.copy(postChangeImageNormalized)
  
  
  #Checking image dimension
  imageSize=data1.shape
  imageSizeRow=imageSize[0]
  imageSizeCol=imageSize[1]
  imageNumberOfChannel=imageSize[2]
  
  
  

  
  save = torch.load("trainedNet/resnet18/rgbEurosat/modelBest.pt", map_location='cpu')
  normalization = save['normalization']
  net = models.resnet18(num_classes=save['model_state']['fc.bias'].numel())
  net.load_state_dict(save['model_state'])
  net = net.to('cuda')

 
  
  netForFeatureExtractionLayer1 = nn.Sequential(*list(net.children())[0:5])
  netForFeatureExtractionLayer2 = nn.Sequential(*list(net.children())[0:6])
  netForFeatureExtractionLayer3 = nn.Sequential(*list(net.children())[0:7])
  netForFeatureExtractionLayer4 = nn.Sequential(*list(net.children())[0:8])
  
 

  
  
  ##changing all nets to eval mode
  netForFeatureExtractionLayer1.eval()
  netForFeatureExtractionLayer1.requires_grad=False
  
  netForFeatureExtractionLayer2.eval()
  netForFeatureExtractionLayer2.requires_grad=False
  
  netForFeatureExtractionLayer3.eval()
  netForFeatureExtractionLayer3.requires_grad=False
  
  netForFeatureExtractionLayer4.eval()
  netForFeatureExtractionLayer4.requires_grad=False
  
  
  torch.no_grad()
  
  
  eachPatch=512
  numImageSplitRow=imageSizeRow/eachPatch
  numImageSplitCol=imageSizeCol/eachPatch
  cutY=list(range(0,imageSizeRow,eachPatch))
  cutX=list(range(0,imageSizeCol,eachPatch))
  additionalPatchPixel=64
  
  
  layerWiseFeatureExtractorFunction=[nanVar,netForFeatureExtractionLayer1,netForFeatureExtractionLayer2,netForFeatureExtractionLayer3,netForFeatureExtractionLayer4]
  
   
  
  ##Checking validity of feature extraction layers
  validFeatureExtractionLayers=[1,2,3,4] ##Feature extraction from only these layers have been defined here
  for outputLayer in outputLayerNumbers:
      if outputLayer not in validFeatureExtractionLayers:
          sys.exit('Feature extraction layer is not valid, valid values are 1,2,3,4')
          
  ##Extracting bi-temporal features
  modelInputMean=0  ##Input maps are already normalized with BigEarthNet mean and std
  
  
  for outputLayerIter in range(0,len(outputLayerNumbers)):
          outputLayerNumber=outputLayerNumbers[outputLayerIter]
          filterNumberForOutputLayer=filterNumberTable[outputLayerNumber]
          featurePercentileToDiscard=featurePercentileToDiscardTable[outputLayerNumber]
          featureNumberToRetain=int(np.floor(filterNumberForOutputLayer*((100-featurePercentileToDiscard)/100)))
          sizeReductionForOutputLayer=sizeReductionTable[outputLayerNumber]
          patchOffsetFactor=int(additionalPatchPixel/sizeReductionForOutputLayer)
          print('Processing layer number:'+str(outputLayerNumber))
          
          timeVector1Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
          timeVector2Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
          
          
          if ((imageSizeRow<eachPatch) | (imageSizeCol<eachPatch)):
              if imageSizeRow>imageSizeCol:
                  patchToProcessDate1=np.pad(data1,[(0,0),(0,imageSizeRow-imageSizeCol),(0,0)],'symmetric')
                  patchToProcessDate2=np.pad(data2,[(0,0),(0,imageSizeRow-imageSizeCol),(0,0)],'symmetric')
              if imageSizeCol>imageSizeRow:
                  patchToProcessDate1=np.pad(data1,[(0,imageSizeCol-imageSizeRow),(0,0),(0,0)],'symmetric')
                  patchToProcessDate2=np.pad(data2,[(0,imageSizeCol-imageSizeRow),(0,0),(0,0)],'symmetric')
              if imageSizeRow==imageSizeCol:
                  patchToProcessDate1=data1
                  patchToProcessDate2=data2
              #print('This image (or this subpatch) is small and hence processing in 1 step')  
               #converting to pytorch varibales and changing dimension for input to net
              patchToProcessDate1=patchToProcessDate1-modelInputMean
              inputToNetDate1=torch.from_numpy(patchToProcessDate1)
              inputToNetDate1=inputToNetDate1.float()
              inputToNetDate1 = torch.permute(inputToNetDate1, (2, 0, 1))
              inputToNetDate1=inputToNetDate1.unsqueeze(0)
              del patchToProcessDate1
                          
              patchToProcessDate2=patchToProcessDate2-modelInputMean
              inputToNetDate2=torch.from_numpy(patchToProcessDate2)
              inputToNetDate2=inputToNetDate2.float()
              inputToNetDate2 = torch.permute(inputToNetDate2, (2, 0, 1))
              inputToNetDate2=inputToNetDate2.unsqueeze(0)
              del patchToProcessDate2
              
              #running model on image 1 and converting features to numpy format
              inputToNetDate1 = inputToNetDate1.cuda()
              with torch.no_grad():
                  obtainedFeatureVals1=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate1)
              obtainedFeatureVals1=obtainedFeatureVals1.squeeze()
              obtainedFeatureVals1=obtainedFeatureVals1.data.cpu().numpy()
              del inputToNetDate1
              
              #running model on image 2 and converting features to numpy format
              inputToNetDate2 = inputToNetDate2.cuda()
              with torch.no_grad():
                  obtainedFeatureVals2=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate2)
              obtainedFeatureVals2=obtainedFeatureVals2.squeeze()
              obtainedFeatureVals2=obtainedFeatureVals2.data.cpu().numpy()
              del inputToNetDate2
              
              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector1Feature[0:imageSizeRow,\
                                             0:imageSizeCol,processingFeatureIter]=\
                                             resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                         0:int(imageSizeRow/sizeReductionForOutputLayer),\
                                                                         0:int(imageSizeCol/sizeReductionForOutputLayer)],\
                                                                         (imageSizeRow,imageSizeCol))
              for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector2Feature[0:imageSizeRow,\
                                             0:imageSizeCol,processingFeatureIter]=\
                                             resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                         0:int(imageSizeRow/sizeReductionForOutputLayer),\
                                                                         0:int(imageSizeCol/sizeReductionForOutputLayer)],\
                                                                         (imageSizeRow,imageSizeCol))
         
              
              
                  
          
          if not((imageSizeRow<eachPatch) | (imageSizeCol<eachPatch)):
              for kY in range(0,len(cutY)):
                  for kX in range(0,len(cutX)):
                      

                      #extracting subset of image 1
                      if (kY==0 and kX==0):
                          patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                         cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                      elif (kY==0 and kX!=(len(cutX)-1)):
                          patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                         (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                      elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                          patchToProcessDate1=data1[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                         (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:] 
                      elif (kX==0 and kY!=(len(cutY)-1)):
                          patchToProcessDate1=data1[(cutY[kY]-additionalPatchPixel):\
                                                    (cutY[kY]+eachPatch),\
                                                         cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                      elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                          patchToProcessDate1=data1[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                                    (imageSizeRow),\
                                                         cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                      elif (kY==(len(cutY)-1) and kX==(len(cutX)-1)):
                          patchToProcessDate1=data1[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                                    (imageSizeRow),\
                                                         (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:]
                      else:
                          patchToProcessDate1=data1[(cutY[kY]-additionalPatchPixel):\
                                                    (cutY[kY]+eachPatch),\
                                                    (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                      #extracting subset of image 2   
                      if (kY==0 and kX==0):
                          patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                         cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                      elif (kY==0 and kX!=(len(cutX)-1)):
                          patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                         (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                      elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                          patchToProcessDate2=data2[cutY[kY]:(cutY[kY]+eachPatch+additionalPatchPixel),\
                                                         (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:] 
                      elif (kX==0 and kY!=(len(cutY)-1)):
                          patchToProcessDate2=data2[(cutY[kY]-additionalPatchPixel):\
                                                    (cutY[kY]+eachPatch),\
                                                        cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                      elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                          patchToProcessDate2=data2[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                                    (imageSizeRow),\
                                                         cutX[kX]:(cutX[kX]+eachPatch+additionalPatchPixel),:]
                      elif (kY==(len(cutY)-1) and kX==(len(cutX)-1)):
                          patchToProcessDate2=data2[(imageSizeRow-eachPatch-additionalPatchPixel):\
                                                    (imageSizeRow),\
                                                         (imageSizeCol-eachPatch-additionalPatchPixel):(imageSizeCol),:]    
                      else:
                          patchToProcessDate2=data2[(cutY[kY]-additionalPatchPixel):\
                                                    (cutY[kY]+eachPatch),\
                                                    (cutX[kX]-additionalPatchPixel):(cutX[kX]+eachPatch),:]
                      print(kY)
                      print(kX)
                      print(patchToProcessDate1.shape)
                      print(patchToProcessDate2.shape)
                                      
                      #converting to pytorch varibales and changing dimension for input to net
                      patchToProcessDate1=patchToProcessDate1-modelInputMean
                      
                      inputToNetDate1 = torch.from_numpy(patchToProcessDate1)
                      del patchToProcessDate1
                      inputToNetDate1  =inputToNetDate1.float()
                      inputToNetDate1 = torch.permute(inputToNetDate1, (2, 0, 1))
                      inputToNetDate1 = inputToNetDate1.unsqueeze(0)
                      
                      
                      patchToProcessDate2=patchToProcessDate2-modelInputMean
                      
                      inputToNetDate2=torch.from_numpy(patchToProcessDate2)
                      del patchToProcessDate2
                      inputToNetDate2 = inputToNetDate2.float()
                      inputToNetDate2 = torch.permute(inputToNetDate2, (2, 0, 1))
                      inputToNetDate2 = inputToNetDate2.unsqueeze(0)
                      
                      
                      #running model on image 1 and converting features to numpy format
                      inputToNetDate1 = inputToNetDate1.cuda()
                      with torch.no_grad():
                          obtainedFeatureVals1=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate1)
                      obtainedFeatureVals1=obtainedFeatureVals1.squeeze()
                      obtainedFeatureVals1=obtainedFeatureVals1.cpu().numpy()
                      del inputToNetDate1
                      
                      #running model on image 2 and converting features to numpy format
                      inputToNetDate2 = inputToNetDate2.cuda()
                      with torch.no_grad():
                          obtainedFeatureVals2=layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate2)
                      obtainedFeatureVals2=obtainedFeatureVals2.squeeze()
                      obtainedFeatureVals2=obtainedFeatureVals2.cpu().numpy()
                      del inputToNetDate2
                      #this features are in format (filterNumber, sizeRow, sizeCol)
                      
                      
                      
                      ##clipping values to +1 to -1 range, be careful, if network is changed, maybe we need to modify this
                      obtainedFeatureVals1=np.clip(obtainedFeatureVals1,-1,+1)
                      obtainedFeatureVals2=np.clip(obtainedFeatureVals2,-1,+1)
                      
                                      
                      #obtaining features from image 1: resizing and truncating additionalPatchPixel
                      if (kY==0 and kX==0):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                         (eachPatch,eachPatch))
                          
                      elif (kY==0 and kX!=(len(cutX)-1)):                                                    
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                         (patchOffsetFactor+1):\
                                                                         (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                         (eachPatch,eachPatch))
                      elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):                        
                              timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:imageSizeCol,processingFeatureIter]=\
                                             resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                         (obtainedFeatureVals1.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                         (obtainedFeatureVals1.shape[2])],\
                                                                         (eachPatch,(imageSizeCol-cutX[kX])))
                      elif (kX==0 and kY!=(len(cutY)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                         (patchOffsetFactor+1):\
                                                                         (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                         (eachPatch,eachPatch))
                      elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector1Feature[cutY[kY]:imageSizeRow,\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                         (obtainedFeatureVals1.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                         (obtainedFeatureVals1.shape[1]),\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                         ((imageSizeRow-cutY[kY]),eachPatch))
                      elif (kX==(len(cutX)-1) and kY==(len(cutY)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                         (obtainedFeatureVals1.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                         (obtainedFeatureVals1.shape[1]),\
                                                                         (obtainedFeatureVals1.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                         (obtainedFeatureVals1.shape[2])],\
                                                                         ((imageSizeRow-cutY[kY]),(imageSizeCol-cutX[kX])))
                      else:
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector1Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals1[processingFeatureIter,\
                                                                         (patchOffsetFactor+1):\
                                                                         (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                         (patchOffsetFactor+1):\
                                                                         (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                         (eachPatch,eachPatch))
                      #obtaining features from image 2: resizing and truncating additionalPatchPixel
                      if (kY==0 and kX==0):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                         (eachPatch,eachPatch))
                          
                      elif (kY==0 and kX!=(len(cutX)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                         (patchOffsetFactor+1):\
                                                                         (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                         (eachPatch,eachPatch))
                      elif (kY!=(len(cutY)-1) and kX==(len(cutX)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:imageSizeCol,processingFeatureIter]=\
                                             resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer),\
                                                                         (obtainedFeatureVals2.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                         (obtainedFeatureVals2.shape[2])],\
                                                                         (eachPatch,(imageSizeCol-cutX[kX])))
                      elif (kX==0 and kY!=(len(cutY)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                         (patchOffsetFactor+1):\
                                                                         (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                         (eachPatch,eachPatch))
                      elif (kX!=(len(cutX)-1) and kY==(len(cutY)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector2Feature[cutY[kY]:imageSizeRow,\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                         (obtainedFeatureVals2.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                         (obtainedFeatureVals2.shape[1]),\
                                                                         0:int(eachPatch/sizeReductionForOutputLayer)],\
                                                                         ((imageSizeRow-cutY[kY]),eachPatch))
                      elif (kX==(len(cutX)-1) and kY==(len(cutY)-1)):
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                         (obtainedFeatureVals2.shape[1]-1-int((imageSizeRow-cutY[kY])/sizeReductionForOutputLayer)):\
                                                                         (obtainedFeatureVals2.shape[1]),\
                                                                         (obtainedFeatureVals2.shape[2]-1-int((imageSizeCol-cutX[kX])/sizeReductionForOutputLayer)):\
                                                                         (obtainedFeatureVals2.shape[2])],\
                                                                         ((imageSizeRow-cutY[kY]),(imageSizeCol-cutX[kX])))
                      else:
                          for processingFeatureIter in range(0,filterNumberForOutputLayer):
                              timeVector2Feature[cutY[kY]:(cutY[kY]+eachPatch),\
                                             cutX[kX]:(cutX[kX]+eachPatch),processingFeatureIter]=\
                                             resize(obtainedFeatureVals2[processingFeatureIter,\
                                                                         (patchOffsetFactor+1):\
                                                                         (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1),\
                                                                         (patchOffsetFactor+1):\
                                                                         (int(eachPatch/sizeReductionForOutputLayer)+patchOffsetFactor+1)],\
                                                                         (eachPatch,eachPatch))
                                         
          del  obtainedFeatureVals1,obtainedFeatureVals2                             
          timeVectorDifferenceMatrix=timeVector1Feature-timeVector2Feature
          print(timeVectorDifferenceMatrix.shape)
          
          nonZeroVector=[]
          stepSizeForStdCalculation=min(int(imageSizeRow/2),1024)
          for featureSelectionIter1 in range(0,imageSizeRow,stepSizeForStdCalculation):
              for featureSelectionIter2 in range(0,imageSizeCol,stepSizeForStdCalculation):
                  timeVectorDifferenceSelectedRegion=timeVectorDifferenceMatrix\
                                                     [featureSelectionIter1:(featureSelectionIter1+stepSizeForStdCalculation),\
                                                      featureSelectionIter2:(featureSelectionIter2+stepSizeForStdCalculation),
                                                      0:filterNumberForOutputLayer]
                  stdVectorDifferenceSelectedRegion=np.std(timeVectorDifferenceSelectedRegion,axis=(0,1))
                  featuresOrderedPerStd=np.argsort(-stdVectorDifferenceSelectedRegion)   #negated array to get argsort result in descending order
                  nonZeroVectorSelectedRegion=featuresOrderedPerStd[0:featureNumberToRetain]
                  nonZeroVector=np.union1d(nonZeroVector,nonZeroVectorSelectedRegion)
              
         
                 
          modifiedTimeVector1=timeVector1Feature[:,:,nonZeroVector.astype(int)]
          modifiedTimeVector2=timeVector2Feature[:,:,nonZeroVector.astype(int)]
          del timeVector1Feature,timeVector2Feature
          
          ##Normalize the features (separate for both images)
          meanVectorsTime1Image=np.mean(modifiedTimeVector1,axis=(0,1))      
          stdVectorsTime1Image=np.std(modifiedTimeVector1,axis=(0,1))
          normalizedModifiedTimeVector1=(modifiedTimeVector1-meanVectorsTime1Image)/stdVectorsTime1Image
          
          meanVectorsTime2Image=np.mean(modifiedTimeVector2,axis=(0,1))      
          stdVectorsTime2Image=np.std(modifiedTimeVector2,axis=(0,1))
          normalizedModifiedTimeVector2=(modifiedTimeVector2-meanVectorsTime2Image)/stdVectorsTime2Image
          
          ##feature aggregation across channels
          if outputLayerIter==0:
              timeVector1FeatureAggregated=np.copy(normalizedModifiedTimeVector1)
              timeVector2FeatureAggregated=np.copy(normalizedModifiedTimeVector2)
          else:
              timeVector1FeatureAggregated=np.concatenate((timeVector1FeatureAggregated,normalizedModifiedTimeVector1),axis=2)
              timeVector2FeatureAggregated=np.concatenate((timeVector2FeatureAggregated,normalizedModifiedTimeVector2),axis=2)
      
   
      
      
  del netForFeatureExtractionLayer1, netForFeatureExtractionLayer2, netForFeatureExtractionLayer3, netForFeatureExtractionLayer4  
      
  absoluteModifiedTimeVectorDifference=np.absolute(saturateImage().saturateSomePercentileMultispectral(timeVector1FeatureAggregated,1)-\
  saturateImage().saturateSomePercentileMultispectral(timeVector2FeatureAggregated,1)) 
  
  
  #take absolute value for binary CD
  detectedChangeMap=np.linalg.norm(absoluteModifiedTimeVectorDifference,axis=(2))

  
  sio.savemat(saveNormalizedChangeMapPath,{"detectedChangeMapThisSubArea": detectedChangeMap})
  
  
  
  
