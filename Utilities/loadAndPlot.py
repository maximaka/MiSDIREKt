# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 20:03:12 2022

@author: maxim
"""

import numpy as np
import scipy.io as sio
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import preProcessPickle as proc, architectureCreation as arch, visualizeAniPlot as visu
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
    
#%%
if __name__ == '__main__':
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    windowLength = 75
    strideLength = 1
    batchSize = 128
    split=0.8
    needsShuffle = True
    needsNPArray = True
    timeSeriesParam = [windowLength,strideLength,batchSize,needsShuffle,needsNPArray]
    datasets = "hand_emg"
    dataFormat = "equal"
    
    #EMG
    emgDimensions = 8
    handDimensions = 27
    latentEMGDimensions = 4
    latentHandDimensions = 4
    
    modelLoadNum = 24
    modelPath = "E:/NonBackupThings/MaximResearch/SauerKraut/Models/Injector/"
    customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
        
    encoderHand = keras.models.load_model(modelPath+"HandEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
    encoderHand.compile(optimizer='Adam',loss='mse')
    decoderHand = keras.models.load_model(modelPath+"HandDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
    decoderHand.compile(optimizer='Adam',loss='mse')
    encoderEMG = keras.models.load_model(modelPath+"EMGEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
    encoderEMG.compile(optimizer='Adam',loss='mse')
    decoderEMG = keras.models.load_model(modelPath+"EMGDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
    decoderEMG.compile(optimizer='Adam',loss='mse')
    print("loaded and compiled old models")

    emgAutoencoder = arch.createAutoencoderFromComponentsNoDenseParts(windowLength, emgDimensions, latentEMGDimensions, latentEMGDimensions, encoderEMG, decoderEMG)
    emgAutoencoder.compile(optimizer='Adam',loss='mse')
    handAutoencoder = arch.createAutoencoderFromComponentsNoDenseParts(windowLength, handDimensions, latentHandDimensions, latentHandDimensions, encoderHand, decoderHand)
    handAutoencoder.compile(optimizer='Adam',loss='mse')
    #%%
    strideLength = windowLength
    needsShuffle = False
    timeSeriesParam = [windowLength,strideLength,batchSize,needsShuffle,needsNPArray]
    dataFormat = "equal"
    
    startSession = 87
    endSession = 88
    sessionsToInclude = range(startSession,endSession)
    
    _, _, emgFilteredNS, emgDimensions, emgColumns, _, _, handFilteredNS, handDimensions, handColumns = proc.preprocessBothToDataset(sessionsToInclude,timeSeriesParam,datasets,dataFormat)
    print("Loaded Test Data")
    
    emgInputDataTest, _ = proc.splitDataset(emgFilteredNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    handInputDataTest, _ = proc.splitDataset(handFilteredNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    print('Split Test Data')

    testerPredsNSEMG= emgAutoencoder.predict([emgInputDataTest])
    print("Created Predictor and Predicted")
    
    visu.createTruthPredPlots(emgInputDataTest, testerPredsNSEMG, _ , emgDimensions, emgColumns, "hand")
            
    embedsEMGTest = encoderEMG.predict(emgInputDataTest)
    embedsHandTest = encoderHand.predict(handInputDataTest)
    visu.createEmbedPlot(embedsEMGTest," original emg embed, test")
    visu.createEmbedPlot(embedsHandTest," original hand embed, test")

    # visu.createAnimation(embedsHandTest[:,:,0:2], "hand")
    print("Calculated Plots and Embeds")
            