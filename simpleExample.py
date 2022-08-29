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
from Utilities import dataLoading as proc
from Utilities import architectureCreation as arch
from Utilities import visualizeAniPlot as visu
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
    
if __name__ == '__main__':
    
    #Parameters
    projectDirectory = "E:/NonBackupThings/MaximResearch/Misdirekt/"  

    # lossesH = {'decoder':'mean_squared_error','decoder_1':"categorical_crossentropy"}
    # lossWeightsH = [1.0,1.0]
    # metricsH = {'decoder':'mse','decoder_1':'categorical_accuracy'}
    
    losses = ["categorical_crossentropy"]
    lossWeights = [1.0]
    metrics = ['categorical_accuracy']

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4) 
    callbackBig = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 

    windowLengthH = 1
    windowLengthE = 24
    # windowLengthF = 1
    windowLengthF = windowLengthE
    strideLength = 1
    batchSize = 128
    split=0.8
    needsShuffle = True
    needsNPArray = True
    needsLabels = True
    timeSeriesParamH = [windowLengthH,strideLength,batchSize,needsShuffle,needsNPArray]
    timeSeriesParamE = [windowLengthE,strideLength,batchSize,needsShuffle,needsNPArray]
    
    #EMG
    emgDimensions = 8
    handDimensions = 27

    latentEMGDimensions = 4
    latentHandDimensions = 4
    
    # classDimensions = 12
    classDimensions = len(list(proc.interactionLabels.keys()))
    
    print("Done Parameter Definitions")
    totallyNew = True
    
    if totallyNew:
        #only classifier
        encoderEMG = arch.createEncoderLSTM(latentEMGDimensions, windowLengthF, emgDimensions)
        decoderEMG = arch.createLabelPredictor(latentEMGDimensions, classDimensions)
    
        autoencoderEMG = arch.createSequentialTrainTune([encoderEMG,decoderEMG], [True, True])
        autoencoderEMG.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)

        # autoencoderEMG = arch.createClassifierFromComponentsNoDenseParts(windowLengthE, emgDimensions, latentEMGDimensions, latentEMGDimensions, encoderEMG, decoderEMG)
        # autoencoderEMG.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)
    
    else:
        modelLoadNum = 36
        modelPath = "E:/NonBackupThings/MaximResearch/Misdirekt/Code/Models/Classifier/"
        customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
        
        encoderEMG = keras.models.load_model(modelPath+"EMGEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        decoderEMG = keras.models.load_model(modelPath+"EMGDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)

        
    loadModels = False
    # loadModels = True
    activityType = "interactionGesture0"

#%%
    if loadModels == False:
        corruptmaybe = [10,11,35]
        irange = np.arange(0,38,1)

        for i in irange:
            if i not in corruptmaybe: 
                startSession = i
                _, _, emgDimensions, emgColumns, emgFiltered, emgLabels, emgFilteredVal, emgLabelsVal = proc.preprocessSessionToDataset(startSession,activityType,"emg",needsLabels,timeSeriesParamE)
    
                oneHotEncodedELabels = np_utils.to_categorical(emgLabels,num_classes=classDimensions)
                oneHotEncodedELabels = np.array(oneHotEncodedELabels,dtype=float)
                oneHotEncodedELabelsVal = np_utils.to_categorical(emgLabelsVal,num_classes=classDimensions)
                oneHotEncodedELabelsVal = np.array(oneHotEncodedELabelsVal,dtype=float)
                
                print("Loaded and Processed Data")
                
                emgInputDataTrain, emgInputDataVal, labelsETrain, labelsEVal = emgFiltered, emgFilteredVal, oneHotEncodedELabels, oneHotEncodedELabelsVal
                
                print('Split Data')
                        
            # EMG
            if(emgInputDataTrain.shape[0] != 0):

                #Fit the autoencoder
                history1 = autoencoderEMG.fit(x = [emgInputDataTrain], 
                                                # y=[emgInputDataTrain,labelsETrain], 
                                                y=[labelsETrain], 
                                                # validation_data=([emgInputDataVal],[emgInputDataVal,labelsEVal]),
                                                validation_data=([emgInputDataVal],[labelsEVal]),
                                                epochs=65, batch_size=batchSize, callbacks=[callback],shuffle = True)                 
                
            print("DONE " + str(i))
            
        #%%
        modelSaveNum = 48

        encoderEMG.save("Models/Collective/EMGEncoder"+str(modelSaveNum)+".h5")
        decoderEMG.save("Models/Collective/EMGDecoder"+str(modelSaveNum)+".h5")
    else:
        modelLoadNum = 48
        modelPath = "E:/NonBackupThings/MaximResearch/Misdirekt/Code/Models/Collective/"
        customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
        
        encoderEMG = keras.models.load_model(modelPath+"EMGEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        decoderEMG = keras.models.load_model(modelPath+"EMGDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)

        autoencoderEMG = arch.createClassifierFromComponentsNoDenseParts(windowLengthE, emgDimensions, latentEMGDimensions, latentEMGDimensions, encoderEMG, decoderEMG)
        autoencoderEMG.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)

    
#%%
    modelLoadNum = 48
    labelList = list(proc.interactionLabels.keys())
    classLabels = list(proc.interactionLabels.values())
    modelPath = "E:/NonBackupThings/MaximResearch/Misdirekt/Code/Models/Collective/"
    customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 
    
    epochNum = 19
    
    for startSession in range(36,42):
        _, _, _, _, emgFilteredValNS, emgLabelsValNS, emgFilteredTrainNS, emgLabelsTrainNS, emgFilteredTestNS, emgLabelsTestNS = proc.preprocessSessionToDataset(startSession,activityType,"emg",needsLabels,timeSeriesParamE,True)
        # emgFilteredValNS,emgLabelsValNS = augmentEMG(emgFilteredValNS, emgLabelsValNS)
        # emgFilteredTrainNS,emgLabelsTrainNS = augmentEMG(emgFilteredTrainNS, emgLabelsTrainNS)
        # emgFilteredTestNS,emgLabelsTestNS = augmentEMG(emgFilteredTestNS, emgLabelsTestNS)
        # emgFilteredValNS,emgLabelsValNS = addNoise(emgFilteredValNS, emgLabelsValNS)
        # emgFilteredTrainNS,emgLabelsTrainNS = addNoise(emgFilteredTrainNS, emgLabelsTrainNS)
        # emgFilteredTestNS,emgLabelsTestNS = addNoise(emgFilteredTestNS, emgLabelsTestNS)

        oneHotEncodedELabelsTrainNS = np_utils.to_categorical(emgLabelsTrainNS,num_classes=classDimensions)
        oneHotEncodedELabelsTrainNS = np.array(oneHotEncodedELabelsTrainNS,dtype=float)
        oneHotEncodedELabelsValNS = np_utils.to_categorical(emgLabelsValNS,num_classes=classDimensions)
        oneHotEncodedELabelsValNS = np.array(oneHotEncodedELabelsValNS,dtype=float)
        oneHotEncodedELabelsTestNS = np_utils.to_categorical(emgLabelsTestNS,num_classes=classDimensions)
        oneHotEncodedELabelsTestNS = np.array(oneHotEncodedELabelsTestNS,dtype=float)
        
        emgInputDataTrainFakeJiveNS, _ = proc.splitDataset(emgFilteredTrainNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
        emgInputDataValNS, _ = proc.splitDataset(emgFilteredValNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
        emgInputDataTestNS, _ = proc.splitDataset(emgFilteredTestNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
        labelsETrainFakeJiveNS, _ = proc.splitDataset(oneHotEncodedELabelsTrainNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
        labelsEValNS, _ = proc.splitDataset(oneHotEncodedELabelsValNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
        labelsETestNS, _ = proc.splitDataset(oneHotEncodedELabelsTestNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
            
        encoderEMGTest = keras.models.load_model(modelPath+"EMGEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        decoderEMGTest = keras.models.load_model(modelPath+"EMGDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
            
        embedsEMGTest = encoderEMGTest.predict(emgInputDataValNS)
        predClassesE = decoderEMGTest.predict(embedsEMGTest)
        predClassesEAbs = np.argmax(predClassesE, axis = 1)

        percentages = visu.createPercentages(predClassesEAbs)
        visu.plotConfusionMatrix(emgLabelsValNS,predClassesEAbs,labelList)
        visu.plotTruePredClasses(emgLabelsValNS, predClassesEAbs)
        visu.plotEmbeddingsLatent4Color(embedsEMGTest,emgLabelsValNS,classLabels, "emg?")
        visu.plotClassesInOrder(predClassesEAbs, emgLabelsValNS, labelList)

  
    savePath = "E:/NonBackupThings/MaximResearch/Misdirekt/Paper/Data/"

