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

    lossesH = {'decoder':'mean_squared_error','decoder_1':"categorical_crossentropy"}
    lossWeightsH = [1.0,1.0]
    metricsH = {'decoder':'mse','decoder_1':'categorical_accuracy'}
    
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
    # emgDimensions = 1
    handDimensions = 27
    # handDimensions = 1
  
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
    
        autoencoderEMG = arch.createClassifierFromComponentsNoDenseParts(windowLengthE, emgDimensions, latentEMGDimensions, latentEMGDimensions, encoderEMG, decoderEMG)
        autoencoderEMG.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)
    
    else:
        modelLoadNum = 46
        modelPath = "E:/NonBackupThings/MaximResearch/Misdirekt/Code/Models/Classifier/"
        customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
        
        encoderEMG = keras.models.load_model(modelPath+"EMGEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        decoderEMG = keras.models.load_model(modelPath+"EMGDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)

        
    # loadModels = False
    loadModels = True
    activityType = "interactionGesture0"

#%%
    if loadModels == False:
        corrupt = [10,11,35]
        steps = 1
        ####
        irange = np.arange(0,15,steps)
        # irange = [0,10]
        trainingAccuracies = []
        validationAccuracies = []
        trainingAccuraciesP = []
        validationAccuraciesP = []
        trainingLoss = []
        validationLoss = []
        trainingLossP = []
        validationLossP = []
        
        trainingAccHist = []
        validationAccHist = []
        trainingAccHistP = []
        validationAccHistP = []
        
        trainingLossHist = []
        validationLossHist = []
        trainingLossHistP = []
        validationLossHistP = []

        #%%
        for i in irange:
            emgInputDataTrainC = np.zeros((0,windowLengthF,emgDimensions))
            emgInputDataValC = np.zeros((0,windowLengthF,emgDimensions))
            labelsETrainC = np.zeros((0,classDimensions))
            labelsEValC = np.zeros((0,classDimensions))
            
            for j in range(0,steps):
                if i+j not in corrupt: 
                    startSession = i+j
                    endSession = j+1
                    sessionsToInclude = range(startSession,endSession)
                    _, _, emgDimensions, emgColumns, emgFiltered, emgLabels, emgFilteredVal, emgLabelsVal = proc.preprocessSessionToDataset(startSession,activityType,"emg",needsLabels,timeSeriesParamE)

                    oneHotEncodedELabels = np_utils.to_categorical(emgLabels,num_classes=classDimensions)
                    oneHotEncodedELabels = np.array(oneHotEncodedELabels,dtype=float)
                    oneHotEncodedELabelsVal = np_utils.to_categorical(emgLabelsVal,num_classes=classDimensions)
                    oneHotEncodedELabelsVal = np.array(oneHotEncodedELabelsVal,dtype=float)
                    
                    print("Loaded and Processed Data")
                    
                    emgInputDataTrain, emgInputDataVal, labelsETrain, labelsEVal = emgFiltered, emgFilteredVal, oneHotEncodedELabels, oneHotEncodedELabelsVal
                    print('Split Data')
                    
                    
                    emgInputDataTrainC = np.concatenate([emgInputDataTrainC,emgInputDataTrain])
                    emgInputDataValC = np.concatenate([emgInputDataValC,emgInputDataVal])
                    labelsETrainC = np.concatenate([labelsETrainC,labelsETrain])
                    labelsEValC = np.concatenate([labelsEValC,labelsEVal])
                        
            # EMG
            if(emgInputDataTrainC.shape[0] != 0):
                
                sandwichPreLayer = arch.createSessionSpecializer(emgDimensions,windowLengthE, latentEMGDimensions)
                autoencoderEMG = arch.createSequentialTrainTune([sandwichPreLayer, encoderEMG,decoderEMG], [True, False, False])
                autoencoderEMG.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)

                #Fit the autoencoder
                history1 = autoencoderEMG.fit(x = [emgInputDataTrainC], 
                                                # y=[emgInputDataTrain,labelsETrain], 
                                                y=[labelsETrainC], 
                                                # validation_data=([emgInputDataVal],[emgInputDataVal,labelsEVal]),
                                                validation_data=([emgInputDataValC],[labelsEVal]),
                                                epochs=65, batch_size=batchSize, callbacks=[callback],shuffle = True)                 
                
                autoencoderEMGRetrain = arch.createSequentialTrainTune([sandwichPreLayer, encoderEMG,decoderEMG], [False, True, True])
                autoencoderEMGRetrain.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)

                #Fit the autoencoder
                history1 = autoencoderEMGRetrain.fit(x = [emgInputDataTrainC], 
                                                # y=[emgInputDataTrain,labelsETrain], 
                                                y=[labelsETrainC], 
                                                # validation_data=([emgInputDataVal],[emgInputDataVal,labelsEVal]),
                                                validation_data=([emgInputDataValC],[labelsEVal]),
                                                epochs=65, batch_size=batchSize, callbacks=[callbackBig],shuffle = True)                 

              

            print("DONE " + str(i))
            
        #%%
        modelSaveNum = 37

        #autoencoderEMG.save("Models/Injector/EMGFullAutoEncoder"+str(modelSaveNum)+".h5")
        encoderEMG.save("Models/Classifier/EMGEncoder"+str(modelSaveNum)+".h5")
        decoderEMG.save("Models/Classifier/EMGDecoder"+str(modelSaveNum)+".h5")
    else:
        modelLoadNum = 36
        modelPath = "E:/NonBackupThings/MaximResearch/Misdirekt/Code/Models/Classifier/"
        customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
        
        encoderEMG = keras.models.load_model(modelPath+"EMGEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        decoderEMG = keras.models.load_model(modelPath+"EMGDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        sandwichPreLayer = keras.models.load_model(modelPath+"sandwichPreLayer"+str(47)+".h5",custom_objects=customObjectList)

        autoencoderEMG = arch.createClassifierFromComponentsNoDenseParts(windowLengthE, emgDimensions, latentEMGDimensions, latentEMGDimensions, encoderEMG, decoderEMG)
        
        autoencoderEMG.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)
       
    # savePath = "E:/NonBackupThings/MaximResearch/Misdirekt/Paper/Data/"



    print(autoencoderEMG.summary(expand_nested=True))
    #%%
    strideLength = 1
    # strideLength = windowLength
    needsShuffle = False
    timeSeriesParamH = [windowLengthH,strideLength,batchSize,needsShuffle,needsNPArray]
    timeSeriesParamE = [windowLengthE,strideLength,batchSize,needsShuffle,needsNPArray]
    labelList = list(proc.interactionLabels.keys())
    classLabels = list(proc.interactionLabels.values())

    #80 and 82
    startSession = 34
    endSession = 34
    sessionsToInclude = range(startSession,endSession)
    # needsLabels = False
    
    _, _, _, _, emgFilteredValNS, emgLabelsValNS, emgFilteredTrainNS, emgLabelsTrainNS, emgFilteredTestNS, emgLabelsTestNS = proc.preprocessSessionToDataset(startSession,activityType,"emg",needsLabels,timeSeriesParamE,True)
    # emgFilteredValNS,emgLabelsValNS = addNoise(emgFilteredValNS, emgLabelsValNS)
    # emgFilteredTrainNS,emgLabelsTrainNS = addNoise(emgFilteredTrainNS, emgLabelsTrainNS)
    # emgFilteredTestNS,emgLabelsTestNS = addNoise(emgFilteredTestNS, emgLabelsTestNS)

    oneHotEncodedELabelsTrainNS = np_utils.to_categorical(emgLabelsTrainNS,num_classes=classDimensions)
    oneHotEncodedELabelsTrainNS = np.array(oneHotEncodedELabelsTrainNS,dtype=float)
    oneHotEncodedELabelsValNS = np_utils.to_categorical(emgLabelsValNS,num_classes=classDimensions)
    oneHotEncodedELabelsValNS = np.array(oneHotEncodedELabelsValNS,dtype=float)
    oneHotEncodedELabelsTestNS = np_utils.to_categorical(emgLabelsTestNS,num_classes=classDimensions)
    oneHotEncodedELabelsTestNS = np.array(oneHotEncodedELabelsTestNS,dtype=float)
    print("Loaded Test Data")
    
    emgInputDataTrainFakeJiveNS, _ = proc.splitDataset(emgFilteredTrainNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    emgInputDataValNS, _ = proc.splitDataset(emgFilteredValNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    emgInputDataTestNS, _ = proc.splitDataset(emgFilteredTestNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    labelsETrainFakeJiveNS, _ = proc.splitDataset(oneHotEncodedELabelsTrainNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    labelsEValNS, _ = proc.splitDataset(oneHotEncodedELabelsValNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    labelsETestNS, _ = proc.splitDataset(oneHotEncodedELabelsTestNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    print('Split Test Data')           

    print("Created Predictor and Predicted")
    
    sandwichEMGTest = sandwichPreLayer.predict(emgInputDataValNS)
    embedsEMGTest = encoderEMG.predict(sandwichEMGTest)
    predClassesE = decoderEMG.predict(embedsEMGTest)
    predClassesEAbs = np.argmax(predClassesE, axis = 1)
    print("Created Predictor and Predicted")

    percentages = visu.createPercentages(predClassesEAbs)
    visu.plotEmbeddingsLatent4Color(embedsEMGTest,emgLabelsValNS,classLabels, "emg?")
    visu.printF1Scoring(predClassesEAbs, emgLabelsValNS, labelList)
    
    accuracyE = np.mean(predClassesEAbs == emgLabelsValNS)
    
    modelPath = "E:/NonBackupThings/MaximResearch/Misdirekt/Code/Models/Classifier/"
    customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 
    
    acc = []
    fones = []
    acc.append(accuracyE)
    epochNum = 19
    
    totalGuessesForMajority = []
    retrainPredsForConMat = []

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

        for iteration in range(0,5):
            
            # encoderHand = keras.models.load_model(modelPath+"HandEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
            # decoderHand = keras.models.load_model(modelPath+"HandDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
            encoderEMGTest = keras.models.load_model(modelPath+"EMGEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
            decoderEMGTest = keras.models.load_model(modelPath+"EMGDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
            #sandwich from session 34
            sandwichPreLayerOriginal = keras.models.load_model(modelPath+"sandwichPreLayer"+str(47)+".h5",custom_objects=customObjectList)
            
            # if iteration < 5 and startSession == 39:
            #     embedsEMGTest0 = sandwichPreLayerOriginal.predict(emgInputDataValNS)
            #     embedsEMGTest = encoderEMGTest.predict(embedsEMGTest0)
            #     visu.plotEmbeddingsLatent4Color(embedsEMGTest,emgLabelsValNS,classLabels, str(startSession) + "generalNetworkEmbeds")
                        
            sandwichPreLayerTest = arch.createSessionSpecializer(emgDimensions,windowLengthE, latentEMGDimensions)

            autoencoderEMGTest = arch.createSequentialTrainTune([sandwichPreLayerTest, encoderEMGTest,decoderEMGTest], [True, False, False])
            autoencoderEMGTest.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)
    
    
            history1 = autoencoderEMGTest.fit(x = [emgInputDataTrainFakeJiveNS], 
                                                y=[labelsETrainFakeJiveNS], 
                                                # validation_data=([emgInputDataVal],[emgInputDataVal,labelsEVal]),
                                                validation_data=([emgInputDataTestNS],[labelsETestNS]),
                                                epochs=75, batch_size=batchSize,shuffle = True,callbacks=[callback])                 
            
            
            initialEmbeds = sandwichPreLayerTest.predict(emgInputDataValNS)
            embedsEMGTest = encoderEMGTest.predict(initialEmbeds)
            predClassesE = decoderEMGTest.predict(embedsEMGTest)
            predClassesEAbs = np.argmax(predClassesE, axis = 1)
    
            accuracyE = np.mean(predClassesEAbs == emgLabelsValNS)
            print(str(startSession) + " accuracy is " + str(accuracyE))
            retrainPredsForConMat.append([emgLabelsValNS,predClassesEAbs,embedsEMGTest])

            # acc.append(accuracyE)
            
            # visu.plotClassesInOrder(predClassesEAbs, emgLabelsValNS, labelList)
            # a,b = visu.printF1Scoring(predClassesEAbs, emgLabelsValNS, labelList)
            # fones.append(a)
            # totalGuessesForMajority.append(predClassesEAbs)
            
            # if iteration < 5 and startSession == 39:
            #     visu.plotEmbeddingsLatent4Color(embedsEMGTest,emgLabelsValNS,classLabels, str(startSession) + "sandwichEmbeds")

            
            # initialEmbeds = sandwichPreLayer.predict(emgInputDataTestNS)
            # embedsEMGTest = encoderEMG.predict(initialEmbeds)
            # predClassesE = decoderEMG.predict(embedsEMGTest)
            # predClassesEAbs = np.argmax(predClassesE, axis = 1)
            # percentages = visu.createPercentages(predClassesEAbs)
            # visu.plotConfusionMatrix(emgLabelsValNS,predClassesEAbs,labelList,plt.cm.Purples)
            # visu.plotTruePredClasses(emgLabelsTestNS, predClassesEAbs)
            # visu.plotEmbeddingsLatent4(embedsEMGTest, "emg?")
            # visu.plotClassesInOrder(predClassesEAbs, emgLabelsTestNS, labelList)
            # visu.printF1Scoring(predClassesEAbs, emgLabelsValNS, labelList)
    
        
        # percentages = visu.createPercentages(predClassesEAbs)
        # visu.plotConfusionMatrix(emgLabelsValNS,predClassesEAbs,labelList)
        # visu.plotTruePredClasses(emgLabelsValNS, predClassesEAbs)
            # visu.plotEmbeddingsLatent4Color(embedsEMGTest,emgLabelsValNS,classLabels, str(startSession) + "retrainedEmbeds")
        # visu.plotClassesInOrder(predClassesEAbs, emgLabelsValNS, labelList)
    # flattenedHandInputDataTest = handRawNS# np.reshape(handRawNS,(handRawNS.shape[0]*handRawNS.shape[1],handRawNS.shape[2]))

    #%%
    percentages = visu.createPercentages(emgLabelsValNS)
    print(np.median(acc[1:11]))
    #%%
    # savePath = "E:/NonBackupThings/MaximResearch/Misdirekt/Paper/Data/"

    # np.save(savePath + "retrainingAcc.npy",acc,allow_pickle=True)
    # np.save(savePath + "retrainingFones.npy",fones,allow_pickle=True)

    #%%
    import scipy.stats
    percentages = visu.createPercentages(emgLabelsValNS)
    
    totalGuessesForMajorityCombined = np.array(totalGuessesForMajority)
    totalGuessesForMajorityCombined = scipy.stats.mode(totalGuessesForMajorityCombined,axis=0)[0]
    totalGuessesForMajorityCombined = np.squeeze(totalGuessesForMajorityCombined)
    visu.printF1Scoring(totalGuessesForMajorityCombined, emgLabelsValNS, labelList)
    
    #%%
    initialEmbeds = sandwichPreLayerTest.predict(emgInputDataValNS)
    embedsEMGTest = encoderEMGTest.predict(initialEmbeds)
    predClassesE = decoderEMGTest.predict(embedsEMGTest)
    predClassesEAbs = np.argmax(predClassesE, axis = 1)

#%%
    modelSaveNum = 48
    #25 is good performance
    #autoencoderEMG.save("Models/Injector/EMGFullAutoEncoder"+str(modelSaveNum)+".h5")
    sandwichPreLayerTest.save("Models/OneShot/EMGSandwich"+str(modelSaveNum)+".h5")
    encoderEMGTest.save("Models/OneShot/EMGEncoder"+str(modelSaveNum)+".h5")
    decoderEMGTest.save("Models/OneShot/EMGDecoder"+str(modelSaveNum)+".h5")   

