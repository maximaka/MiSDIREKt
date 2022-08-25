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
    projectDirectory = "E:/NonBackupThings/MaximResearch/SauerKraut/"  

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
    ####NO DIMENSIONALITY
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
        modelPath = "E:/NonBackupThings/MaximResearch/SauerKraut/Code/Models/Classifier/"
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
                    
                    del emgFiltered
                    
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
                trainingAccHistP.append(history1.history["categorical_accuracy"])
                trainingLossHistP.append(history1.history["loss"])
                validationAccHistP.append(history1.history["val_categorical_accuracy"])
                validationLossHistP.append(history1.history["val_loss"])
                
                # predClassesE = autoencoderEMG.predict(emgInputDataTrainC)
                # predClassesEAbs = np.argmax(predClassesE, axis = 1)
                # trainingAccuraciesP.append(np.mean(predClassesEAbs == emgLabels))

                # predClassesE = autoencoderEMG.predict(emgInputDataValC)
                # predClassesEAbs = np.argmax(predClassesE, axis = 1)
                # validationAccuraciesP.append(np.mean(predClassesEAbs == emgLabelsVal))
                
                autoencoderEMGRetrain = arch.createSequentialTrainTune([sandwichPreLayer, encoderEMG,decoderEMG], [False, True, True])
                autoencoderEMGRetrain.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)

                #Fit the autoencoder
                history1 = autoencoderEMGRetrain.fit(x = [emgInputDataTrainC], 
                                                # y=[emgInputDataTrain,labelsETrain], 
                                                y=[labelsETrainC], 
                                                # validation_data=([emgInputDataVal],[emgInputDataVal,labelsEVal]),
                                                validation_data=([emgInputDataValC],[labelsEVal]),
                                                epochs=65, batch_size=batchSize, callbacks=[callbackBig],shuffle = True)                 
                
                trainingAccHist.append(history1.history["categorical_accuracy"])
                trainingLossHist.append(history1.history["loss"])
                validationAccHist.append(history1.history["val_categorical_accuracy"])
                validationLossHist.append(history1.history["val_loss"])
                
                # predClassesE = autoencoderEMGRetrain.predict(emgInputDataTrainC)
                # predClassesEAbs = np.argmax(predClassesE, axis = 1)
                # trainingAccuracies.append(np.mean(predClassesEAbs == emgLabels))

                # predClassesE = autoencoderEMGRetrain.predict(emgInputDataValC)
                # predClassesEAbs = np.argmax(predClassesE, axis = 1)
                # validationAccuracies.append(np.mean(predClassesEAbs == emgLabelsVal))
                
            
                del emgInputDataTrainC,labelsETrainC
                del emgInputDataTrain, emgInputDataVal, labelsETrain, labelsEVal
                # del predClassesE, predClassesEAbs
                del oneHotEncodedELabels, oneHotEncodedELabelsVal
            print("DONE " + str(i))
            
        #%%
        modelSaveNum = 48
        #4 was cnn on both
        #5 is cnn on both, batch norm
        #6 is lstm on both, dropout
        #7 is 6 with cropped
        #8 is no reconstruction lstm
        #9 is many trials
        #15 is trimmed, dropout lstm
        #16 is trimmed, removed weird classes dropout lstm
        #22 is envelope
        #25 is good performance
        #26 is good oneshot
        #36 is good 80%, downsamped at 24
        #43 is non-latent
        #44 is perfect autoencoder
        #45 is perfect autoencoder, smaller
        #autoencoderEMG.save("Models/Injector/EMGFullAutoEncoder"+str(modelSaveNum)+".h5")
        encoderEMG.save("Models/Classifier/EMGEncoder"+str(modelSaveNum)+".h5")
        decoderEMG.save("Models/Classifier/EMGDecoder"+str(modelSaveNum)+".h5")
    else:
        modelLoadNum = 36
        modelPath = "E:/NonBackupThings/MaximResearch/SauerKraut/Code/Models/Classifier/"
        customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
        
        # encoderHand = keras.models.load_model(modelPath+"HandEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        # decoderHand = keras.models.load_model(modelPath+"HandDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        encoderEMG = keras.models.load_model(modelPath+"EMGEncoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        decoderEMG = keras.models.load_model(modelPath+"EMGDecoder"+str(modelLoadNum)+".h5",custom_objects=customObjectList)
        sandwichPreLayer = keras.models.load_model(modelPath+"sandwichPreLayer"+str(47)+".h5",custom_objects=customObjectList)

        autoencoderEMG = arch.createClassifierFromComponentsNoDenseParts(windowLengthE, emgDimensions, latentEMGDimensions, latentEMGDimensions, encoderEMG, decoderEMG)
        # autoencoderHand = arch.createClassifierFromComponentsNoDenseParts(windowLengthH, handDimensions, latentHandDimensions, latentHandDimensions, encoderHand, decoderHand)
    
        autoencoderEMG.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)
        # autoencoderHand.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics=metrics)

#%%
    # savePath = "E:/NonBackupThings/MaximResearch/SauerKraut/Paper/Data/"

    # np.save(savePath + "trainAccHist1.npy",trainingAccHist,allow_pickle=True)
    # np.save(savePath + "trainAccHistP1.npy",trainingAccHistP,allow_pickle = True)
    # np.save(savePath + "trainingLossHist1.npy",trainingLossHist,allow_pickle = True)
    # np.save(savePath + "trainingLossHistP1.npy",trainingLossHistP,allow_pickle = True)
    # np.save(savePath + "validationAccHist1.npy",validationAccHist,allow_pickle = True)
    # np.save(savePath + "validationAccHistP1.npy",validationAccHistP,allow_pickle = True)
    # np.save(savePath + "validationLossHist1.npy",validationLossHist,allow_pickle = True)
    # np.save(savePath + "validationLossHistP1.npy",validationLossHistP,allow_pickle = True)
    
    #%%
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
    # visu.plotConfusionMatrix(emgLabelsValNS,predClassesEAbs,labelList)
    # visu.plotTruePredClasses(emgLabelsValNS, predClassesEAbs)
    visu.plotEmbeddingsLatent4Color(embedsEMGTest,emgLabelsValNS,classLabels, "emg?")
    # visu.plotClassesInOrder(predClassesEAbs, emgLabelsValNS, labelList)
    visu.printF1Scoring(predClassesEAbs, emgLabelsValNS, labelList)
    
    accuracyE = np.mean(predClassesEAbs == emgLabelsValNS)
    print("accuracy is " + str(accuracyE))
    
#%%
    modelPath = "E:/NonBackupThings/MaximResearch/SauerKraut/Code/Models/Classifier/"
    customObjectList = { 'custom_mse_weighted': arch.custom_mse_weighted,'customDoubleLoss': arch.customDoubleLoss,'KLDivergenceLayer': arch.KLDivergenceLayer}
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 
    
    acc = []
    fones = []
    acc.append(accuracyE)
    epochNum = 19
    
    totalGuessesForMajority = []
    retrainPredsForConMat = []

    # for startSession in range(36,42):
    for startSession in range(39,40):
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

            # if iteration < 5 and startSession == 39:
            #     embedsEMGTest0 = sandwichPreLayerTest.predict(emgInputDataValNS)
            #     embedsEMGTest = encoderEMGTest.predict(embedsEMGTest0)
            #     visu.plotEmbeddingsLatent4Color(embedsEMGTest,emgLabelsValNS,classLabels, str(startSession) + "sandwichEmbeds")
       
            autoencoderEMGRetrainTest = arch.createSequentialTrainTune([sandwichPreLayerTest, encoderEMGTest,decoderEMGTest], [False, True, True])
            autoencoderEMGRetrainTest.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)
    
            history1 = autoencoderEMGRetrainTest.fit(x = [emgInputDataTrainFakeJiveNS], 
                                                    y=[labelsETrainFakeJiveNS], 
                                                    # validation_data=([emgInputDataVal],[emgInputDataVal,labelsEVal]),
                                                    validation_data=([emgInputDataTestNS],[labelsETestNS]),
                                                    epochs=75, batch_size=batchSize,shuffle = True,callbacks=[callbackBig])                 
            
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
    # savePath = "E:/NonBackupThings/MaximResearch/SauerKraut/Paper/Data/"

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
    # flattenedHandInputDataTest = handRawNS# np.reshape(handRawNS,(handRawNS.shape[0]*handRawNS.shape[1],handRawNS.shape[2]))

    
    #%%
    naiveAcc= []
    naivePredsForConMat = []
    # for startSession in range(39,40):
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

        for iteration in range(0,10):
            encoderEMGNaive = arch.createEncoderLSTM(latentEMGDimensions, windowLengthF, emgDimensions)
            decoderEMGNaive = arch.createLabelPredictor(latentEMGDimensions, classDimensions)
            # personLayers = arch.createSessionSpecializer(emgDimensions,windowLengthF,latentEMGDimensions)
            autoencoderEMGNaive = arch.createSequentialTrainTune([encoderEMGNaive,decoderEMGNaive], [True, True])
            autoencoderEMGNaive.compile(optimizer='Adam', loss=losses, loss_weights=lossWeights,metrics = metrics)
        
            history1 = autoencoderEMGNaive.fit(x = [emgInputDataTrainFakeJiveNS], 
                                                y=[labelsETrainFakeJiveNS], 
                                                # validation_data=([emgInputDataVal],[emgInputDataVal,labelsEVal]),
                                                validation_data=([emgInputDataTestNS],[labelsETestNS]),
                                                epochs=57, batch_size=batchSize,shuffle = True, callbacks=[callback])                 
            # history1 = autoencoderEMGNaive.fit(x = [emgInputDataValNS], 
            #                                     y=[labelsEValNS], 
            #                                     # validation_data=([emgInputDataVal],[emgInputDataVal,labelsEVal]),
            #                                     validation_data=([emgInputDataTrainFakeJiveNS],[labelsETrainFakeJiveNS]),
            #                                     epochs=57, batch_size=batchSize,shuffle = True, callbacks=[callback])                 
        
            print("Created Predictor and Predicted")
            
            embedsEMGTest = encoderEMGNaive.predict(emgInputDataValNS)
            predClassesE = decoderEMGNaive.predict(embedsEMGTest)
            predClassesEAbs = np.argmax(predClassesE, axis = 1)
            print("Created Predictor and Predicted")
            
            accuracyE = np.mean(predClassesEAbs == emgLabelsValNS)
            print("accuracy is " + str(accuracyE))
        
            # a,_ = visu.printF1Scoring(predClassesEAbs, emgLabelsValNS, labelList)
            
            naiveAcc.append(accuracyE)
            
            naivePredsForConMat.append([emgLabelsValNS,predClassesEAbs,embedsEMGTest])
            
        # percentages = visu.createPercentages(predClassesEAbs)
        # visu.plotConfusionMatrix(emgLabelsValNS,predClassesEAbs,labelList)
        # visu.plotTruePredClasses(emgLabelsValNS, predClassesEAbs)
        # visu.plotEmbeddingsLatent4Color(embedsEMGTest,emgLabelsValNS,classLabels, "emg?")
        # visu.plotClassesInOrder(predClassesEAbs, emgLabelsValNS, labelList)
    
#%%
    savePath = "E:/NonBackupThings/MaximResearch/SauerKraut/Paper/Data/"

    print(acc[-10:])
    # np.save(savePath + "naiveF1.npy",naiveAcc,allow_pickle=True)

    #%%
    testAcc = np.mean(acc[1:-1])
    #%%
    flattenedHandInputDataTest = handRawNS# np.reshape(handRawNS,(handRawNS.shape[0]*handRawNS.shape[1],handRawNS.shape[2]))
    flattenedHandInputDataTestNorm = np.reshape(handInputDataTest,(handInputDataTest.shape[0]*handInputDataTest.shape[1],handInputDataTest.shape[2]))
    flattenedHandInputDataTestNorm = flattenedHandInputDataTestNorm[0:flattenedHandInputDataTest.shape[0],:]
    exportHandTruths = proc.calculateAndDenormalize(flattenedHandInputDataTest,flattenedHandInputDataTestNorm)
    flattenedtesterPredsNSHand = np.reshape(testerPredsNSHand,(testerPredsNSHand.shape[0]*testerPredsNSHand.shape[1],testerPredsNSHand.shape[2]))
    exportHandPreds = proc.calculateAndDenormalize(flattenedHandInputDataTest,flattenedtesterPredsNSHand)
    #%%
    flattenedEmgInputDataTest = emgRawNS#np.reshape(emgRawNS,(emgRawNS.shape[0]*emgRawNS.shape[1],emgRawNS.shape[2]))
    flattenedEmgInputDataTestNorm = np.reshape(emgInputDataTest,(emgInputDataTest.shape[0]*emgInputDataTest.shape[1],emgInputDataTest.shape[2]))
    flattenedEmgInputDataTestNorm = flattenedEmgInputDataTestNorm[0:flattenedEmgInputDataTest.shape[0],:]
    exportEmgTruths = proc.calculateAndDenormalize(flattenedEmgInputDataTest,flattenedEmgInputDataTestNorm)
#%%
    visualizerPath = projectDirectory+"Visualization//HandVisualizer//rombolabs-hands-dataset//"
    csvFile = proc.exportToCSVHand(exportHandTruths, visualizerPath+"O2hands-stacking")
    csvFile2 = proc.exportToCSVHand(exportHandPreds, visualizerPath+"P1hands-stacking")

#%%
    # csvFile3 = proc.exportToCSVemg(exportEmgTruths, visualizerPath+"Oemg-stacking")
    
    #%%
    # latentSpace.save("Models/Injector/latentSpace"+str(modelSaveNum)+".h5")
    # emgToHand.save("Models/Injector/emgToHand"+str(modelSaveNum)+".h5")
            
#%%
    strideLength = 1
    # strideLength = windowLength
    needsShuffle = False
    timeSeriesParamH = [windowLengthH,strideLength,batchSize,needsShuffle,needsNPArray]
    timeSeriesParamE = [windowLengthE,strideLength,batchSize,needsShuffle,needsNPArray]
    
    #80 and 82
    startSession = 40
    endSession = 40
    sessionsToInclude = range(startSession,endSession)
    
    _, emgRawNS, emgFilteredNS, emgDimensions, emgColumns, emgLabelsNS = proc.preprocessSessionToDataset(startSession,activityType,"emg",needsLabels,timeSeriesParamE)
    _, handRawNS, handFilteredNS, handDimensions, handColumns, handLabelsNS = proc.preprocessSessionToDataset(startSession,activityType,"hand",needsLabels,timeSeriesParamH)
    preprocessFakeJive([2,1,128,False,True])

    oneHotEncodedELabelsNS = np_utils.to_categorical(emgLabelsNS,num_classes=21)
    oneHotEncodedELabelsNS = np.array(oneHotEncodedELabelsNS,dtype=float)
    oneHotEncodedHLabelsNS = np_utils.to_categorical(handLabelsNS,num_classes=21)
    oneHotEncodedHLabelsNS = np.array(oneHotEncodedHLabelsNS,dtype=float)
    print("Loaded Test Data")
    
    emgInputDataTest, _ = proc.splitDataset(emgFilteredNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    handInputDataTest, _ = proc.splitDataset(handFilteredNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    labelsETrainNS, _ = proc.splitDataset(oneHotEncodedELabelsNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    labelsHTrainNS, _ = proc.splitDataset(oneHotEncodedHLabelsNS, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    print('Split Test Data')

    print("Created Predictor and Predicted")
    
    embedsEMGTest = encoderEMG.predict(emgInputDataTest)
    embedsHandTest = encoderHand.predict(handInputDataTest)
    # testerPredsNSEMG,predClassesE = decoderEMG.predict(embedsEMGTest)
    testerPredsNSHand,predClassesH = decoderHandDuo.predict(embedsHandTest)
    predClassesE = decoderEMG.predict(embedsEMGTest)
    # predClassesH = decoderHand.predict(embedsHandTest)
    print("Created Predictor and Predicted")
    
    # visu.createTruthPredPlots(handInputDataTest[:2000], handInputDataTest[:2000], _ , handDimensions, handColumns, "hand")
    # visu.createTruthPredPlots(emgInputDataTest[:2000], emgInputDataTest[:2000], _ , emgDimensions,emgColumns, "emg")
    print("Calculated Plots and Embeds")
    
    #%%    
    caE = np.mean(tf.keras.metrics.categorical_accuracy(labelsETrainNS, predClassesE).numpy())
    caH = np.mean(tf.keras.metrics.categorical_accuracy(labelsHTrainNS, predClassesH).numpy())
    
    #%%
    predClassesEAbs = np.argmax(predClassesE, axis = 1)
    predClassesHAbs = np.argmax(predClassesH, axis = 1)
    
    #%%
    uniques, counts = np.unique(predClassesEAbs, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(predClassesEAbs)))
    uniquesT, countsT = np.unique(emgLabelsNS, return_counts=True)
    percentagesT = dict(zip(uniquesT, countsT * 100 / len(emgLabelsNS)))

    #%%
    
    conMatE = (confusion_matrix(emgLabelsNS, predClassesEAbs,normalize="true",labels = list(proc.interactionLabels.values())))
    plt.figure()
    fig, px = plt.subplots(figsize=(7.5, 7.5))
    px.matshow(conMatE, cmap=plt.cm.Blues, alpha=1)
    
    conMatH = (confusion_matrix(handLabelsNS, predClassesHAbs,normalize="true"))
    plt.figure()
    fig, px = plt.subplots(figsize=(7.5, 7.5))
    px.matshow(conMatH, cmap=plt.cm.Reds, alpha=1)
    
    #%%
    cStart = 0
    cEnd = -1
    plt.figure(figsize = (17.5,2))    
    # plt.plot(emgLabelsNS)
    plt.plot(emgLabelsNS[cStart:cEnd])
    # plt.figure()
    plt.plot(predClassesEAbs[cStart:cEnd])
    # plt.plot(predClassesEAbs)
    # #%%
    # visu.createTruthPredPlots(handInputDataTest[cStart:cEnd], handInputDataTest[cStart:cEnd], _ , handDimensions, handColumns, "hand")

    #%%
    accuracyE = np.mean(predClassesEAbs == emgLabelsNS)
    accuracyH = np.mean(predClassesHAbs == handLabelsNS)
    print(accuracyE)
    print(accuracyH)
    
    #%%
    # embedsEMGTest = encoderEMG.predict(emgInputDataTest)
    # embedsHandTest = encoderHand.predict(handInputDataTest)
    # embedsEMGTest = np.reshape(emgbedsEMGTest,())
    rep = embedsEMGTest
    fig, axes = plt.subplots(2,2)
    dim = 0
    plt.title("emg?")
    for row in range(2):
        for col in range(2):
            if dim == 3:
                axes[row, col].scatter(rep[:,3], rep[:,0], s = 0.2, alpha = 0.5, cmap = 'Spectral')
            else:
                axes[row, col].scatter(rep[:,dim], rep[:,dim+1], s = 0.2, alpha = 0.5, cmap = 'Spectral')
            dim += 1
    rep = embedsHandTest
    fig, axes = plt.subplots(2,2)
    dim = 0
    for row in range(2):
        for col in range(2):
            if dim == 3:
                axes[row, col].scatter(rep[:,3], rep[:,0], s = 0.2, alpha = 0.5, cmap = 'Spectral')
            else:
                axes[row, col].scatter(rep[:,dim], rep[:,dim+1], s = 0.2, alpha = 0.5, cmap = 'Spectral')
            dim += 1
    
    #%%
    dataPath = "E:/NonBackupThings/MaximResearch/SauerKraut/Data/"
    # np.save(dataPath + "latentDataToPlotModel" + str(15) + ".npy",embedsEMGTest,allow_pickle=True)
    # np.save(dataPath + "latentLabelToPlotModel" + str(15) + ".npy",emgLabelsNS,allow_pickle=True)

    #%%
    plt.figure(figsize = (36,16),dpi=1000)    
    ax = plt.subplot2grid((8, 4), (1, 2), rowspan=1, colspan=1)
    pexpand = np.expand_dims(predClassesEAbs,axis=[1])
    proc.plotColoredTrajectory3(ax,pexpand,emgLabelsNS,0,"scatter")
    ax.grid(True)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    # ax.set_yticklabels((list(proc.interactionLabels.keys())))
    ax = plt.subplot2grid((8, 4), (2, 2), rowspan=1, colspan=1)
    pexpand = np.expand_dims(emgLabelsNS,axis=[1])
    proc.plotColoredTrajectory3(ax,pexpand,emgLabelsNS,0,"line")
    ax.grid(True)
    
    #%%
    import myMetrics
    leest = (list(proc.interactionLabels.keys()))
    preds = np.array(predClassesEAbs,dtype=int)
    trues = np.array(emgLabelsNS,dtype=int)
    poo,CM = myMetrics.F1Scoring(preds,trues,leest,show=True)

#%%
    savePath = "E:/NonBackupThings/MaximResearch/SauerKraut/Paper/Data/"

    # for i in range(0,5):
    for i in [3]:
        [trueLabels, predLabels, embeds] = retrainPredsForConMat[i]
        # retrainData = pd.read_csv(savePath + "retrainConMatData.csv")
        # trueLabels = retrainData.iloc[:,0].values
        # predLabels = retrainData.iloc[:,1].values
        visu.plotConfusionMatrix(trueLabels,predLabels,labelList)
        conMatE = (confusion_matrix(trueLabels, predLabels,normalize="true"))
        visu.plotEmbeddingsLatent4Color(embeds,trueLabels,classLabels, "")
        print(str(i) + " " + str(np.mean(trueLabels == predLabels)))
    #%%
    visu.plotEmbeddingsLatent4Color(embeds,trueLabels,classLabels, "", savePath + "embedsRetrained.svg")

    plt.savefig(savePath + "embedsRetrained.svg",dpi = 600, format = "svg")
    [a,b,c] = retrainPredsForConMat[3]
    a = np.expand_dims(a,axis=1)
    b = np.expand_dims(b,axis=1)
    concat = np.concatenate([a,b],axis=1)
    np.savetxt(savePath + "retrainRawLabelsData.csv",concat,delimiter = ",")
    np.savetxt(savePath + "retrainEmbeds.csv",c,delimiter = ",")
    np.savetxt(savePath + "retrainConMatData.csv",conMatE,delimiter = ",")
    
    #%%
    # for i in [44]:
    for i in range(0,20):
        [trueLabels, predLabels, embeds] = naivePredsForConMat[i]
        visu.plotConfusionMatrix(trueLabels,predLabels,labelList)
        conMatE = (confusion_matrix(trueLabels, predLabels,normalize="true"))
        # visu.plotEmbeddingsLatent4Color(embeds,trueLabels,classLabels, "")
        print(str(i) + " " + str(np.mean(trueLabels == predLabels)))
    #%%
    plt.savefig(savePath + "naiveRetrained.svg",dpi = 600, format = "svg")
    [a,b,c] = naivePredsForConMat[44]
    concat = np.concatenate([a,b],axis=1)
    np.savetxt(savePath + "naiveRawLabelsData.csv",concat,delimiter = ",")
    np.savetxt(savePath + "naiveEmbeds.csv",c,delimiter = ",")
    np.savetxt(savePath + "naiveConMatData.csv",conMatE,delimiter = ",")

#%%
# np.save(savePath + "poop.pkl",naivePredsForConMat,allow_pickle = True)
# poo1 = np.load(savePath + "poop.pkl",allow_pickle=True)
savePath = "E:/NonBackupThings/MaximResearch/SauerKraut/Paper/Data/"
labelList = list(proc.interactionLabels.keys())
classLabels = list(proc.interactionLabels.values())

poo2 = np.load(savePath + "poop.npy",allow_pickle=True)
#[14, 30, 46]
# for i in range(40,50):
for i in [46]:
    trueLabels = poo2[i,0]
    predLabels = poo2[i,1]
    embeds = poo2[i,2]
    # visu.plotConfusionMatrix(trueLabels,predLabels,labelList)
    # conMatE = (confusion_matrix(trueLabels, predLabels,normalize="true"))
    # visu.plotEmbeddingsLatent4Color(embeds,trueLabels,classLabels, "")
    print(str(i) + " " + str(np.mean(trueLabels == predLabels)))
    
#%%
    visu.plotEmbeddingsLatent4Color(embeds,trueLabels,classLabels, "", savePath + "embedsNaive.svg")

    plt.savefig(savePath + "embedsNaive.svg",dpi = 600, format = "svg")
    [a,b,c] = [trueLabels,predLabels,embeds]
    a = np.expand_dims(a,axis=1)
    b = np.expand_dims(b,axis=1)
    concat = np.concatenate([a,b],axis=1)
    np.savetxt(savePath + "naiveRawLabelsData.csv",concat,delimiter = ",")
    np.savetxt(savePath + "naiveEmbeds.csv",c,delimiter = ",")
    np.savetxt(savePath + "naiveConMatData.csv",conMatE,delimiter = ",")

