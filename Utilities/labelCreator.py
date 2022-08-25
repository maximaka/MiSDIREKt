import numpy as np
import tensorflow as tf
import pandas as pd
import preProcessPickle as proc, architectureCreation as arch, visualizeAniPlot as visu
import matplotlib.pyplot as plt

#%%
if __name__ == '__main__':
    projectDirectory = "E:/NonBackupThings/MaximResearch/SauerKraut/"  

    windowLength = 1
    strideLength = 1
    # strideLength=windowLength
    batchSize = 128
    split=0.8
    needsShuffle = False
    needsNPArray = True
    timeSeriesParam = [windowLength,strideLength,batchSize,needsShuffle,needsNPArray]
    datasets = "hand_emg"
    dataFormat = "equal"

    #30 usually
    startSession = 9
    endSession = 10
    sessionsToInclude = range(startSession,endSession)
    
    _, _, _, _, _, _, _, handFiltered, handDimensions, handColumns = proc.preprocessBothToDataset(sessionsToInclude,timeSeriesParam,datasets,dataFormat)
    handInputDataTest, _ = proc.splitDataset(handFiltered, 1) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    visu.createTruthPredPlots(handInputDataTest, handInputDataTest, _ , handDimensions, handColumns, "hand")

    handCropped = handFiltered[300:-30,:,:]
    #%%
    def createLabelForFrame(frameData):
        #index1Z
        indexVal = frameData[0,4]
        indexClass = -1
        if indexVal < -0.7:
            indexClass = 0
        # elif indexVal < -0.1 and indexVal > -0.7:
        #     indexClass = 1
        elif indexVal > -0.1:
            indexClass = 2
        else:
            indexClass = -1
        
        middleVal = frameData[0,8]
        middleClass = -1
        if middleVal < -0.55:
            middleClass = 0
        # elif middleVal < 0.2 and middleVal > -0.55:
        #     middleClass = 1
        elif middleVal > 0.2:
            middleClass = 2
        else:
            middleClass = -1
        
        pinkyVal = frameData[0,13]
        pinkyClass = -1
        if pinkyVal < -0.75:
            pinkyClass = 0
        # elif pinkyVal < 0.1 and pinkyVal > -0.75:
        #     pinkyClass = 1
        elif pinkyVal > 0.1:
            pinkyClass = 2
        else:
            pinkyClass = -1
        
        ringVal = frameData[0,17]
        ringClass = -1
        if ringVal < -0.7:
            ringClass = 0
        # elif ringVal < 0 and ringVal > -0.7:
        #     ringClass = 1
        elif ringVal > 0:
            ringClass = 2
        else:
            ringClass = -1
        
        thumbVal = frameData[0,22]
        thumbClass = -1
        if thumbVal < -0.2:
            thumbClass = 0
        # elif thumbVal < 0.4 and thumbVal > -0.2:
        #     thumbClass = 1
        elif thumbVal > 0.4:
            thumbClass = 2
        else:
            thumbClass = -1
        
        totalClass = -1
        if thumbClass == -1 or indexClass == -1 or middleClass == -1 or ringClass == -1 or pinkyClass == -1:
            totalClass = -1
        else:
            totalClass = (3**4)*pinkyClass + (3**3)*ringClass + (3**2)*middleClass + (3**1)*indexClass + (3**0)*thumbClass
        return np.array([thumbClass,indexClass,middleClass,ringClass,pinkyClass])

    def createLabelsForSession(sessionData):
        classList = np.empty((sessionData.shape[0],5))
        for i in range(0,sessionData.shape[0]):
            classList[i,:] = createLabelForFrame(sessionData[i,:,:])
        return classList

    handLabels = createLabelsForSession(handFiltered)
    print("Loaded and Processed Data")

    print("Zipped Data")
    
    handInputDataTrain, handInputDataVal = proc.splitDataset(handCropped, split) # np.expand_dims(handFiltered[0:int(split*len(handFiltered))],axis=2)
    print('Split Data')

