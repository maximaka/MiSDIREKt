import numpy as np
import pandas as pd
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import scipy.io

def createSessionDocument():
    rawDataDirectory = dataPath+"SessionData/"
    sessionTopLevelFiles = []
    sessionNum = 0
    for file in os.listdir(rawDataDirectory):
        if "hands-dataset" in file:
            sessionTopLevelFiles.append([file,sessionNum])
            sessionNum = sessionNum + 1
    sessionTopLevelFiles = pd.DataFrame(data=np.array(sessionTopLevelFiles),columns=["fileName","sessionNumber"])
    sessionTopLevelFiles.to_csv(rawDataDirectory + "TopLevelFiles.csv")
    return sessionTopLevelFiles

def importSpecificFile(sessionNum, sessionType, sensorType):
    
    #Grab the defined dictionary of sessionTypes
    global sessionTypes
    
    #Set directory path for data
    rawDataPath = dataPath+"SessionData/"
    processedEMGPath = "preprocessedEMG/"
    
    #Find the top level file from the topLevel mapping
    topLevelInfoFile = pd.read_csv(rawDataPath + "TopLevelFiles.csv")
    targetTopLevelFileName = topLevelInfoFile["fileName"][topLevelInfoFile["sessionNumber"] == sessionNum].to_numpy()[0]

    #Parse session file for specific file names and find full file directory
    sessionInfoFile = pd.read_csv(rawDataPath + targetTopLevelFileName)
    
    #Determine if the desired session is in the data
    targetFileName = sessionInfoFile[sensorType + "FileName"][sessionInfoFile["trial"] == sessionType[:-1]]
    if "jive" in sessionType or "gesture" in sessionType:
        iteration = int(sessionType[-1])
        if targetFileName.shape[0] > 1:
            targetFileName = targetFileName.iloc[iteration]
        elif targetFileName.shape[0] > 0 and iteration == 0:
            targetFileName = targetFileName.iloc[0]
        else:
            print("doesn't exist: " + str(sessionNum) + " " + sessionType + " " + sensorType)
            return None
    else:
        if targetFileName.shape[0] > 0:
            targetFileName = targetFileName.iloc[0]
        else:
            print("doesn't exist: " + str(sessionNum) + " " + sessionType + " " + sensorType)
            return None
    
    #If emg, set the directory to the processed folder
    if(sensorType == "emg"):
        targetFileName = processedEMGPath + targetFileName[:-4] + "-preprocessed.csv"
    
    #Read target csv file
    targetFile = pd.read_csv(rawDataPath + targetFileName,dtype = 'float64', converters = {'targetPose': str})
    
    print('Imported session ' + str(sessionNum) + ': ' + targetFileName)
    
    return targetFile

def preprocessSessionNinaPro(sessionNum,repNum, timeSeriesParam,needsTest = False):
    if True:
        global emgColumns, handColumns 
        windowLengthE, windowLengthH, strideLength, batchSize, needShuffle, needsNPArray = timeSeriesParam
        
        projectDirectory = 'E:\\NonBackupThings\\MaximResearch\\DragonFli\\'
        dataPath = projectDirectory+'Data\\'
        emgColumns = [0,1,2,3,4,5,6,7,8,9]
        handColumns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        emgDimensions = len(emgColumns)
        handDimensions = len(handColumns)

        sessionDataEMG = scipy.io.loadmat(dataPath+'s' + str(sessionNum) + '\\' + 'S' + str(sessionNum) + '_A1_E' + str(repNum) + '.mat')["emg"]
        sessionDataHand = scipy.io.loadmat(dataPath+'s' + str(sessionNum) + '\\' + 'S' + str(sessionNum) + '_A1_E' + str(repNum) + '.mat')["glove"]
            
        dataRawEMG = sessionDataEMG
        dataRawHand = sessionDataHand
                        
        if dataRawEMG.shape[0] != 0 and dataRawHand.shape[0] != 0:
            
            dataNormedEMG = normalizeData(dataRawEMG,"emg")
            dataNormedEMG = processEMGFromRaw(dataNormedEMG)
            
            dataNormedHand = dataRawHand
            dataNormedHand = normalizeData(dataNormedHand,"hand")
                        
            sessionDataFramesEMG = []
            sessionDataFramesHand = []
            
            divFrame = 10
            
            for frame in range(0,dataNormedEMG.shape[0],strideLength):

                rawFrameWindowHand = dataNormedHand[np.max(int(frame-(10/10*np.ceil(windowLengthH))),0):np.max(int(frame+(0/10*np.ceil(windowLengthH))),0):divFrame,:]
                rawFrameWindowEMG = dataNormedEMG[np.max(int(frame-(10/10*np.ceil(windowLengthE))),0):np.max(int(frame+(0/10*np.ceil(windowLengthE))),0):divFrame,:]
  
                if rawFrameWindowEMG.shape[0] == windowLengthE/divFrame and rawFrameWindowHand.shape[0] == windowLengthH/divFrame:
                    sessionDataFramesEMG.append(rawFrameWindowEMG)
                    sessionDataFramesHand.append(rawFrameWindowHand)
                    
            dataEMG = np.array(sessionDataFramesEMG)
            dataHand = np.array(sessionDataFramesHand)
            
            print('Processed session ' + str(sessionNum) + ': ' + " dual")
            return sessionDataEMG, dataRawEMG, emgDimensions, emgColumns, sessionDataHand, dataRawHand, handDimensions, handColumns, dataEMG, dataHand
        
    print("doesn't exist?")
    return np.zeros((0,1,1)), np.zeros((0,1,1)), 0, np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1))


def preprocessSessionToDataset(sessionNum,sessionType,sensorType,needsLabels,timeSeriesParam,needsTest = False):
    if True:
        global emgColumns, handColumns 
        windowLength, strideLength, batchSize, needShuffle, needsNPArray = timeSeriesParam
    
        sessionData = importSpecificFile(sessionNum, sessionType, sensorType)
    
        if sensorType == "emg":
            columns = emgColumns
            dimensions = len(columns)
        elif sensorType == "hand":
            columns = handColumns
            dimensions = len(columns)
            sessionData = processHandAngleDiscontinuity(sessionData)
    
        dataRaw = sessionData[columns].to_numpy()
        labelSessionData = sessionData[["frameCount"]]
        labelSessionDataFake = []
        labelSessionDataFake2 = []
        
        if needsLabels:
            if sensorType == "emg":
                labelSessionData = importSpecificFile(sessionNum, sessionType, "hand")
                labelSessionData, labelSessionDataFake, labelSessionDataFake2 = processLabels(labelSessionData,sessionType, needsTest)
            elif sensorType == "hand":
                labelSessionData, labelSessionDataFake, labelSessionDataFake2 = processLabels(sessionData,sessionType, needsTest)
        if dataRaw.shape[0] != 0:
            
            dataNormed = dataRaw
            if sensorType == "hand":
                dataNormed = normalizeData(dataNormed,sensorType)
                # dataNormed = dataRaw
                
            if sensorType == "emg":
                dataNormed = processEMGFromRaw(dataNormed)
                dataNormed = normalizeData(dataNormed,sensorType)
            
            dataTrain, labelsTrain = frameSplitter(sessionData, dataNormed, labelSessionData, windowLength, needsLabels,sensorType,strideLength)
            dataVal, labelsVal = frameSplitter(sessionData, dataNormed, labelSessionDataFake, windowLength, needsLabels,sensorType,strideLength)
            if needsTest == True:
                dataTest, labelsTest = frameSplitter(sessionData, dataNormed, labelSessionDataFake2, windowLength, needsLabels,sensorType,strideLength)
                return sessionData, dataRaw, dimensions, columns, dataTrain, labelsTrain, dataVal, labelsVal, dataTest, labelsTest

            print('Processed session ' + str(sessionNum) + ': ' + sessionType + " " + sensorType)
            return sessionData, dataRaw, dimensions, columns, dataTrain, labelsTrain, dataVal, labelsVal
        
    print("doesn't exist?")
    return np.zeros((0,1,1)), np.zeros((0,1,1)), 0, np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1))

def preprocessSessionDualModes(sessionNum,sessionType,needsLabels,timeSeriesParam,needsTest = False):
    if True:
        global emgColumns, handColumns 
        windowLengthE, windowLengthH, strideLength, batchSize, needShuffle, needsNPArray = timeSeriesParam
    
        sessionDataEMG = importSpecificFile(sessionNum, sessionType, "emg")
        sessionDataHand = importSpecificFile(sessionNum, sessionType, "hand")
    
        columns = emgColumns
        emgDimensions = len(columns)
        columns = handColumns
        handDimensions = len(columns)
        sessionDataHand = processHandAngleDiscontinuity(sessionDataHand)
        sessionDataHand = sessionDataHand[sessionDataHand["tracking"] == 1]
        sessionDataEMG = sessionDataEMG.groupby('frameCount').nth(0).reset_index()
        sessionDataEMG = sessionDataEMG.assign(counter=sessionDataEMG.frameCount.isin(sessionDataHand.frameCount).astype(int))
        sessionDataEMG = sessionDataEMG[sessionDataEMG["counter"] == 1]

        # sessionDataEMG = sessionDataEMG[0:-1:13]
        # sessionDataHand = sessionDataHand.reset_index(drop = True)
        # sessionDataEMG = sessionDataEMG.reset_index(drop=True)
        # print(sessionDataEMG)
        # print(sessionDataEMG.frameCount.unique().shape)
        # print(sessionDataHand.frameCount.unique().shape)
        # sessionDataEMG = sessionDataEMG.groupby('frameCount').nth(0).reset_index()
        # sessionDataEMG = sessionDataEMG.assign(counter=sessionDataEMG.frameCount.isin(sessionDataHand.frameCount).astype(int))
        # sessionDataEMG = sessionDataEMG[sessionDataEMG["counter"] == 1]
        
        dataRawEMG = sessionDataEMG[emgColumns].to_numpy()
        dataRawHand = sessionDataHand[handColumns].to_numpy()
        labelSessionData = sessionDataHand[["frameCount"]]
        labelSessionDataFake = []
        labelSessionDataFake2 = []
        
        if needsLabels:
                labelSessionData = sessionDataHand
                labelSessionData, labelSessionDataFake, labelSessionDataFake2 = processLabels(labelSessionData,sessionType, needsTest)
                
        if dataRawEMG.shape[0] != 0 and dataRawHand.shape[0] != 0:
            
            dataNormedEMG = normalizeData(dataRawEMG,"emg")
            dataNormedEMG = processEMGFromRaw(dataNormedEMG)
            dataNormedHand = dataRawHand
            dataNormedHand = normalizeData(dataNormedHand,"hand")
            
            frameCounts = labelSessionData[['frameCount']] if needsLabels else sessionDataHand[['frameCount']]
            frameCountsHand = sessionDataHand[['frameCount']].frameCount.unique()
            frameCountsEMG = sessionDataEMG[['frameCount']].frameCount.unique()

            if frameCountsEMG.shape[0] < frameCountsHand.shape[0]:
                frameCounts = frameCountsEMG
            else:
                frameCounts = frameCountsHand
            frameStrides = frameCounts[0:frameCounts.shape[0]:strideLength]
            
            sessionDataFrames = []
            if needsLabels:
                sessionDataFramesLabel = []
            else:
                sessionDataFramesLabel = np.zeros((frameCounts.shape[0],1))
            
            sessionDataFramesEMG = []
            sessionDataFramesHand = []
            sessionDataFramesLabel = []
            print(frameCounts.shape)
            for frame in range(0,frameCounts.shape[0],strideLength):
                exactFrame = frameCounts[frame]
                startIndexHand = sessionDataHand[sessionDataHand['frameCount'] == exactFrame].index.values[0]
                startIndexEMG = sessionDataEMG[sessionDataEMG['frameCount'] == exactFrame].index.values[0]
                # print(startIndexEMG)
                # print(startIndexHand)
                rawFrameWindowHand = dataNormedHand[np.max(int(startIndexHand-(10/10*np.ceil(windowLengthH))),0):np.max(int(startIndexHand+(0/10*np.ceil(windowLengthH))),0),:]
                rawFrameWindowEMG = dataNormedEMG[np.max(int(startIndexEMG-(10/10*np.ceil(windowLengthE))),0):np.max(int(startIndexEMG+(0/10*np.ceil(windowLengthE))),0),:]
  
                # for tempFrame in range(frame,np.min([frame+windowLengthE,frameCounts.shape[0]-1])):

                #     emgFrames = dataNormedEMG[sessionDataEMG["frameCount"]==frameCounts[tempFrame]]
                #     if(emgFrames.shape[0] > 0):
                #         rawFrameWindowEMG.append(emgFrames[0,:])
                # rawFrameWindowEMG = np.array(rawFrameWindowEMG)
                # print(str(len(rawFrameWindowEMG)) + str(len(rawFrameWindowHand)))
                if rawFrameWindowEMG.shape[0] == windowLengthE and rawFrameWindowHand.shape[0] == windowLengthH:
                    sessionDataFramesEMG.append(rawFrameWindowEMG)
                    sessionDataFramesHand.append(rawFrameWindowHand)
                    if needsLabels:
                        labelFrame = labelSessionData[labelSessionData['frameCount'] == frame]
                        labelFrame = labelFrame[['targetPose']].to_numpy()
                        sessionDataFramesLabel.append(labelFrame[0])
                    
            sessionDataFrames = np.array(sessionDataFrames)
            dataEMG = np.array(sessionDataFramesEMG)
            dataHand = np.array(sessionDataFramesHand)
            sessionDataFramesLabel = np.squeeze(np.array(sessionDataFramesLabel))
            labelData = sessionDataFramesLabel
            print('Processed session ' + str(sessionNum) + ': ' + sessionType + " dual")
            return sessionDataEMG, dataRawEMG, emgDimensions, emgColumns, sessionDataHand, dataRawHand, handDimensions, handColumns, dataEMG, dataHand, labelData
        
    print("doesn't exist?")
    return np.zeros((0,1,1)), np.zeros((0,1,1)), 0, np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1)), np.zeros((0,1,1))


def processEMGFromRaw(dataRaw):
    dataNormed = dataRaw
    # dataNormed = iemg(dataNormed) 
    # dataMax = np.max(dataNormed,axis=0)
    # dataNormed = dataNormed / dataMax
    
    # dataNormed = envelopeWindow(dataNormed)
    # dataNormed = (dataNormed - np.mean(dataNormed,axis=0)) 
    # dataNormed = np.abs(dataNormed)
    
    return dataNormed
 
def frameSplitter(topLevelDataIn, dataNormIn, labelDataIn, windowLength, needsLabels,sensorType,strideLength):
    frameCounts = labelDataIn[['frameCount']] if needsLabels else topLevelDataIn[['frameCount']]
    frameCounts = frameCounts.frameCount.unique()
    
    sessionDataFrames = []
    if needsLabels:
        sessionDataFramesLabel = []
    else:
        sessionDataFramesLabel = np.zeros((frameCounts.shape[0],1))
    
    sessionDataFrames = []
    sessionDataFramesLabel = []
    # for frame in range(0,frameCounts.shape[0],strideLength):
    for frame in frameCounts:
        dataFrame = topLevelDataIn[topLevelDataIn['frameCount'] == frame]
        if dataFrame.shape[0]!=0:
            startIndex = dataFrame.index[-1]
            if sensorType == "hand":
                rawFrameWindow = dataNormIn[startIndex-int(np.ceil(windowLength)):startIndex,:]
            else:    
                rawFrameWindow = dataNormIn[startIndex-windowLength*14:startIndex:14,:]
            if rawFrameWindow.shape[0] == windowLength:
                sessionDataFrames.append(rawFrameWindow)
                if needsLabels:
                    labelFrame = labelDataIn[labelDataIn['frameCount'] == frame]
                    labelFrame = labelFrame[['targetPose']].to_numpy()
                    sessionDataFramesLabel.append(labelFrame[0])
            
    sessionDataFrames = np.array(sessionDataFrames)
    dataNorm = sessionDataFrames
    sessionDataFramesLabel = np.squeeze(np.array(sessionDataFramesLabel))
    labelData = sessionDataFramesLabel
    
    return dataNorm, labelData
    
def stdDevWindowFeature(emgDataIn):
    stdDevF = []
    for i in range(0,emgDataIn.shape[0]):
        stdDevF.append(np.std(emgDataIn[i,:,:],axis=0))
    stdDevF = np.array(stdDevF)
    stdDevF = np.expand_dims(stdDevF,axis=1)
    return stdDevF

def meanWindowFeature(emgDataIn):
    meanF = []
    for i in range(0,emgDataIn.shape[0]):
        meanF.append(np.mean(np.abs(emgDataIn[i,:,:]),axis=0))
    meanF = np.array(meanF)
    meanF = np.expand_dims(meanF,axis=1)
    return meanF

def envelopeWindow(dataIn,windowLength=360):
    mean = np.mean(dataIn,axis=0)
    dataInAbs = np.abs(dataIn - mean)#np.abs(dataIn)
    dataWindowing = []
    window = windowLength
    for i in range(0,dataInAbs.shape[0]):
        leftWindow = int(windowLength/2)
        rightWindow = int(windowLength/2)
        if i < int(windowLength/2):
            leftWindow = 0
        elif i > dataInAbs.shape[0]-int(windowLength/2):
            rightWindow = dataInAbs.shape[0]
        window = dataInAbs[i-leftWindow:i+rightWindow]
        meaned = np.sum(window,axis=0)/window.shape[0]
        dataWindowing.append(meaned)
    
    dataWindowing = np.array(dataWindowing)
    # dataWindowing = dataWindowing / np.std(dataWindowing,axis=0) / 5
    
    return np.array(dataWindowing)

def processLabels(labelData, sessionType, trainValTest = False):

    labelData["targetPose"] = labelData['targetPose'].fillna('')
    tempPoses = labelData['targetPose'].to_numpy()
    inStartingPoints = []
    outStartingPoints = []
    for i in range(0,tempPoses.shape[0]):
        if "in" in tempPoses[i][0:2]:
            inStartingPoints.append(i)
        if "exit" in tempPoses[i][0:4]:
            outStartingPoints.append(i)
    
    for i in range(0,len(inStartingPoints)):
        tempPoses[inStartingPoints[i]+0:outStartingPoints[i]-0] = tempPoses[inStartingPoints[i]]
    
    labelData["targetPose"] = tempPoses
    labelData["targetPose"] = labelData["targetPose"].map(interactionLabels) 
    labelData["targetPose"] = labelData['targetPose'].fillna(0)
    # tempPoses = labelData['targetPose'].to_numpy()
    labelData = labelData.loc[labelData["tracking"] == 1]

    thresholdJoints = handColumns[3:]
    jointThreshold = 20
    for thresholdJoint in thresholdJoints:
        for i in range(1,len(interactionLabels.keys())):
            poseData = labelData.loc[labelData["targetPose"] == i]
            medianFingerArray = poseData[thresholdJoint].to_numpy()
            medianFinger = np.median(medianFingerArray)
            badPose = poseData.loc[poseData[thresholdJoint] < (medianFinger - jointThreshold)]
            badPose2 = poseData.loc[poseData[thresholdJoint] > (medianFinger + jointThreshold)]
            labelData.loc[badPose.index.values,"targetPose"] = 0
            labelData.loc[badPose2.index.values,"targetPose"] = 0
    
    fakeJiveIndices1 = []
    fakeJiveIndices2 = []
    for i in range(1,len(interactionLabels.keys())):
    # for i in range(9,11):
        specificClass = labelData.loc[labelData["targetPose"] == i]
        # startIndex = specificClass.index[0]
        fakeJiveEnd = int(specificClass.shape[0]*.2)
        specificClassIndices = specificClass.index[0:fakeJiveEnd]
        for indice in specificClassIndices:
            fakeJiveIndices1.append(indice)
        if trainValTest == True:
            specificClassIndices = specificClass.index[fakeJiveEnd:2*fakeJiveEnd]
            for indice in specificClassIndices:
                fakeJiveIndices2.append(indice)

    labelDataFake = labelData.loc[fakeJiveIndices1]
    labelDataTrain = labelData.drop(fakeJiveIndices1) 
    
    labelDataFake2 = []
    if trainValTest == True:
        labelDataFake2 = labelData.loc[fakeJiveIndices2]
        labelDataTrain = labelData.drop(fakeJiveIndices2)
        # labelDataFake2 = labelDataFake2.loc[labelDataFake2["targetPose"] != 0]
        labelDataFake2["targetPose"] = labelDataFake2["targetPose"].map(interactionLabelsRemove) 
        labelDataFake2 = labelDataFake2.loc[labelDataFake2["targetPose"] != 0]
        # labelDataFake2 = labelDataFake2.loc[labelDataFake2["targetPose"] != 0]
        # majClass = labelDataFake2.loc[labelDataFake2["targetPose"] == 0]
        # majClassToDrop = majClass[::30]
        # majClassToDrop = majClass.drop(majClassToDrop.index) 
        # labelDataFake2 = labelDataFake2.drop(majClassToDrop.index)



    # labelDataFake = labelDataFake.loc[labelDataFake["targetPose"] != 0]
    # majClass = labelDataFake.loc[labelDataFake["targetPose"] == 0]
    # majClassToDrop = majClass[::30]
    # majClassToDrop = majClass.drop(majClassToDrop.index) 
    # labelDataFake = labelDataFake.drop(majClassToDrop.index)
    
    # labelDataTrain = labelDataTrain.loc[labelDataTrain["targetPose"] != 0]
    # majClass = labelDataTrain.loc[labelDataTrain["targetPose"] == 0]
    # majClassToDrop = majClass[::30]
    # majClassToDrop = majClass.drop(majClassToDrop.index) 
    # labelDataTrain = labelDataTrain.drop(majClassToDrop.index)
    # labelDataTrain = labelDataTrain.loc[labelDataTrain["targetPose"] != 0]
    
    labelDataFake["targetPose"] = labelDataFake["targetPose"].map(interactionLabelsRemove) 
    labelDataFake = labelDataFake.loc[labelDataFake["targetPose"] != 0]
    
    labelDataTrain["targetPose"] = labelDataTrain["targetPose"].map(interactionLabelsRemove) 
    labelDataTrain = labelDataTrain.loc[labelDataTrain["targetPose"] != 0]
    
    return labelDataTrain,labelDataFake,labelDataFake2

def processHandAngleDiscontinuity(handDataIn):
    maxColumn = (np.max(handDataIn[handColumns],axis=0))
    minColumn = (np.min(handDataIn[handColumns],axis=0))
    discont = (np.abs(maxColumn - minColumn) > 300)
    columnsToModify = discont[discont == True].index.values
    for column in columnsToModify:
        handDataIn[column] = findSeparationThreshold(handDataIn[column].to_numpy())
    return handDataIn

def findSeparationThreshold(handDataIn):
    flag = False
    threshold = -175
    while flag is False:
        underThresh = handDataIn[handDataIn<threshold]
        aboveThresh = handDataIn[handDataIn>threshold]
        if(underThresh.shape[0] == 0 or aboveThresh.shape[0] == 0):
            threshold += 1
        else:
            maxNeg = np.max(handDataIn[handDataIn<threshold])
            minPos = np.min(handDataIn[handDataIn>threshold])
            if (np.abs(maxNeg - minPos) < 20):
                threshold += 1
            else:
                flag = True
    handDataIn[handDataIn < threshold] = handDataIn[handDataIn<threshold] + 360
    return handDataIn

def filterByVariance(rawDataFrame):
    
    # Calculate variance on all columns
    variancesByColumn = rawDataFrame.var()

    badCols = []
    for colName in rawDataFrame.columns:
        #Drop
        if("render" in colName):
            badCols.append(colName)
        
        if(variancesByColumn[colName] < handJointVarThreshold):
            badCols.append(colName)
         
        ##TESTING FOR ONLY FINGER
        # if(colName != "b_l_index1_actual_oz"):
        #     badCols.append(colName)
    
    return rawDataFrame.drop(badCols,axis=1)
    
def normalizeData(data,emgOrHand = "emg"):
    if(emgOrHand == "all"):
        yMean = np.mean(data,axis=None)
        yVar = np.std(data,axis=None)
    elif(emgOrHand == "hand"):
        yMean = np.mean(data,axis=0)
        yVar = np.std(data,axis=0) * 1
    else:
        yMean = np.mean(data,axis=0)
        yVar = np.std(data,axis=0)
        # yMean = np.mean(data,axis=1)
        # yVar = np.std(data,axis=1)
    normalized = (data - yMean) / yVar
    return normalized

def calculateAndDenormalize(original,normalized,emgOrHand = "emg"):
    if(emgOrHand == "emg"):
        yMean = np.mean(original,axis=0)
        yVar = np.std(original,axis=0)
        # yMean = np.mean(original,axis=1)
        # yVar = np.std(original,axis=1)
    elif(emgOrHand == "hand"):
        yMean = np.mean(original,axis=0)
        yVar = np.std(original,axis=0) * 1
    else:
        yMean = np.tile(np.mean(original,axis=None),(normalized.shape[0],1))
        yVar = np.tile(180,(normalized.shape[0],1))
        
    denormalized = normalized * yVar + yMean
    return denormalized

def convertFromTensors(data):   
    biggestShape = 0
    for batch in data:
        if batch.shape[0] > biggestShape:
            tensorShape = batch.shape
            biggestShape = batch.shape[0]
    temp = np.empty((0,tensorShape[-2],tensorShape[-1]))
    for batch in data:
        if batch.shape == tensorShape:
            temp = np.append(temp,batch,axis = 0)
    data = temp
    return data

def convertToRealTime(data):
    tensorShape=data.shape
    temp = np.empty(tensorShape)
    for batch in data:
        batch = np.array([batch.numpy()])
        temp = np.append(temp,batch,axis = 0)
    data = temp
    return data

def processTimeseries(data,labels,windowLength,strideLength,batchSize,dimensions,needShuffle,needsNParray):
    # inputsProcessed, outputsProcessed, x = keras.preprocessing.timeseries_dataset_from_array(data = data,targets = labels, sequence_length=windowLength, sequence_stride=strideLength, batch_size=batchSize)
    inputsProcessed = keras.preprocessing.timeseries_dataset_from_array(data = data,targets = None, sequence_length=windowLength, sequence_stride=strideLength, batch_size=batchSize, shuffle = needShuffle, seed = shuffleSeed)
    outputsProcessed = keras.preprocessing.timeseries_dataset_from_array(data = labels,targets = None, sequence_length=windowLength, sequence_stride=strideLength, batch_size=batchSize, shuffle = needShuffle, seed = shuffleSeed)
    if(needsNParray):
        inputsProcessed = convertFromTensors(inputsProcessed)
        outputsProcessed = convertFromTensors(outputsProcessed)
    return inputsProcessed, np.squeeze(outputsProcessed[:,0])

def processTimeseriesNoKeras(data,labels,windowLength,strideLength,batchSize,dimensions,needShuffle,needsNParray):
    # inputsProcessed, outputsProcessed, x = keras.preprocessing.timeseries_dataset_from_array(data = data,targets = labels, sequence_length=windowLength, sequence_stride=strideLength, batch_size=batchSize)
    inputsProcessed = keras.preprocessing.timeseries_dataset_from_array(data = data,targets = None, sequence_length=windowLength, sequence_stride=strideLength, batch_size=batchSize, shuffle = needShuffle, seed = shuffleSeed)
    outputsProcessed = keras.preprocessing.timeseries_dataset_from_array(data = labels,targets = None, sequence_length=windowLength, sequence_stride=strideLength, batch_size=batchSize, shuffle = needShuffle, seed = shuffleSeed)
    if(needsNParray):
        inputsProcessed = convertFromTensors(inputsProcessed)
        outputsProcessed = convertFromTensors(outputsProcessed)
    return inputsProcessed, np.squeeze(outputsProcessed[:,0])
    
def splitDataset(data, split):
    dataTrain = data[0:int(split*len(data))]
    dataVal = data[int(split*len(data)):len(data)]
    return dataTrain, dataVal

def exportToCSVHand(handArray, filename):
    global handColumns
    handDF = importSpecificFile(0,'interactionGesture0',"hand")
    handNew = handDF.iloc[0:handArray.shape[0],:]
    handStatic = handDF.iloc[0:2,:]
    handNP = np.empty(shape=handNew.shape)
    originalColumns = handDF.columns
    # handFiltered = (filterByVariance(handDF.drop(columns = ['time','frameCount','tracking','targetPose','correlation'])))
    hColumns = handColumns
    incomingIndex = 0;
    for i in range(len(originalColumns)):
        if originalColumns[i] == 'time' or originalColumns[i] == 'frameCount':
            handNP[:,i] = range(1+i,handNew.shape[0]+1+i)
        elif originalColumns[i] == 'targetPose':
            handNP[:,i] = 1
        elif originalColumns[i] in hColumns:
            handNP[:,i] = handArray[:,incomingIndex]
            incomingIndex = incomingIndex + 1
        else:
            handNP[:,i] = np.full(shape=(handArray.shape[0]),fill_value = handStatic.loc[0,originalColumns[i]])
    handNew = pd.DataFrame(data=handNP, columns = originalColumns)
    handNew.rename(columns={'correlation': 'targetPose'})
    handNew.to_csv(filename+".csv",index=False, encoding='utf-8')
    return handNew

def exportToCSVemg(emgArray, filename):
    emgDF = importSpecificFile(0,'gesture0',"emg")
    emgNew = emgDF.iloc[0:emgArray.shape[0],:]
    emgStatic = emgDF.iloc[0:2,:]
    emgNP = np.empty(shape=emgNew.shape)
    originalColumns = emgDF.columns
    emgColumns = ['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7']
    incomingIndex = 0;
    for i in range(len(originalColumns)):
        if originalColumns[i] == 'localTimestamp' or originalColumns[i] == 'frameCount':
            emgNP[:,i] = range(2+i,emgNew.shape[0]+2+i)
        elif originalColumns[i] in emgColumns:
            emgNP[:,i] = emgArray[:,incomingIndex]
            incomingIndex = incomingIndex + 1
        else:
            emgNP[:,i] = np.full(shape=(emgArray.shape[0]),fill_value = emgStatic.loc[0,originalColumns[i]])
    emgNew = pd.DataFrame(data=emgNP, columns = originalColumns)
    emgNew = emgNew.drop(columns = ["session"])
    emgNew["counter"] = emgNew["frameCount"]
    newColumns = originalColumns.tolist()[0:2] + ["counter"] + emgColumns
    emgNew = emgNew[newColumns]
    emgNew.to_csv(filename+".csv",index=False, encoding='utf-8')
    return emgNew

def exportToCSVHandNotMod(handArray, filename):
    handArray.rename(columns={'correlation': 'targetPose', 'session': 'correlation'})
    handArray.to_csv(filename+".csv",index=False, encoding='utf-8')
    return handArray

def exportToCSVemgNotMod(emgArray, filename):
    emgNew = emgArray.drop(columns = ["session"])
    emgNew["counter"] = range(52,emgNew.shape[0]+52)
    newColumns = emgArray.columns.tolist()[0:2] + ["counter"] + emgArray.columns.tolist()[2:-1]
    emgNew = emgNew[newColumns]
    emgNew.to_csv(filename+".csv",index=False, encoding='utf-8')
    return emgNew

def expandAndNormalize(data):
    reshaped = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))
    normalizedReshaped = normalizeData(reshaped)
    normalized = np.reshape(normalizedReshaped,data.shape)
    return normalized

def findDiscrepancies():
    missingInSessions = []
    presentInSessions = []
    for sessionNum in range(0,46):
        for sessionType in sessionTypes.keys():
            for sensorType in ["emg","hand"]:
                a = importSpecificFile(sessionNum, sessionType, sensorType)
                if a is None:
                    missingInSessions.append([sessionNum, sessionType,"not exist"])
                else:
                    presentInSessions.append(a)
    missingTopLevelConnection= []
    
    rawDataPath = dataPath+"SessionData/"
    processedEMGPath = "preprocessedEMG/"
    
    for file in os.listdir(rawDataPath):
        if "hands" in file and "dataset" not in file and file not in presentInSessions:
            missingTopLevelConnection.append(file)
    for file in os.listdir(processedEMGPath):
        if "emg" in file and ("preprocessedEMG/"+file) not in presentInSessions:
            missingTopLevelConnection.append(file)
    
    return missingInSessions, missingTopLevelConnection

def findGestureReferences():
    gestures = importSpecificFile(0, "gesture0","hand")
    gesturesFrames = pd.DataFrame(columns = gestures.columns)
    for pose in inAirPoseReferenceFrames.keys():
        refFrameInd = inAirPoseReferenceFrames[pose]
        refFrame = gestures.iloc[refFrameInd]
        gesturesFrames = gesturesFrames.append(refFrame,ignore_index=True)
    
    
    yMean = np.mean(gestures[handColumns].values,axis=0)
    yVar = np.std(gestures[handColumns].values,axis=0)
        # yMean = np.mean(data,axis=1)
        # yVar = np.std(data,axis=1)
    gesturesFrames[handColumns] = (gesturesFrames[handColumns] - yMean) / yVar
    gesturesFrames.to_csv(dataPath+"inAirReferenceHands.csv")

#Set top level parameters and global parameters
projectDirectory = "E:/NonBackupThings/MaximResearch/SauerKraut/"
dataPath = projectDirectory+"Data/"

#Dictionary for session types
sessionTypes = {"jive0": 0, "gesture0" : 1, "stacking0" : 2,
                "interactionGesture0" : 3, "gesture1" : 4, "jive1" : 5}

#5 is what ive been using for a while now, use as baseline, results in 27
handJointVarThreshold = 5

handColumns = ['wrist_actual_ox', 'wrist_actual_oy', 'wrist_actual_oz',
           'b_l_index1_actual_oy', 'b_l_index1_actual_oz', 'b_l_index2_actual_oz',
           'b_l_index3_actual_oz', 'b_l_middle1_actual_oy',
           'b_l_middle1_actual_oz', 'b_l_middle2_actual_oz',
           'b_l_middle3_actual_oz', 'b_l_pinky1_actual_oy', 'b_l_pinky1_actual_oz',
           'b_l_pinky2_actual_oz', 'b_l_pinky3_actual_oz', 'b_l_ring1_actual_oy',
           'b_l_ring1_actual_oz', 'b_l_ring2_actual_oz', 'b_l_ring3_actual_oz',
           'b_l_thumb0_actual_ox', 'b_l_thumb0_actual_oy', 'b_l_thumb0_actual_oz',
           'b_l_thumb1_actual_ox', 'b_l_thumb1_actual_oy', 'b_l_thumb1_actual_oz',
           'b_l_thumb2_actual_oz', 'b_l_thumb3_actual_oz']

emgColumns = ['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7']

inAirLabels = {'':0, 'Idle':1, 'LittleAndRingFlexedTogether':2, 'MiddleFingerFlex':3,
               'IndexFlex':4, 'FingersThumbAdduction':5, 'FingersAbdThumbAdd':6, 
               'FingersThumbAbduction2':7, 'WristExtension':8, 'WristFlexion':9,
               'FingersAdduction':10, 'Four':11, 'ILoveYou':12, 'Horns':13, 'Shaka':14, 
               'FingerGun':15, 'Victory':16, 'Pinky':17, 'Point':18, 'ThumbsUp':19, 'Fist':20}

inAirPoseReferenceFrames = {'Idle':2647, 'LittleAndRingFlexedTogether':11621, 'MiddleFingerFlex':7865,
                            'IndexFlex':13994, 'FingersThumbAdduction':6704, 'FingersAbdThumbAdd':10411, 
                            'FingersThumbAbduction2':14869, 'WristExtension':4171, 'WristFlexion':369,
                            'FingersAdduction':5759, 'Four':9669, 'ILoveYou':13111, 'Horns':1387, 'Shaka':2276, 
                            'FingerGun':2944, 'Victory':4793, 'Pinky':3693, 'Point':8513, 'ThumbsUp':11201, 'Fist':12458}

interactionLabels = {'':0, 'inRest':1, 'inFistHard':2, 'inFistStiffenForearm':3, 'inFistLoose':4, 'inKeyGrip':5,
                     'inHardPinchThumbandIndex':6, 'inPinchThumbandIndex':7, 'inPinchThumbandMiddle':8, 
                     'inPinchThumbandRing':9, 'inPinchThumbandPinky':10, 'inPinchThumbandAllFingers':11, 
                      # 'inNearer':12, 'inFurther':13,#, 'inZoom':14
                     }


interactionLabelsRemove = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 
                           8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:0}

inAirLabelsRemove = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 
                     12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20}
# inAirLabelsRemove = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:0, 9:0, 10:10, 11:11, 
#                      12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20}

interactionLabelsa = {'':0, 'enterNearer':1, 'enterRest':2, 'enterFistHard':3,
                      'enterPinchThumbandIndex':4, 'enterPinchThumbandRing':5, 'enterFistStiffenForearm':6,'enterKeyGrip':7, 
                      'enterZoom':8, 'enterPinchThumbandPinky':9, 'enterPinchThumbandMiddle':10, 
                      'enterPinchThumbandAllFingers':11, 'enterFurther':12, 
                      'enterHardPinchThumbandIndex':13, 'enterFistLoose':14}

#testing for the stuffff
# handJointVarThreshold = 15
# a = importSpecificFile(1,"gesture2","hand")
# labels = a["targetPose"]
# labels = labels.unique().T
# print(labels)
shuffleSeed = 25
# a = importSpecificFile(1,'interactionGesture0',"hand")
# a = processLabels(a,"interactionGesture0")
# labels = a['targetPose']

def createColorPlots(lim):
    colorMaps = []
    for red in range(0,lim):
        for blue in range(0,lim):
            for green in range(0,lim):
                colorMaps.append([(255/lim)*(red+1),(255/lim)*(blue+1),(255/lim)*(green+1),255]);
    return np.array(colorMaps)

def plotColoredTrajectory(axes,data,labels,channel):
    colormap = createColorPlots(3)
    startInd = 0
    for i in range(1,len(interactionLabels.keys())-10):
        poses1 = data[labels==i,channel]
        # poses1 = data[labels==i]
        # poses1Ind = range(startInd,startInd+poses1.shape[0])
        poses1Ind = np.squeeze(np.argwhere(labels == i))
        plt.plot(poses1Ind,poses1,c=tuple(colormap[i,:]/255),linewidth=0.5)
        startInd = startInd+poses1.shape[0]
def plotColoredTrajectory2(axes,data,labels,channel):
    colormap = createColorPlots(3)
    startInd = 0
    for i in range(1,len(interactionLabels.keys())):
        poses1 = data[labels==i,channel]
        # poses1 = data[labels==i]
        poses1Ind = range(startInd,startInd+poses1.shape[0])
        # poses1Ind = np.squeeze(np.argwhere(labels == i))
        plt.scatter(poses1Ind,poses1,color=tuple(colormap[i,:]/255),s=0.5)
        # plt.plot(poses1Ind,poses1,c=tuple(colormap[i,:]/255),linewidth=0.5)
        startInd = startInd+poses1.shape[0]
def plotColoredTrajectory3(axes,data,labels,channel,plotType="line"):
    colormap = createColorPlots(3)
    startInd = 0
    for i in range(1,len(interactionLabels.keys())):
    # for i in range(9,11):
        poses1 = data[labels==i,channel]
        poses1Ind = range(startInd,startInd+poses1.shape[0])
        if plotType == "scatter":
            plt.scatter(poses1Ind,poses1,color=tuple(colormap[i,:]/255),s=0.01,marker='x')
        elif plotType == "line":
            plt.plot(poses1Ind,poses1,c=tuple(colormap[i,:]/255),linewidth=.5)
        startInd = startInd+poses1.shape[0]

def createPercentages(valueList):
    uniques, counts = np.unique(valueList, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(valueList)))
    print(percentages)
    return percentages

if __name__ == '__main__':

#%%
    # _, plendes, emgDimensions, emgColumns, phendes, _, handDimensions, handColumns, emgInputDataTrain, handInputDataTrain, labelData = preprocessSessionDualModes(0,"gesture0",False,[360,360,200,128,False,True])
    # peendes, _, _, _, handFilteredNS1, labels1, handFilteredNS1V, labels1V = preprocessSessionToDataset(0,"gesture0","emg",False,[360,200,128,False,True],False)
    
    # plt.figure()
    # plt.plot(emgInputDataTrain[1])
    # plt.figure()
    # plt.plot(handFilteredNS1[1])
    sessionDataEMG10 = importSpecificFile(1, "jive0", "emg")
    dataRawEMG10 = sessionDataEMG10[emgColumns].to_numpy()         
    sessionDataEMG11 = importSpecificFile(1, "jive1", "emg")
    dataRawEMG11 = sessionDataEMG11[emgColumns].to_numpy()         
    sessionDataEMG20 = importSpecificFile(2, "jive0", "emg")
    dataRawEMG20 = sessionDataEMG20[emgColumns].to_numpy()         
    sessionDataEMG21 = importSpecificFile(2, "jive1", "emg")
    dataRawEMG21 = sessionDataEMG21[emgColumns].to_numpy()         
    
    #%%
    # def processEMGFromRaw(dataRaw):
    #     dataNormed = dataRaw
    #     dataNormed = envelopeWindow(dataNormed,360) 
    #     # dataMax = np.max(dataNormed,axis=0)
    #     # dataNormed = dataNormed / dataMax
        
    #     # dataNormed = envelopeWindow(dataNormed)
    #     # dataNormed = (dataNormed - np.mean(dataNormed,axis=0)) 
    #     # dataNormed = np.abs(dataNormed)
        
    #     return dataNormed
    
    def iemg(dataIn):
        dataInAbs = np.abs(dataIn)#np.abs(dataIn)
        dataInAbs =  np.cumsum(dataInAbs,axis = 0)
        return dataIn
        # data = []
        # prevSum = 0
        # for i in range(1,dataInAbs.shape[0]):
        #     window = dataInAbs[i]
        #     prevSum = prevSum + window
        #     data.append(np.sum(window,axis=0))
        # return np.array(data)
    
    def ssi(dataIn):
        dataInAbs = np.abs(dataIn)#np.abs(dataIn)
        return np.square(dataInAbs)
        # data = []
        # prevMean = 0
        # for i in range(1,dataInAbs.shape[0]):
        #     window = dataInAbs[0:i]
        #     prevMean = (prevMean + dataInAbs[i])/i #np.sum(window,axis=0)
        #     # print(prevMean)
        #     data.append(prevMean)    
        # return np.array(data)
    
    dataNormedEMG10 = normalizeData(dataRawEMG10,"emg")
    dataNormedEMG10 = processEMGFromRaw(dataNormedEMG10)
    dataNormedEMG11 = normalizeData(dataRawEMG11,"emg")
    dataNormedEMG11 = processEMGFromRaw(dataNormedEMG11)
    dataNormedEMG20 = normalizeData(dataRawEMG20,"emg")
    dataNormedEMG20 = processEMGFromRaw(dataNormedEMG20)
    dataNormedEMG21 = normalizeData(dataRawEMG21,"emg")
    dataNormedEMG21 = processEMGFromRaw(dataNormedEMG21)
    #%%
    fig = plt.figure(figsize = (40,40))
    ylimMax = 3
    startIndex = 25000
    endIndex = 50000
    axes = fig.subplots(8,4)
    for i in range(0,8):
        axes[i,0].plot(dataNormedEMG10[startIndex:endIndex:14,i])
        axes[i,1].plot(dataNormedEMG11[startIndex:endIndex:14,i])
        axes[i,2].plot(dataNormedEMG20[startIndex:endIndex:14,i])
        axes[i,3].plot(dataNormedEMG21[startIndex:endIndex:14,i])
        axes[i,0].set_ylim([0,ylimMax])
        axes[i,1].set_ylim([0,ylimMax])
        axes[i,2].set_ylim([0,ylimMax])
        axes[i,3].set_ylim([0,ylimMax])

#%%
# if __name__ == '__main__':
    # a,aa,aaa,aaaa = preprocessFakeJive([2,1,128,False,True])
    # _, _, handFilteredNS1, _, _, labels1 = preprocessSessionToDataset(0,"interactionGesture0","hand",True,[2,1,128,False,True])
    print("eat poopy")
    #%%
    _, _, emgDimensions, emgColumns, _, handRawNS, handDimensions, handColumns, emgInputDataTrain, handInputDataTrain, labelData = preprocessSessionDualModes(1,"gesture0",False,[360,360,50,128,False,True])
    pendes, _, _, _, handFilteredNS1, labels1, handFilteredNS1V, labels1V, handTest, labels1Test = preprocessSessionToDataset(40,"interactionGesture0","emg",True,[24,1,128,False,True],True)
    handFilteredNS1=np.squeeze(handFilteredNS1)
    # _, _, _, _, emgFilteredNS1, labels1, emgFilteredNS1V, labels1V = preprocessSessionToDataset(0,"interactionGesture0","emg",True,[1,1,128,False,True])
    # emgFilteredNS1=np.squeeze(emgFilteredNS1)
    
    #%%
    percentages = createPercentages(labels1)
    percentages1 = createPercentages(labels1V)
    percentages2 = createPercentages(labels1Test)
    #%%
    labelData = pendes["targetPose"]
    labelData = labelData.fillna('')
    tempPoses = labelData.to_numpy()
    inStartingPoints = []
    outStartingPoints = []
    for i in range(0,tempPoses.shape[0]):
        if "enter" in tempPoses[i][0:5]:
            inStartingPoints.append(i)
        if "exit" in tempPoses[i][0:4]:
            outStartingPoints.append(i)
    
    for i in range(0,len(inStartingPoints)):
        tempPoses[inStartingPoints[i]+0:outStartingPoints[i]-0] = tempPoses[inStartingPoints[i]]
    
    labelData = tempPoses
    labelData = pd.Series(tempPoses)
    labelData = labelData.map(interactionLabelsa) 
    labelData = labelData.fillna(0)
    labelsT = labelData.to_numpy()
    
    #%%
    start = 1
    end = -1
    dimensions=27
    plt.figure(figsize = (80,80))
    plt.title("reconstruct")
    for i in range(0,dimensions):
            ax = plt.subplot2grid((dimensions, 4), (i, 0), rowspan=1, colspan=1)
            # plotColoredTrajectory3(ax,handFilteredNS1,labelsT[1:],i,"line")
            ax.plot(handFilteredNS1[start:end,i])
            # ax.set_ylim(-20,20)
    
    #%%
    pendes2, _, _, _, handFilteredNS2, labels2, handFilteredNS2V, labels2V = preprocessSessionToDataset(0,"gesture0","hand",False,[1,1,128,False,True])
    handFilteredNS2=np.squeeze(handFilteredNS2)
    
    #%%
    labelsG = pendes2["targetPose"]
    labelsG = labelsG.map(inAirLabels) 
    labelsG = labelsG.fillna(0)
    labelsG = labelsG.to_numpy()
    dimensions=27
    plt.figure(figsize = (80,80))
    plt.title("reconstruct")
    for i in range(0,dimensions):
            ax = plt.subplot2grid((dimensions, 4), (i, 0), rowspan=1, colspan=1)
            # plotColoredTrajectory3(ax,handFilteredNS2,labelsG[1:],i,"line")
            # print(range(0,handFilteredNS2.shape[0]))
            # print(handFilteredNS2[start:end,i].shape[0])
            ax.scatter(range(0,handFilteredNS2.shape[0]-2),handFilteredNS2[start:end,i], c = labelsG[start+1:end],s=0.01)
            # ax.set_ylim(-20,20)
    #%%
    # handFilteredNS1 = emgFilteredNS1
    # plt.figure()
    # plt.plot(handFilteredNS1[0:5000,0])
    # plt.figure()
    # plt.plot(handFilteredNS1[0:5000,1])
    # plt.figure()
    # plt.plot(handFilteredNS1[0:5000,2])
    # plt.figure()
    # plt.plot(handFilteredNS1[0:5000,3])
    # plt.figure()
    # plt.plot(handFilteredNS1[0:5000,4])
    # plt.figure()
    # plt.plot(handFilteredNS1[0:5000,5])
    # plt.figure()
    # plt.plot(handFilteredNS1[0:5000,6])
    # plt.figure()
    # plt.plot(handFilteredNS1[0:5000,7])
    # uniques, counts = np.unique(labels1, return_counts=True)
    # percentages = dict(zip(uniques, counts * 100 / len(labels1)))
    # plt.figure(figsize = (20,20),dpi=500)
    # plt.title("reconstruct")
    
    # for a in range(0,100):
    #     ax = plt.subplot2grid((25, 4), (a%25, int(a/25)), rowspan=1, colspan=1)
    #     frame = 5076 + a
    #     channel = 5
    #     ax.plot(emgFilteredNS1[frame,:,:])
    #     ax.set_ylim(0,1)
    
    #%%
    # # #%%
    # # # # def doThings():
    # # # # start = 0
    # # # # end = 30000 
    # # # # targetPost = 11
    # # # # wristLabel = "emg0"
    windowL = 1
    # _, handRaw, _, _, handFilteredNS1, labels1, _, _ = preprocessSessionToDataset(10,"interactionGesture0","emg",True,[windowL,1,128,False,True])
    # _, _, _, _, handFilteredNS2, labels2, _, _ = preprocessSessionToDataset(1,"interactionGesture0","emg",True,[windowL,1,128,False,True])
    # _, _, _, _, handFilteredNS3, labels3, _, _ = preprocessSessionToDataset(2,"interactionGesture0","emg",True,[windowL,1,128,False,True])
    # # # # # print(handFilteredNS1.shape)
    # # # # # # # #%%
    # handFilteredNS1=np.squeeze(handFilteredNS1)
    # handFilteredNS2=np.squeeze(handFilteredNS2)
    # handFilteredNS3=np.squeeze(handFilteredNS3)
    #%%
    # # # # joint = 4
    # # # # ymin = -0.5
    # # # # ymax = 0.5
    # # # # # for i in range(joint,joint + 4):
    # # # # #     fig=plt.figure(dpi=500)
    # # # # #     axes = fig.add_subplot(3,1,1)
    # # # # #     axes.plot(poses1[:,i])
    # # # # #     axes.set_ylim([ymin,ymax])
    # # # # #     axes = fig.add_subplot(3,1,2)
    # # # # #     axes.plot(poses2[:,i])
    # # # # #     axes.set_ylim([ymin,ymax])
    # # # # #     axes = fig.add_subplot(3,1,3)
    # # # # #     axes.plot(poses3[:,i])
    # # # # #     axes.set_ylim([ymin,ymax])
    # # # #%%
    # # # # #%%
    # # # # targetPost = 11
    # # # # handFilteredNS1=np.squeeze(handFilteredNS1)
    # # # # poses1 = handFilteredNS1
    # # # # handFilteredNS2=np.squeeze(handFilteredNS2)
    # # # # poses2 = handFilteredNS2
    # # # # handFilteredNS3=np.squeeze(handFilteredNS3)
    # # # # poses3 = handFilteredNS3
    # # # # joint = 4
    # # findGestureReferences()
    
    # # # # _, handRawNS, handFilteredNS3, _, _, labels3 = preprocessSessionToDataset(2,"interactionGesture0","hand",True,[1,1,128,False,True])
    # # # # handFilteredNS3=np.squeeze(handFilteredNS3)
    
    # # def createColorPlots(lim):
    # #     colorMaps = []
    # #     for red in range(0,lim):
    # #         for blue in range(0,lim):
    # #             for green in range(0,lim):
    # #                 colorMaps.append([(255/lim)*(red+1),(255/lim)*(blue+1),(255/lim)*(green+1),255]);
    # #     return np.array(colorMaps)
    # # colormap = createColorPlots(3)
    # # def plotColoredTrajectory(axes,data,labels,channel):
    # #     startInd = 0
    # #     for i in range(1,len(interactionLabels.keys())-10):
    # #         poses1 = data[labels==i,channel]
    # #         # poses1 = data[labels==i]
    # #         # poses1Ind = range(startInd,startInd+poses1.shape[0])
    # #         poses1Ind = np.squeeze(np.argwhere(labels == i))
    # #         plt.plot(poses1Ind,poses1,c=tuple(colormap[i,:]/255),linewidth=0.5)
    # #         startInd = startInd+poses1.shape[0]
    # # def plotColoredTrajectory2(axes,data,labels,channel):
    # #     startInd = 0
    # #     for i in range(1,len(interactionLabels.keys())):
    # #         poses1 = data[labels==i,channel]
    # #         # poses1 = data[labels==i]
    # #         poses1Ind = range(startInd,startInd+poses1.shape[0])
    # #         # poses1Ind = np.squeeze(np.argwhere(labels == i))
    # #         plt.plot(poses1Ind,poses1,c=tuple(colormap[i,:]/255),linewidth=0.5)
    # #         startInd = startInd+poses1.shape[0]
    # # # joint = 4
    # # # for i in range(joint,joint + 1):
    # # #     fig=plt.figure(dpi=2000)
    # # #     axes = fig.add_subplot(3,1,1)
    # # #     plotColoredTrajectory(axes,handFilteredNS1,labels1,i)
    # # #     axes = fig.add_subplot(3,1,2)
    # # #     plotColoredTrajectory(axes,handFilteredNS2,labels2,i)
    # # #     axes = fig.add_subplot(3,1,3)
    # # #     plotColoredTrajectory(axes,handFilteredNS3,labels3,i)
    # #%%
    # # start = 0
    # # end = -1
    # # dimensions=8
    # # plt.figure(figsize = (80,80))
    # # plt.title("reconstruct")
    # # for i in range(0,dimensions):
    # #         ax = plt.subplot2grid((dimensions, 4), (i, 0), rowspan=1, colspan=1)
    # #         plotColoredTrajectory3(ax,handFilteredNS1,labels1,i,"line")
    # #         # ax.plot(handFilteredNS1[start:end,i])
    # #         # ax.set_ylim(-20,20)
    # # for i in range(0,dimensions):
    # #         ax = plt.subplot2grid((dimensions, 4), (i, 1), rowspan=1, colspan=1)
    # #         plotColoredTrajectory3(ax,handFilteredNS2,labels2,i,"line")
    # #         # ax.set_ylim(-20,20)
    # # for i in range(0,dimensions):
    # #         ax = plt.subplot2grid((dimensions, 4), (i, 2), rowspan=1, colspan=1)
    # #         plotColoredTrajectory3(ax,handFilteredNS3,labels3,i,"line")
    # #         # ax.set_ylim(-20,20)
    # dimensions=8
    # plt.figure(figsize = (40,10),dpi=100)
    # plt.title("reconstruct")
    # start = 1000
    # end = start+500
    # targetPose = 9
    
    # for j in range(0,40):
    #     _, _, _, _, handFilteredNS3, elabels1, _, _ = preprocessSessionToDataset(j,"interactionGesture0","emg",True,[windowL,1,128,False,True])
    #     handFilteredNS3=np.squeeze(handFilteredNS3)
    #     if(handFilteredNS3.shape != 0):
    #         ax = plt.subplot2grid((10,4),(j%10,int(j/10)), rowspan=1, colspan=1)
    #         plotColoredTrajectory3(ax,handFilteredNS3,elabels1,5,"line")
    #         # ax.plot(emgFilteredNS1[elabels1==pose,i])
    #         ax.set_ylim(-0.5,3)
        
    # starter = 2000
    # ender = 4000
    # plt.figure(figsize = (40,10),dpi=500)
    # plt.title("reconstruct")
    # for i in range(0,8):
    #     ax = plt.subplot2grid((8, 4), (i, 0), rowspan=1, colspan=1)
    #     plotColoredTrajectory3(ax,handFilteredNS1*4,labels1,i)
    #     ax.set_ylim(-2,2)
    #     # ax.set_ylim(0,4)
    #     ax = plt.subplot2grid((8, 4), (i, 1), rowspan=1, colspan=1)
    #     plotColoredTrajectory3(ax,handFilteredNS2*4,labels2,i)
    #     ax.set_ylim(-2,2)
    #     # ax.set_ylim(0,4)
    #     ax = plt.subplot2grid((8, 4), (i, 2), rowspan=1, colspan=1)
    #     plotColoredTrajectory3(ax,handFilteredNS3*4,labels3,i)
    #     ax.set_ylim(-2,2)
    #     ax.set_ylim(0,4)
    # # # # #%%
    # # _, handRaw, handPlot, _, _, labels = preprocessSessionToDataset(0,"gesture0","hand",True,[1,1,128,False,True])
    
    # # flattenedHandInputDataTest = handRaw# np.reshape(handRawNS,(handRawNS.shape[0]*handRawNS.shape[1],handRawNS.shape[2]))
    # # flattenedHandInputDataTestNorm = np.reshape(handPlot,(handPlot.shape[0]*handPlot.shape[1],handPlot.shape[2]))
    # # # flattenedHandInputDataTestNorm = handPlot#handPlot[0:flattenedHandInputDataTest.shape[0],:]
    # # exportHandTruths = calculateAndDenormalize(flattenedHandInputDataTest,flattenedHandInputDataTestNorm)
    
    # # projectDirectory = "E:/NonBackupThings/MaximResearch/SauerKraut/"  
    # # visualizerPath = projectDirectory+"Visualization//HandVisualizer//rombolabs-hands-dataset//"
    # # csvFile = exportToCSVHand(exportHandTruths, visualizerPath+"O11hands-stacking")
    
    # #%%
        
    # # # #%%
    # # # # plt.figure()
    # # # # plt.scatter(range(start,end),a[wristLabel][start:end], marker=".")
    # # # # plt.figure()
    # # # # handFilteredNS=np.squeeze(handFilteredNS)
    # # # # d=handFilteredNS[:,2]
    # # # # b = d[d < 0]
    # # # # c = np.max(b)
    # # # # handProc = processHandAngleDiscontinuity(a)
    # # # # handDataIn = a
    # # # # maxColumn = (np.max(handDataIn[handColumns],axis=0))
    # # # # minColumn = (np.min(handDataIn[handColumns],axis=0))
    # # # # rangeOfValues = maxColumn-minColumn
    # # # # disc = handDataIn[wristLabel].to_numpy()
    # # # # plt.figure()
    # # # # plt.scatter(range(start,end),d[start:end], marker=".")
    # # # # plt.plot(range(start,end),handProc[wristLabel][start:end])
    # # # # maxMax = np.max(disc)
    # # # # minMin = np.min(disc)
    # # # # disc,thres = findSeparationThreshold(disc)
    # # # # plt.figure()
    # # # # plt.plot(disc[0:end])
    # # # # # maxNeg2 = np.max(disc[disc<0])
    # # # # maxMax2 = np.max(disc)
    # # # # minMin2 = np.min(disc)
    
    # # # # labels = labels.unique()
    # # # # print(labels)
    # # # # majClass = a.loc[a["targetPose"] == 0]
    # # # # b = a.loc[a["targetPose"] != 0]
    # # # # majClass = majClass.iloc[::30, :]
    # # # # c = pd.concat([b,majClass])
    # # # # # find highest percentage class. for in air, is "idle"
    # # # # uniques, counts = np.unique(labels, return_counts=True)
    # # # # percentages = dict(zip(uniques, counts * 100 / len(labels)))
    
    
    # # # # with open('jive_emg_preprocessedP4.pickle', 'wb') as handle:
    # # # #     pickle.dump(emgDF, handle, protocol=4)
        
    # # # # with open('jive_handP4.pickle', 'wb') as handle:
    # # # #     pickle.dump(handDF, handle, protocol=4)
    
    # # # # doThings() 