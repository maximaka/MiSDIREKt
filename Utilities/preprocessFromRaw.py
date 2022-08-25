import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sc
import dataLoading as proc

#%%
windowLength = 1000
strideLength=windowLength
batchSize = 128
split=0.8
needsNPArray = True
datasets = "hand_emg"
strideLength = windowLength
needsShuffle = False
timeSeriesParam = [windowLength,strideLength,batchSize,needsShuffle,needsNPArray]
dataFormat = "independent"

emgRawPD, emgRaw, _, emgDimensions, emgColumns, _, _, _, _, _ = proc.preprocessBothToDataset(range(1,2),timeSeriesParam,datasets,dataFormat)
# flattenedEmgInputDataTest = emgRawNS#np.reshape(emgRawNS,(emgRawNS.shape[0]*emgRawNS.shape[1],emgRawNS.shape[2]))
# flattenedEmgInputDataTestNorm = np.reshape(emgFilteredNS,(emgFilteredNS.shape[0]*emgFilteredNS.shape[1],emgFilteredNS.shape[2]))
# flattenedEmgInputDataTestNorm = flattenedEmgInputDataTestNorm[0:flattenedEmgInputDataTest.shape[0],:]
# exportEmgTruths = proc.calculateAndDenormalize(flattenedEmgInputDataTest,flattenedEmgInputDataTestNorm)

#%%
sessionTypes = {"jive1": 0,
                "gesture1" : 1,
                "stacking" : 2,
                "interaction" : 3,
                "gesture2" : 4,
                "jive2" : 5}

def importSpecificFile(sessionNum, sessionType, sensorType):
    
    #Grab the defined dictionary of sessionTypes
    global sessionTypes
    
    #Set directory path for data
    rawDataDirectory = "E:/NonBackupThings/MaximResearch/SauerKraut/Data/RawSessionData/"
    
    sessionInfoFile = pd.read_csv(rawDataDirectory + "hands-dataset-2021-08-03-18-27-42.csv")
    
    targetFileName = sessionInfoFile[sensorType + "FileName"][sessionTypes[sessionType]]
    targetFile = pd.read_csv(rawDataDirectory + targetFileName)
    
    print('Loaded session ' + str(sessionNum) + ' of type ' + sessionType)
    return targetFile

def processEMG(emgDataIn):
    global bpFilter, nFilter1, nFilter2
    
    rail = 5
    gain = 8
    ADC = 16777215
    millivoltPerVolt = 1000
    
    emgData = emgDataIn * rail / gain / ADC * millivoltPerVolt
    emgData = sc.filtfilt(bpFilter[0], bpFilter[1], emgData,axis=0, padtype = 'odd', padlen=3*(max(len(bpFilter[0]),len(bpFilter[1]))-1))
    emgData = sc.filtfilt(nFilter1[0], nFilter1[1], emgData)
    emgData = sc.filtfilt(nFilter2[0], nFilter2[1], emgData)

    return emgData

def defineFilter():
    samplingFreq = 1001
    
    #Bandpass filter with 60 and 500 as the cutoff frequencies
    filterCoeffs1 = sc.butter(2, [60, 500], 'bandpass', fs=samplingFreq)
    
    #Notch filter with 60 as cutout freq, Quality factor of 12
    filterCoeffs2 = sc.iirnotch(60/(1), 12, samplingFreq)

    #Notch filter with 120 as cutout freq, Quality factor of 40
    filterCoeffs3 = sc.iirnotch(120/(1), 40, samplingFreq)
    
    return filterCoeffs1, filterCoeffs2, filterCoeffs3

bpFilter, nFilter1, nFilter2 = defineFilter()

# wo = 60/(1000/2);  
# bw = wo/12;
# [b,a] = iirnotch(wo,bw);
# filt = filtfilt(b,a,filt);

# wo = 120/(1000/2);  
# bw = wo/40;
# [b,a] = iirnotch(wo,bw);
# filt = filtfilt(b,a,filt);

a1 = importSpecificFile(1,'jive1',"emg")
emgStuff = processEMG(a1["emg1"])
# a2 = importSpecificFile(1,'gesture1',"hand")
# a3 = importSpecificFile(1,'stacking',"hand")
# a4 = importSpecificFile(1,'interaction',"hand")
# a5 = importSpecificFile(1,'gesture2',"hand")
# a6 = importSpecificFile(1,'jive2',"hand")

#%%
# plt.figure()
# plt.plot(emgRawPD["emg1"][0:10000])
# plt.figure()
plt.plot(emgRaw[0:20000,1])
plt.figure()
plt.plot(emgStuff[0:20000])
plt.figure()
# plt.plot(a1["emg1"][0:10000])