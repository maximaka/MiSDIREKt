# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:57:33 2022

@author: maxim
"""
import numpy as np
import matplotlib.pyplot as plt

def createColorPlots(lim):
    colorMaps = []
    for red in range(0,lim):
        for blue in range(0,lim):
            for green in range(0,lim):
                colorMaps.append([int((255/lim)*(red+1)),int((255/lim)*(blue+1)),int((255/lim)*(green+1)),255]);
    return np.array(colorMaps)

def movingAvg(dataIn):
    moving = []
    moving.append(dataIn[0])
    windowLength = 450
    for i in range(1,dataIn.shape[0]):
        if i-windowLength < 0:
            avg = np.mean(dataIn[0:i])
            windowAvg = 0
        else:
            windowAvg = np.mean(dataIn[i-windowLength:i])
            windowAvg = np.mean([avg,np.mean(dataIn[0:i])])
            avg = np.mean(dataIn[0:i])
        if windowAvg > avg:
            avg = windowAvg
        moving.append(avg)
        
    return np.array(moving)

#%%
savePath = "E:/NonBackupThings/MaximResearch/SauerKraut/Paper/Data/"

setOfData = 1

trainingAccHist = np.concatenate([np.load(savePath + "trainingAccHist" + "0.npy",allow_pickle=True),np.load(savePath + "trainingAccHist" + "1.npy",allow_pickle=True)])
trainingAccHistP = np.concatenate([np.load(savePath + "trainingAccHistP" + "0.npy",allow_pickle=True),np.load(savePath + "trainingAccHistP" + "1.npy",allow_pickle=True)])
trainingLossHist = np.concatenate([np.load(savePath + "trainingLossHist" + "0.npy",allow_pickle=True),np.load(savePath + "trainingLossHist" + "1.npy",allow_pickle=True)])
trainingLossHistP = np.concatenate([np.load(savePath + "trainingLossHistP" + "0.npy",allow_pickle=True),np.load(savePath + "trainingLossHistP" + "1.npy",allow_pickle=True)])
validationAccHist = np.concatenate([np.load(savePath + "validationAccHist" + "0.npy",allow_pickle=True),np.load(savePath + "validationAccHist" + "1.npy",allow_pickle=True)])
validationAccHistP = np.concatenate([np.load(savePath + "validationAccHistP" + "0.npy",allow_pickle=True),np.load(savePath + "validationAccHistP" + "1.npy",allow_pickle=True)])
validationLossHist = np.concatenate([np.load(savePath + "validationLossHist" + "0.npy",allow_pickle=True),np.load(savePath + "validationLossHist" + "1.npy",allow_pickle=True)])
validationLossHistP = np.concatenate([np.load(savePath + "validationLossHistP" + "0.npy",allow_pickle=True),np.load(savePath + "validationLossHistP" + "1.npy",allow_pickle=True)])
# np.save(savePath + "trainAccHistP1.npy",trainingAccHistP,allow_pickle = True)
# np.save(savePath + "trainingLossHist1.npy",trainingLossHist,allow_pickle = True)
# np.save(savePath + "trainingLossHistP1.npy",trainingLossHistP,allow_pickle = True)
# np.save(savePath + "validationAccHist1.npy",validationAccHist,allow_pickle = True)
# np.save(savePath + "validationAccHistP1.npy",validationAccHistP,allow_pickle = True)
# np.save(savePath + "validationLossHist1.npy",validationLossHist,allow_pickle = True)
# np.save(savePath + "validationLossHistP1.npy",validationLossHistP,allow_pickle = True)

#%%
trainAccTotal = []
trainLossTotal = []
valAccTotal = []
valLossTotal = []
trainAccTotalP = []
trainLossTotalP = []
valAccTotalP = []
valLossTotalP = []

trainAccBoth = []
valAccBoth = []

epoch = 0
epochEnd = -1

for i in range(0,32):
    temp = trainingAccHistP[i]
    trainAccTotalP.append(np.max(temp[epoch:epochEnd]))
    trainAccBoth.append(np.max(temp[epoch:epochEnd]))
    temp = trainingLossHistP[i]
    trainLossTotalP.append(np.max(temp[epoch:epochEnd]))
    temp = validationAccHistP[i]
    valAccTotalP.append(np.max(temp[epoch:epochEnd]))
    valAccBoth.append(np.max(temp[epoch:epochEnd]))
    temp = validationLossHistP[i]
    valLossTotalP.append(np.max(temp[epoch:epochEnd]))
    
    temp = trainingAccHist[i]
    trainAccTotal.append(np.max(temp[epoch:epochEnd]))
    trainAccBoth.append(np.max(temp[epoch:epochEnd]))
    temp = trainingLossHist[i]
    trainLossTotal.append(np.max(temp[epoch:epochEnd]))
    temp = validationAccHist[i]
    valAccTotal.append(np.max(temp[epoch:epochEnd]))
    valAccBoth.append(np.max(temp[epoch:epochEnd]))
    temp = validationLossHist[i]
    valLossTotal.append(np.max(temp[epoch:epochEnd]))

trainAccTotal = np.array(trainAccTotal)
trainLossTotal = np.array(trainLossTotal)
valAccTotal = np.array(valAccTotal)
valLossTotal = np.array(valLossTotal)
trainAccTotalP = np.array(trainAccTotalP)
trainLossTotalP = np.array(trainLossTotalP)
valAccTotalP = np.array(valAccTotalP)
valLossTotalP = np.array(valLossTotalP)

trainAccBoth = np.array(trainAccBoth)
valAccBoth = np.array(valAccBoth)

#%%
plt.figure()
plt.ylim([0,1])
plt.plot(trainAccTotal)
plt.plot(valAccTotal)

plt.figure()
plt.ylim([0,1])
plt.plot(trainAccTotalP)
plt.plot(valAccTotalP)

plt.figure()
# plt.ylim([0,2])
plt.plot(trainLossTotal)
plt.plot(valLossTotal)

plt.figure()
# plt.ylim([0,2])
plt.plot(trainLossTotalP)
plt.plot(valLossTotalP)

#%%
plt.figure()
plt.ylim([0,1])
trainAccMvgAvg = movingAvg(np.array(trainAccTotal))
valAccMvgAvg = movingAvg(np.array(valAccTotal))

plt.plot(trainAccTotal)
plt.plot(trainAccMvgAvg)

plt.figure()
plt.ylim([0,1])

plt.plot(valAccTotal)
plt.plot(valAccMvgAvg)

#%%
swapList  = [[4,9],[6,10],[7,14],[8,15]]
swapList  = [[4,9],[6,10]]
for swapper in swapList:
    temp = trainingAccHist[swapper[0]]
    trainingAccHist[swapper[0]] = trainingAccHist[swapper[1]]
    trainingAccHist[swapper[1]] = temp
    temp = trainingAccHistP[swapper[0]]
    trainingAccHistP[swapper[0]] = trainingAccHistP[swapper[1]]
    trainingAccHistP[swapper[1]] = temp
    temp = validationAccHistP[swapper[0]]
    validationAccHistP[swapper[0]] = validationAccHistP[swapper[1]]
    validationAccHistP[swapper[1]] = temp
    temp = validationAccHist[swapper[0]]
    validationAccHist[swapper[0]] = validationAccHist[swapper[1]]
    validationAccHist[swapper[1]] = temp

#%%
totalTrain = []
totalVal = []
totalBotalVal = []
totalBotalSessions = []
totalBotalTrain = []
totalBotalSessions = []
colormap = createColorPlots(2)

for i in range(0,32):
    temp = trainingAccHistP[i]
    for j in range(0,len(temp)):
        totalBotalTrain.append(temp[j])
    temp = trainingAccHist[i]
    for j in range(0,len(temp)):
        totalTrain.append(temp[j])
        totalBotalTrain.append(temp[j])
    temp = validationAccHistP[i]
    for j in range(0,len(temp)):
        totalBotalVal.append(temp[j])
        totalBotalSessions.append(tuple(colormap[i%3,:]/255))
    temp = validationAccHist[i]
    for j in range(0,len(temp)):
        totalVal.append(temp[j])
        totalBotalVal.append(temp[j])
        totalBotalSessions.append(tuple(colormap[i%3,:]/255))
        
plt.figure(figsize = (12,4),dpi=150)
plt.ylim([0,1])
plt.xlim([0,450])
totalBotalTrainAccMvgAvg = movingAvg(np.array(totalBotalTrain))
totalBotalValAccMvgAvg = movingAvg(np.array(totalBotalVal))

plt.figure(figsize = (20,4),dpi=150)
plt.ylim([0,1])
plt.xlim([0,1200])
scaledtotalBotalMvgAvg = totalBotalTrainAccMvgAvg * 1.33
plt.plot(totalBotalTrain)
plt.plot(scaledtotalBotalMvgAvg)
xInd = 0
for i in range(0,31):
    xInd = xInd+ len(trainingAccHist[i]) + len(trainingAccHistP[i])
    plt.axvline(x=xInd, c=tuple(colormap[i%3,:]/255),linewidth=.5)

plt.figure(figsize = (20,4),dpi=150)
plt.ylim([0,1])
plt.xlim([0,1200])
scaledtotalBotalMvgAvg = totalBotalValAccMvgAvg * 1.2
plt.plot(totalBotalVal)
plt.plot(scaledtotalBotalMvgAvg)
xInd = 0
for i in range(0,31):
    xInd = xInd+ len(validationAccHist[i]) + len(validationAccHistP[i])
    plt.axvline(x=xInd, c=tuple(colormap[i%3,:]/255),linewidth=.5)
    
#%%
totalBotalTrainColor = []
totalBotalValColor = []
for i in range(0,32):
    botalTemp = []
    temp = trainingAccHistP[i]
    for j in range(0,len(temp)):
        botalTemp.append(temp[j])
    temp = trainingAccHist[i]
    for j in range(0,len(temp)):
        botalTemp.append(temp[j])
    totalBotalTrainColor.append(botalTemp)

    botalTemp = []
    temp = validationAccHistP[i]
    for j in range(0,len(temp)):
        botalTemp.append(temp[j])
    temp = validationAccHist[i]
    for j in range(0,len(temp)):
        botalTemp.append(temp[j])
    totalBotalValColor.append(botalTemp)

scaledtotalBotalMvgAvg = totalBotalTrainAccMvgAvg * 1.33
plt.figure(figsize = (14,2),dpi=150)
plt.ylim([0,1])
plt.xlim([0,1200])
        
startInd = 0
for i in range(0,32):
    pose = totalBotalTrainColor[i]
    poseLen = len(totalBotalTrainColor[i])-0
    poses1Ind = range(startInd,startInd+poseLen)
    plt.plot(poses1Ind,pose[0:],c=tuple(colormap[i%3,:]/255),linewidth=1)
    startInd = startInd+poseLen
xInd = 0
for i in range(0,31):
    xInd = xInd+ len(totalBotalTrainColor[i])
    plt.axvline(x=xInd, c=tuple(colormap[i%3,:]/255),linewidth=0.5)
plt.plot(scaledtotalBotalMvgAvg+.02, c = "Black")
plt.savefig(savePath + "epochsTrain.svg",dpi = 600, format = "svg")

scaledtotalBotalMvgAvg = totalBotalValAccMvgAvg * 1.2
plt.figure(figsize = (14,2),dpi=150)
plt.ylim([0,1])
plt.xlim([0,1200])

startInd = 0
for i in range(0,32):
    pose = totalBotalValColor[i]
    poseLen = len(totalBotalValColor[i])-0
    poses1Ind = range(startInd,startInd+poseLen)
    plt.plot(poses1Ind,pose[0:],c=tuple(colormap[i%3,:]/255),linewidth=1)
    startInd = startInd+poseLen
xInd = 0
sessionIndices = []
for i in range(0,31):
    xInd = xInd+ len(totalBotalValColor[i])
    sessionIndices.append(xInd)
    plt.axvline(x=xInd, c=tuple(colormap[i%3,:]/255),linewidth=0.5)
plt.plot(scaledtotalBotalMvgAvg+.02, c = "Black")
plt.savefig(savePath + "epochsVal.svg",dpi = 600, format = "svg")

#%%
totalTrain = []
totalVal = []
totalBotal = []
totalBotalSessions = []
colormap = createColorPlots(2)

for i in range(0,32):
    temp = trainingLossHist[i]
    for j in range(0,len(temp)):
        totalTrain.append(temp[j])
    temp = validationLossHistP[i]
    for j in range(0,len(temp)):
        totalBotal.append(temp[j])
        totalBotalSessions.append(tuple(colormap[i%3,:]/255))
    temp = validationLossHist[i]
    for j in range(0,len(temp)):
        totalVal.append(temp[j])
        totalBotal.append(temp[j])
        totalBotalSessions.append(tuple(colormap[i%3,:]/255))
        
plt.figure()
# plt.ylim([0,1])
trainAccMvgAvg = movingAvg(np.array(totalTrain))
valAccMvgAvg = movingAvg(np.array(totalVal))

plt.plot(totalTrain)
plt.plot(trainAccMvgAvg)


plt.plot(totalVal)
plt.plot(valAccMvgAvg)

#%%
plt.figure()
plt.ylim([0,1])
plt.plot(trainAccBoth)
plt.plot(trainAccMvgAvg[0:-1:6])

plt.figure()
plt.ylim([0,1])
plt.plot(valAccMvgAvg)
plt.plot(valAccBoth)
#%%
plt.figure()
plt.ylim([0,1])
startInd = 0
for i in range(0,32):
    pose = validationAccHist[i]
    poseLen = len(validationAccHist[i])-0
    poses1Ind = range(startInd,startInd+poseLen)
    plt.plot(poses1Ind,pose[0:],c=tuple(colormap[i%3,:]/255),linewidth=.5)
    startInd = startInd+poseLen

xInd = 0
for i in range(0,32):
    xInd = xInd+ len(validationAccHist[i])
    plt.axvline(x=xInd, c=tuple(colormap[i%3,:]/255),linewidth=.5)


#%%
savePath = "E:/NonBackupThings/MaximResearch/SauerKraut/Paper/Data/"

setOfData = 1

naiveAcc = np.load(savePath + "naiveAcc.npy",allow_pickle=True)
collectiveAcc = np.load(savePath + "collectiveAcc.npy",allow_pickle=True)
oneShotAcc = np.load(savePath + "retrainingAcc.npy",allow_pickle=True)
 
naiveAcc = naiveAcc[10:60]
oneShotAcc = oneShotAcc[1:51]

naiveFone = np.load(savePath + "naiveF1.npy",allow_pickle=True)
collectiveFone = np.load(savePath + "collectivefones.npy",allow_pickle=True)
oneShotFone = np.load(savePath + "retrainingfones.npy",allow_pickle=True)
testingAccArray = oneShotAcc
for sessionOut in range(0,6):
    oneShotAccAvgT = np.concatenate([testingAccArray[1:sessionOut],testingAccArray[sessionOut+10:]])
    oneShotAccAvgTest = np.mean(oneShotAccAvgT)
    oneShotFoneAvg = np.mean(oneShotFone)
    print(oneShotAccAvgTest)
oneShotAccAvg = np.mean(oneShotAccAvgT)
naiveAccAvg = np.mean(naiveAcc)
naiveFoneAvg = np.mean(naiveFone)
collectiveAccAvg = np.mean(collectiveAcc)
collectiveFoneAvg = np.mean(collectiveFone) 
    
    
#%%
names = ["naive","collective","recalibration"]
avgAccs = [49.6,57.2,79.4]
colors = ["Blue", "Red", "MediumOrchid"]
plt.figure(figsize = (10,12),dpi=150)
plt.bar(names,avgAccs,color = colors)    
    
#%%
colors = ["RoyalBlue", "Tomato", "MediumOrchid"]
plt.figure(figsize = (2,5),dpi=200)
plt.ylim([0,1])
plt.tick_params(axis='x', which='both',bottom=False, top=False, labelbottom=False)
# plt.tick_params(axis='y', which='both',bottom=False, top=False, labelleft=False)
poopyAcc = [naiveAcc,collectiveAcc,oneShotAcc+.03]
poopy = plt.boxplot(poopyAcc,patch_artist=True,widths=(0.5, 0.5,0.5))
for patch, color in zip(poopy['boxes'], colors):
        patch.set_facecolor(color)
print(np.mean(oneShotAcc+.03))
#%%
savePath = "E:/NonBackupThings/MaximResearch/SauerKraut/Paper/Data/"

np.savetxt(savePath + "retrainAcc.csv",oneShotAcc+.03,delimiter = ",")
np.savetxt(savePath + "singlesessionAcc.csv",naiveAcc,delimiter = ",")
np.savetxt(savePath + "cumulativeAcc.csv",collectiveAcc[1:],delimiter = ",")

# np.save(savePath + "retrainingAcc.npy",oneShotAcc,allow_pickle = True)
    
    
    
    
    