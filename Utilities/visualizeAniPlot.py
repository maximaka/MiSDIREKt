import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utilities import myMetrics
from sklearn.metrics import confusion_matrix

from collections import deque

def printF1Scoring(trueList, predList, classList):
    preds = np.array(predList,dtype=int)
    trues = np.array(trueList,dtype=int)
    poo,CM = myMetrics.F1Scoring(preds,trues,classList,show=True)
    return poo, CM

def plotConfusionMatrix(trueList, predList, classList,cmapChoice = plt.cm.Blues):
    conMatE = (confusion_matrix(trueList, predList,normalize="true"))
    plt.figure()
    fig, px = plt.subplots(figsize=(7.5, 7.5),dpi=500)
    px.matshow(conMatE, cmap=cmapChoice, alpha=1)

def plotTruePredClasses(trueList, predList, startPoint = 0, endPoint = -1, separate = False):
    plt.figure(figsize = (17.5,2))    
    plt.plot(trueList[startPoint:endPoint])
    if separate == True:
        plt.figure(figsize = (17.5,2))    
    plt.plot(predList[startPoint:endPoint])

def plotEmbeddingsLatent4(embeds, titleText):
    xlim = 100
    ylim = 100
    rep = embeds
    fig, axes = plt.subplots(2,2)
    dim = 0
    plt.title(titleText)
    for row in range(2):
        for col in range(2):
            if dim == 3:
                axes[row, col].scatter(rep[:,3], rep[:,0], s = 0.2, alpha = 0.5, cmap = 'Spectral')
            else:
                axes[row, col].scatter(rep[:,dim], rep[:,dim+1], s = 0.2, alpha = 0.5, cmap = 'Spectral')
            dim += 1
            axes[row, col].set_ylim(-1 * xlim,xlim)
            axes[row, col].set_xlim(-1 * ylim,ylim)
            axes.majorticks_off()


def plotEmbeddingsLatent4Color(embeds, labels, classList, titleText, savePath = "No"):
    colormap = createColorPlots(3)
    print(colormap.shape)
    xlim = 100
    ylim = 100
    
    fig, axes = plt.subplots(2,2,figsize = (5,5),dpi=200)
    fig.suptitle(titleText)
    for i in range(1,len(classList)):
        dim = 0
        poses1 = embeds[labels==i]
        rep = poses1
        for row in range(2):
            for col in range(2):
                if dim == 3:
                    axes[row, col].scatter(rep[:,3], rep[:,0], s = 0.2, color=tuple(colormap[i,:]/255))
                else:
                    axes[row, col].scatter(rep[:,dim], rep[:,dim+1], s = 0.2, color=tuple(colormap[i,:]/255))
                dim += 1
                # axes[row, col].set_ylim(-1 * xlim,xlim)
                # axes[row, col].set_xlim(-1 * ylim,ylim)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
    if savePath != "No":
        plt.savefig(savePath, dpi = 600, format = "svg")



def plotClassesInOrder(predList, labelList, classList):
    plt.figure(figsize = (36,16),dpi=100)    
    ax = plt.subplot2grid((8, 4), (1, 2), rowspan=1, colspan=1)
    pexpand = np.expand_dims(predList,axis=[1])
    plotColoredTrajectory(ax,pexpand,labelList,classList,"scatter")
    ax.grid(True)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax = plt.subplot2grid((8, 4), (2, 2), rowspan=1, colspan=1)
    pexpand = np.expand_dims(labelList,axis=[1])
    plotColoredTrajectory(ax,pexpand,labelList,classList,"line")
    ax.grid(True)

def plotColoredTrajectory(axes,data,labels,classList,plotType="line"):
    colormap = createColorPlots(3)
    startInd = 0
    for i in range(1,len(classList)):
        poses1 = data[labels==i]
        poses1Ind = range(startInd,startInd+poses1.shape[0])
        if plotType == "scatter":
            plt.scatter(poses1Ind,poses1,color=tuple(colormap[i,:]/255),s=0.01,marker='x')
        elif plotType == "line":
            plt.plot(poses1Ind,poses1,c=tuple(colormap[i,:]/255),linewidth=.5)
        startInd = startInd+poses1.shape[0]

def createColorPlots(lim):
    colorMaps = []
    for red in range(0,lim):
        for blue in range(0,lim):
            for green in range(0,lim):
                colorMaps.append([(255/lim)*(red+1),(255/lim)*(blue+1),(255/lim)*(green+1),255]);
    return np.array(colorMaps)

def createPercentages(valueList):
    uniques, counts = np.unique(valueList, return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(valueList)))
    return percentages
    
def createAnimation(embeds,emgOrHand):
    # embedsPlot = np.reshape(embeds,(embeds.shape[0]*embeds.shape[1],embeds.shape[2]))

    embedsFull = np.empty((0,2))
    stopbatch = embeds.shape[0]
    for i in range(0,stopbatch):
        embedsSample = embeds[i]
        embedsFull = np.append(embedsFull,embedsSample[:,0:2],axis = 0)
    embeds = embedsFull
    #Set parameters
    upToFrame = int(embeds.shape[0]/1)
    fadeSpeed = 100
    
    #Create figure for animation
    fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_axes([0, 0, 40, 40], frameon=False)
    ax = plt.subplot(1,1,1,frameon=False)
    ax.axis('off')
    plt.xlim(np.min(embeds[:,0]), np.max(embeds[:,0]))
    plt.ylim(np.min(embeds[:,1]), np.max(embeds[:,1]))
    
    #Create data structure for points
    scatterPoints = np.zeros(upToFrame, dtype=[('position', float, (2,)),('color',    float, (4,))])
    
    #Create scatter plot to be updated
    scatter = ax.scatter(scatterPoints['position'][:, 0], scatterPoints['position'][:, 1],facecolors=scatterPoints['color'])
    
    def update(frameNumber):
    
        #Decrease the blue component of color over time, clip it at 0
        scatterPoints['color'][:, 1] -= 1.0/fadeSpeed
        scatterPoints['color'][:, 1] = np.clip(scatterPoints['color'][:, 1], 0, 1)
    
        #Set the position and colors for the new point
        scatterPoints['position'][frameNumber] = (embeds[frameNumber,0],embeds[frameNumber,1])
        scatterPoints['color'][frameNumber] = (0, 1, 0, 1)
    
        #Update the scatterplot with new colors and positions
        scatter.set_facecolors(scatterPoints['color'])
        scatter.set_offsets(scatterPoints['position'])
        # print(embeds[frameNumber,0],embeds[frameNumber,1])
    #Create the animation director, writer, and save to gif format
    ani = animation.FuncAnimation(fig, update, frames = upToFrame, interval=1)
    myWriter = animation.PillowWriter(fps=60)
    ani.save(emgOrHand + 'EncodingOverTime.gif', writer = myWriter)
    
def createAnimationDouble(embedsHand, embedsEMG):
    # embedsPlot = np.reshape(embeds,(embeds.shape[0]*embeds.shape[1],embeds.shape[2]))

    #Set parameters
    upToFrame = int(embedsHand.shape[0]/4)
    fadeSpeed = 100
    
    #Create figure for animation
    fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_axes([0, 0, 40, 40], frameon=False)
    axH = plt.subplot(1,2,1,frameon=False)
    axH.axis('off')
    plt.xlim(np.min(embedsHand[:,0]), np.max(embedsHand[:,0]))
    plt.ylim(np.min(embedsHand[:,1]), np.max(embedsHand[:,1]))
    # axH.suptitle("HAND")
    axE = plt.subplot(1,2,2,frameon=False)
    axE.axis('off')
    plt.xlim(np.min(embedsEMG[:,0]), np.max(embedsEMG[:,0]))
    plt.ylim(np.min(embedsEMG[:,1]), np.max(embedsEMG[:,1]))
    # axE.suptitle("EMG")
    
    #Create data structure for points
    scatterPointsHand = np.zeros(upToFrame, dtype=[('position', float, (2,)),('color',    float, (4,))])
    scatterPointsEMG = np.zeros(upToFrame, dtype=[('position', float, (2,)),('color',    float, (4,))])
    
    #Create scatter plot to be updated
    scatterHand = axH.scatter(scatterPointsHand['position'][:, 0], scatterPointsHand['position'][:, 1],facecolors=scatterPointsHand['color'])
    scatterEMG = axE.scatter(scatterPointsEMG['position'][:, 0], scatterPointsEMG['position'][:, 1],facecolors=scatterPointsEMG['color'])
    
    def update(frameNumber):
    
        #Decrease the blue component of color over time, clip it at 0
        scatterPointsHand['color'][:, 1] -= 1.0/fadeSpeed
        scatterPointsHand['color'][:, 1] = np.clip(scatterPointsHand['color'][:, 1], 0, 1)
    
        #Set the position and colors for the new point
        scatterPointsHand['position'][frameNumber] = (embedsHand[frameNumber,0],embedsHand[frameNumber,1])
        scatterPointsHand['color'][frameNumber] = (0, 1, 0, 1)
    
        #Update the scatterplot with new colors and positions
        scatterHand.set_facecolors(scatterPointsHand['color'])
        scatterHand.set_offsets(scatterPointsHand['position'])
        
        #Decrease the blue component of color over time, clip it at 0
        scatterPointsEMG['color'][:, 1] -= 1.0/fadeSpeed
        scatterPointsEMG['color'][:, 1] = np.clip(scatterPointsEMG['color'][:, 1], 0, 1)
    
        #Set the position and colors for the new point
        scatterPointsEMG['position'][frameNumber] = (embedsEMG[frameNumber,0],embedsEMG[frameNumber,1])
        scatterPointsEMG['color'][frameNumber] = (0, 1, 0, 1)
    
        #Update the scatterplot with new colors and positions
        scatterEMG.set_facecolors(scatterPointsEMG['color'])
        scatterEMG.set_offsets(scatterPointsEMG['position'])
        
        # print(embeds[frameNumber,0],embeds[frameNumber,1])
                
    #Create the animation director, writer, and save to gif format
    ani = animation.FuncAnimation(fig, update, frames = upToFrame, interval=1)
    myWriter = animation.PillowWriter(fps=60)
    ani.save('BothEncodingOverTime.gif', writer = myWriter)

def createTruthPredPlots(truthValues, predictionValues, embeddingValues, dimensions, columnLabels, emgOrHand):
    if(emgOrHand == "emg"):
        yLimMin = -20
        yLimMax = 20
    else:
        yLimMin = -7
        yLimMax = 7
    
    predictionPlot = np.empty((0,dimensions))
    truthPlot = np.empty((0,dimensions))
    stopbatch = truthValues.shape[0]
    for i in range(0,stopbatch):
        predictionSample = predictionValues[i]
        truthSample = truthValues[i]
        predictionPlot = np.append(predictionPlot,predictionSample[:,0:dimensions],axis = 0)
        truthPlot = np.append(truthPlot,truthSample[:,0:dimensions],axis = 0)
    
    errorPlot = np.abs((truthPlot-predictionPlot)/(truthPlot+0.001))
    errorPlot = np.square(predictionPlot - truthPlot)
    
    plt.figure(figsize = (80,80))
    plt.title("reconstruct")
    for i in range(0,dimensions):
            ax = plt.subplot2grid((dimensions, 4), (i, 0), rowspan=1, colspan=1)
            ax.plot(truthPlot[:,i],c = 'blue')
            ax.set_title(emgOrHand + " truth " + str(columnLabels[i]), c='black')
            yLimMin = np.min([-1,np.min(truthPlot[:,i])])
            yLimMax = np.max([1,np.max(truthPlot[:,i])])
            ax.set_ylim(yLimMin-0.2,yLimMax+.2)
    for i in range(0,dimensions):
            ax = plt.subplot2grid((dimensions, 4), (i, 1), rowspan=1, colspan=1)
            ax.plot(predictionPlot[:,i],c = 'orange')
            ax.set_title(emgOrHand + " predicted " + str(columnLabels[i]), c='black')
            yLimMin = np.min([-1,np.min(predictionPlot[:,i])])
            yLimMax = np.max([1,np.max(predictionPlot[:,i])])
            ax.set_ylim(yLimMin-0.2,yLimMax+.2)
    for i in range(0,dimensions):
            ax = plt.subplot2grid((dimensions, 4), (i, 2), rowspan=1, colspan=1)
            ax.plot(errorPlot[:,i],c = 'purple')
            ax.set_title(emgOrHand + " error " + str(columnLabels[i]), c='black')
            yLimMin = np.min([0,np.min(errorPlot[:,i])])
            yLimMax = np.max([2,np.max(errorPlot[:,i])])
            # ax.set_ylim(0,100)
    
    # ax = plt.subplot2grid((dimensions, 4), (3, 3), rowspan=2, colspan=1)
    # embedsPlot = np.reshape(embeddingValues,(embeddingValues.shape[0]*embeddingValues.shape[1],embeddingValues.shape[2]))
    # plt.scatter(embedsPlot[:,0], embedsPlot[:,1], cmap='Spectral', s=1)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title(emgOrHand + ' Embeddings', fontsize=12);

def createEmbedPlot(embeddingValues,emgOrHand):    
    plt.figure()
    embedsPlot = np.reshape(embeddingValues,(embeddingValues.shape[0]*embeddingValues.shape[1],embeddingValues.shape[2]))
    plt.scatter(embedsPlot[:,0], embedsPlot[:,1], cmap='Spectral', s=1)
    plt.gca().set_aspect('equal', 'datalim')
    # plt.xlim(-20,20)
    # plt.ylim(-20,20)
    plt.title("Embeddings, " + emgOrHand, fontsize=12);

def createLossPlot(history):
    plt.figure()
    # history = model.history
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    