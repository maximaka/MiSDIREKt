import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
# from TSClusteringLayer import TSClusteringLayer as TSLayer


def createAEComplete(windowSize,inputDimensions, encoderComponent, latentComponent, decoderComponent):
    
    encoderComponent.trainable = False
    decoderComponent.trainable = False
    latentComponent.trainable = False

    inputs = layers.Input(shape=(windowSize, inputDimensions, ))
    encoder = encoderComponent(inputs)
    latent = latentComponent(encoder)
    decoder = decoderComponent(latent)
    
    return keras.Model(inputs,decoder)

def createSequentialTrainTune(listOfComponents,listOfTrainables):
    inputs = layers.Input(shape = (listOfComponents[0].input_shape[1:]))
    network = listOfComponents[0](inputs)
    for i in range(1,len(listOfComponents)):
        component = listOfComponents[i]
        component.trainable = listOfTrainables[i]
        network = component(network)
    
    return keras.Model(inputs,network)

def createAutoencoderFromComponentsNoDenseParts(windowSize,inputDimensions, latentEncDimensions, latentDecDimensions, encoderComponent, decoderComponent):
    
    encoderComponent.trainable = True
    decoderComponent.trainable = True
    inputs = layers.Input(shape=(encoderComponent.input_shape[1:]))
    encoder = encoderComponent(inputs)
    translationNetwork = layers.Dense((latentEncDimensions+latentDecDimensions)/2,activation="relu")(encoder)

    decoder = decoderComponent(translationNetwork)
    # decoder,d2 = decoderComponent(encoder)
    
    return keras.Model(inputs,decoder)
    # return keras.Model(inputs,[decoder,d2])

def createClassifierFromComponentsNoDenseParts(windowSize,inputDimensions, latentEncDimensions, latentDecDimensions, encoderComponent, decoderComponent):
    
    encoderComponent.trainable = True
    decoderComponent.trainable = True
    inputs = layers.Input(shape=(encoderComponent.input_shape[1:]))
    encoder = encoderComponent(inputs)
    decoder = decoderComponent(encoder)
    
    return keras.Model(inputs,decoder)

def createDoubleClassifier(encEMG, encHand, decEMG, decHand):
    
    encEMG.trainable = True
    encHand.trainable = True
    decEMG.trainable = True
    decHand.trainable = True
    inputsE = layers.Input(shape=(encEMG.input_shape[1:]))
    inputsH = layers.Input(shape=(encHand.input_shape[1:]))
    encoderE = encEMG(inputsE)
    encoderH = encHand(inputsH)
    
    jointLatent = layers.Concatenate()([encoderE,encoderH])
    jointToE = layers.Dense(units=(decEMG.input_shape[1]))(jointLatent)
    jointToH = layers.Dense(units=(decHand.input_shape[1]))(jointLatent)
    
    decoderE = decEMG(jointToE)
    decoderH = decHand(jointToH)
    
    return keras.Model([inputsE,inputsH],[decoderE,decoderH])

def createEncoderLSTM(latentDimensions,windowSize, inputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions,))
    
    encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True,activation='tanh'))(inputs)
    # encoder = layers.Dropout(0.25)(encoder)
    encoder = layers.Bidirectional(layers.LSTM(32, return_sequences=False,activation='tanh'))(encoder)
    encoder = layers.Dense(32, activation="tanh")(encoder)
    encoder = layers.Dropout(0.25)(encoder)
    encoder = layers.Dense(16, activation="tanh")(encoder)
    # encoder = layers.Dropout(0.25)(encoder)
    latentSpace = layers.Dense(latentDimensions)(encoder)
    
    return keras.Model(inputs, latentSpace)

def createEncoderCNN(latentDimensions,windowSize, inputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions,1,))
    
    encoder = layers.Conv2D(32, (3, 2),padding='same')(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    encoder = layers.Dropout(0.05)(encoder)
    encoder = layers.MaxPooling2D((5, 2))(encoder)

    encoder = layers.Conv2D(128, (3, 2),padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    encoder = layers.Dropout(0.05)(encoder)
    encoder = layers.MaxPooling2D((4, 2))(encoder)

    # encoder = layers.Conv2D(256, (3, 2),padding='same')(encoder)
    # encoder = layers.BatchNormalization()(encoder)
    # encoder = layers.Activation("relu")(encoder)
    # encoder = layers.Dropout(0.05)(encoder)
    # encoder = layers.MaxPooling2D((2, 2))(encoder)
    
    x = layers.Flatten()(encoder)
    
    # latentSpace = layers.Dense(latentDimensions*4, activation="relu")(x)
    # latentSpace = layers.Dense(16, activation="relu")(latentSpace)
    latentSpace = layers.Dense(latentDimensions)(x)
    
    return keras.Model(inputs, latentSpace)

def createDecoderCNN(latentDimensions,windowSize, inputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions,1,))
    
    # decoder = layers.Conv2D(256, (3, 2),padding='same')(inputs)
    # decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation("relu")(decoder)
    # decoder = layers.Dropout(0.05)(decoder)
    # decoder = layers.UpSampling2D((5,2))(decoder)
    
    decoder = layers.Conv2D(128, (3, 2),padding='same')(inputs)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Dropout(0.05)(decoder)
    decoder = layers.UpSampling2D((4,2))(decoder)
    
    decoder = layers.Conv2D(32, (3, 2),padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Dropout(0.05)(decoder)
    decoder = layers.UpSampling2D((2,2))(decoder)
    
    return keras.Model(inputs, decoder)

def createEncoderCoimbra(latentDimensions,windowSize, inputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions,))
    
    encoder = layers.Dense(300,activation='tanh')(inputs)
    encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=False,activation='tanh'))(encoder)

    latentSpace = layers.Dense(latentDimensions)(encoder)
    
    return keras.Model(inputs, latentSpace)

def createDecoderLabels(latentDimensions,windowSize, outputDimensions, classDimensions):
####    latentSpace = layers.Input(shape=(windowSize, latentDimensions, ))
    latentSpace = layers.Input(shape=(latentDimensions, ))
    labelPred = createLabelPredictor(latentDimensions,classDimensions)
    decoderToSave = createDecoder(latentDimensions,windowSize, outputDimensions)
    
    labelPred = labelPred(latentSpace)
    decoder = decoderToSave(latentSpace)

    return keras.Model(latentSpace, [decoder,labelPred],name='decoder'), labelPred, decoderToSave

def createLabelPredictor(latentDimensions, classDimensions):
####    latentSpace = layers.Input(shape=(windowSize, latentDimensions, ))
    latentSpace = layers.Input(shape=(latentDimensions, ))
    
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    # decoder = layers.Dropout(0.25)(decoder)
    decoder = layers.Dense(32, activation="relu")(decoder)
    decoder = layers.Dropout(0.25)(decoder)
    decoder = layers.RepeatVector(32)(decoder)
    decoder = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(decoder)
    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(decoder)
    classes = layers.Dense(classDimensions, activation="softmax")(decoder)  

    return keras.Model(latentSpace, classes)

def createDecoder(latentDimensions,windowSize, outputDimensions):
####    latentSpace = layers.Input(shape=(windowSize, latentDimensions, ))
    latentSpace = layers.Input(shape=(latentDimensions, ))
    
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    decoder = layers.Dense(64, activation="relu")(decoder)
    decoder = layers.RepeatVector(windowSize)(decoder)
    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)
    decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoder)
    outputs = layers.Dense(outputDimensions, activation="linear")(decoder)  

    return keras.Model(latentSpace, outputs)

def createPredictor(latentDimensions,windowSize, outputDimensions):
####    latentSpace = layers.Input(shape=(windowSize, latentDimensions, ))
    latentSpace = layers.Input(shape=(latentDimensions, ))
    
    decoder = layers.Dense(64, activation="relu")(latentSpace)
    decoder = layers.RepeatVector(windowSize)(decoder)
    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)
    outputs = layers.Dense(outputDimensions, activation="relu")(decoder)  

    return keras.Model(latentSpace, outputs)
    
def createSessionSpecializer(inputDimensions, windowSize, latentDimensions):
    if windowSize == 1:
        inputs = layers.Input(shape = (inputDimensions, ))
    else:
        inputs = layers.Input(shape = (windowSize, inputDimensions, ))
    personLayer = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    personLayer = layers.Dense(64)(personLayer)
    outputs = layers.Dense(inputDimensions)(personLayer)
    
    return keras.Model(inputs,outputs)

def createEncoderLSTMFakeLatent(latentDimensions,windowSize, inputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions,))
    
    encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True,activation='tanh'))(inputs)
    # encoder = layers.Dropout(0.25)(encoder)
    encoder = layers.Bidirectional(layers.LSTM(32, return_sequences=True,activation='tanh'))(encoder)
    encoder = layers.Dense(32, activation="linear")(encoder)
    encoder = layers.Dropout(0.25)(encoder)
    encoder = layers.Dense(16, activation="linear")(encoder)
    # encoder = layers.Dropout(0.25)(encoder)
    latentSpace = layers.Dense(latentDimensions)(encoder)
    
    return keras.Model(inputs, latentSpace)
def createDecoderLSTMFakeLatent(latentDimensions,windowSize, outputDimensions):
####    latentSpace = layers.Input(shape=(windowSize, latentDimensions, ))
    latentSpace = layers.Input(shape=(windowSize, latentDimensions, ))
    
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    decoder = layers.Dense(64, activation="relu")(decoder)
    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)
    decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoder)
    outputs = layers.Dense(outputDimensions, activation="linear")(decoder)  

    return keras.Model(latentSpace, outputs)
##
def createNeuroPoseArch(windowLength, inputDimensions, outputDimensions):
    scaleFactor = 2
    upscaleFactor = 3
    inputs = layers.Input(shape = (windowLength,inputDimensions,1))

    encoder = layers.Conv2D(32, (3, 2),padding='same')(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("linear")(encoder)
    encoder = layers.Dropout(0.05)(encoder)
    encoder = layers.MaxPooling2D((5, scaleFactor))(encoder)

    encoder = layers.Conv2D(128, (3, 2),padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    encoder = layers.Dropout(0.05)(encoder)
    encoder = layers.MaxPooling2D((4, scaleFactor))(encoder)

    encoder = layers.Conv2D(256, (3, 2),padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    encoder = layers.Dropout(0.05)(encoder)
    encoder = layers.MaxPooling2D((2, scaleFactor))(encoder)
    
    
    decoder = layers.Conv2D(256, (3, 2),padding='same')(encoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Dropout(0.05)(decoder)
    decoder = layers.UpSampling2D((5,11))(decoder)
    
    decoder = layers.Conv2D(128, (3, 2),padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Dropout(0.05)(decoder)
    decoder = layers.UpSampling2D((5,2))(decoder)
    
    decoder = layers.Conv2D(32, (3, 2),padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Dropout(0.05)(decoder)
    decoder = layers.UpSampling2D((2,1))(decoder)

    outputs = layers.Conv2D(1, (1, 1),padding='same')(decoder)
    outputs = layers.Reshape((windowLength,22))(outputs)
    outputs = (layers.LSTM(22, return_sequences=True))(outputs)
    
    return keras.Model(inputs,outputs)

def createNeuroPoseArchMod(windowLength, inputDimensions, outputDimensions):
    scaleFactor = 2
    upscaleFactor = 3
    inputs = layers.Input(shape = (windowLength,inputDimensions,1))
    inputs = layers.Input(shape = (windowLength,inputDimensions))

    # encoder = layers.Conv2D(32, (3, 2),padding='same')(inputs)
    encoder = layers.Bidirectional(layers.LSTM(32, return_sequences=True,activation='tanh'))(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    encoder = layers.Dropout(0.05)(encoder)
    # encoder = layers.MaxPooling2D((5, scaleFactor))(encoder)

    # encoder = layers.Conv2D(128, (3, 2),padding='same')(encoder)
    encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True,activation='tanh'))(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    encoder = layers.Dropout(0.05)(encoder)
    # encoder = layers.MaxPooling2D((4, scaleFactor))(encoder)

    # # encoder = layers.Reshape((360,1,256))(encoder)
    # # encoder = layers.Conv2D(256, (3, 2),padding='same')(encoder)
    # encoder = layers.Bidirectional(layers.LSTM(256, return_sequences=True,activation='tanh'))(encoder)
    # encoder = layers.BatchNormalization()(encoder)
    # encoder = layers.Activation("relu")(encoder)
    # encoder = layers.Dropout(0.05)(encoder)
    # # encoder = layers.MaxPooling2D((2, scaleFactor))(encoder)
    
    # encoder = layers.Reshape((25,256,1))(encoder)
    # # encoder = layers.Repeat(3)(encoder)
    # # presnet = tf.keras.applications.ResNet50()
    # # print(presnet.summary())
    # encoder = tf.keras.applications.resnet50.preprocess_input(encoder)
    # outputs = tf.keras.applications.ResNet50(include_top = False,pooling=None)(encoder)

    # # decoder = layers.Conv2D(256, (3, 2),padding='same')(encoder)
    # decoder = layers.Bidirectional(layers.LSTM(256, return_sequences=True,activation='tanh'))(encoder)
    # decoder = layers.BatchNormalization()(decoder)
    # decoder = layers.Activation("relu")(decoder)
    # decoder = layers.Dropout(0.05)(decoder)
    # # decoder = layers.UpSampling2D((5,upscaleFactor))(decoder)
    # # decoder = layers.Reshape((360,256))(decoder)
    
    # decoder = layers.Conv2D(128, (3, 2),padding='same')(decoder)
    decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True,activation='tanh'))(encoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Dropout(0.05)(decoder)
    # decoder = layers.UpSampling2D((4,upscaleFactor))(decoder)
    
    # decoder = layers.Conv2D(32, (3, 2),padding='same')(decoder)
    decoder = layers.Bidirectional(layers.LSTM(32, return_sequences=True,activation='tanh'))(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = layers.Dropout(0.05)(decoder)
    # decoder = layers.UpSampling2D((2,upscaleFactor))(decoder)

    # outputs = layers.Conv2D(1, (1, 1),padding='same')(decoder)
    # outputs = layers.Reshape((360,27))(decoder)
    outputs = (layers.LSTM(27, return_sequences=True))(decoder)
    outputs = layers.Dense(27,"linear")(outputs)
    
    return keras.Model(inputs,outputs)

#EXPERIMENTAL
#THIS IS MODIFIED FOR LABELLED HAND AND EMG
def createDecodeRetrainer(encoderEMG, encoderHand, decoderEMG, decoderHand):
    encoderEMG.trainable = False
    encoderHand.trainable = False
    decoderHand.trainable = False
    
    inputsH = layers.Input(shape=(encoderHand.input_shape[1:]))
    encoderH = encoderHand(inputsH)
    # decoderH,_ = decoderHand(encoderH)

    # inputsE = layers.Input(shape=(encoderEMG.input_shape[1:]))
    # encoderE = encoderEMG(inputsE)
    decoderE,d2 = decoderEMG(encoderH)
    
    return keras.Model(inputsH,[decoderE,d2])

def createEncodeRetrainer(encoderEMG, encoderHand, decoderEMG, decoderHand):
    decoderEMG.trainable = False
    encoderHand.trainable = False
    decoderHand.trainable = False
    
    # inputsH = layers.Input(shape=(encoderHand.input_shape[1:]))
    # encoderH = encoderHand(inputsH)
    # decoderH,_ = decoderHand(encoderH)

    inputsE = layers.Input(shape=(encoderEMG.input_shape[1:]))
    encoderE = encoderEMG(inputsE)
    decoderE,d2 = decoderEMG(encoderE)
    
    return keras.Model(inputsE,[decoderE,d2])
    
#THIS IS MODIFIED FOR CLIP COPY
def createCLIPcopy(batchSize, encoderEMG, encoderHand, decoderEMG, decoderHand):   
    t = 2
    encoderEMG.trainable = True
    encoderHand.trainable = True
    decoderEMG.trainable = True
    decoderHand.trainable = True

    inputsEMG = layers.Input(shape=(encoderEMG.input_shape[1:]))#, batch_size=batchSize)
    inputsHand = layers.Input(shape=(encoderHand.input_shape[1:]))#, batch_size=batchSize)
    # compressEMG = layers.Reshape((inputsEMG.shape[1:]))(inputsEMG)
    # compressEMG = tf.squeeze(inputsEMG)
    # compressEMG = layers.Reshape((inputsEMG.shape[1:]))(inputsEMG)
    encoderE = encoderEMG(inputsEMG)
    encoderH = encoderHand(inputsHand)
    
    # tf.print(encoderE)
    # tf.print(encoderH)
    logits = tf.tensordot(tf.transpose(encoderE),encoderH,axes=1) * np.exp(t)
    # labels = np.arange(encoderEMG.input_shape[0])
    labels = np.arange(4,dtype=float)
    labelArray = np.zeros((4,4))
    labels = np.fill_diagonal(labelArray,labels)
    labels = tf.convert_to_tensor(labelArray)
    # labels = tf.reshape(labels,(batchSize,1))
    logits, labels,encoderE,encoderH = symmetricLossFunction()([logits, labels,encoderE,encoderH])

    decoderE = decoderEMG(encoderE)
    decoderH = decoderHand(encoderH)
    
    return keras.Model([inputsEMG,inputsHand], [decoderE,decoderH])

class symmetricLossFunction(layers.Layer):

    """ Identity transform layer that adds symmetric CLIP
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(symmetricLossFunction, self).__init__(*args, **kwargs)

    def call(self, inputs):

        logits, labels,encE,encH = inputs
        
        # tf.print(encE)
        # tf.print(encH)
        logits = tf.tensordot(tf.transpose(encE),encH,axes=1) * np.exp(2)
        losserE = tf.keras.losses.CategoricalCrossentropy(axis=0)
        losserH = tf.keras.losses.CategoricalCrossentropy(axis=1)

        lossE = losserE(logits,labels)
        lossH = losserH(logits,labels)

        tf.print(lossE)
        tf.print(lossH)

        self.add_loss(10*(lossE+lossH)/2, inputs=inputs)
        return inputs

#THIS IS MODIFIED FOR CLUSTERING ATTEMPT
def createAutoencoderLSTM(latentDimensions,windowSize, handDimensions):
    inputs = layers.Input(shape=(windowSize, handDimensions, ))
    
    encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder)
    encoder = layers.Dense(64, activation="relu")(encoder)
    encoder = layers.Dense(16, activation="relu")(encoder)
    latentSpace = layers.Dense(latentDimensions)(encoder)
    latentSpace1 = layers.Flatten()(latentSpace)
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    decoder = layers.Dense(64, activation="relu")(decoder)
    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)
    decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoder)
    outputs = layers.Dense(handDimensions, activation="linear")(decoder)  
    
    error = layers.Subtract()([inputs, outputs])

    # return keras.Model([inputs,latentSpace], [outputs, latentSpace, error])        
    return keras.Model(inputs, outputs), keras.Model(inputs, latentSpace1), keras.Model(latentSpace, outputs)
    # return [np.empty(2),np.empty(1),np.empty(2)];
    
# def createAutoencoderLSTMCluster(latentDimensions,windowSize, handDimensions):
#     inputs = layers.Input(shape=(windowSize, handDimensions, ))
    
#     encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
#     encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder)
#     encoder = layers.Dense(64, activation="relu")(encoder)
#     encoder = layers.Dense(16, activation="relu")(encoder)
#     latentSpace = layers.Dense(latentDimensions)(encoder)
#     latentSpace1 = layers.Flatten()(latentSpace)
#     clusteringLayer = TSLayer(20,)(latentSpace)
#     decoder = layers.Dense(16, activation="relu")(latentSpace)
#     decoder = layers.Dense(64, activation="relu")(decoder)
#     decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)
#     decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoder)
#     outputs = layers.Dense(handDimensions, activation="linear")(decoder)  
    
#     error = layers.Subtract()([inputs, outputs])

#     # return keras.Model([inputs,latentSpace], [outputs, latentSpace, error])        
#     return keras.Model(inputs, [outputs,clusteringLayer]), keras.Model(inputs, latentSpace1), keras.Model(latentSpace, outputs)
#     # return [np.empty(2),np.empty(1),np.empty(2)];

    
def createAutoencoderLSTMWORKS(latentDimensions,windowSize, handDimensions):
    inputs = layers.Input(shape=(windowSize, handDimensions))
    
    encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder)
    encoder = layers.Dense(64, activation="relu")(encoder)
    encoder = layers.Dense(16, activation="relu")(encoder)
    latentSpace = layers.Dense(latentDimensions)(encoder)
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    decoder = layers.Dense(64, activation="relu")(decoder)
    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)
    decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoder)
    outputs = layers.Dense(handDimensions, activation="linear")(decoder)
    
    error = layers.Subtract()([inputs, outputs])
        
    return keras.Model(inputs, outputs)


def createLatentToLatent(latentEncDimensions,latentDecDimensions):

    inputs = layers.Input(shape=(latentEncDimensions,))
    
    # translationNetwork = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    # translationNetwork = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(translationNetwork)
    # translationNetwork = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(translationNetwork)
    translationNetwork = layers.Dense((latentEncDimensions+latentDecDimensions)/2,activation="linear")(inputs)
    translationNetwork = layers.Dense((latentEncDimensions+latentDecDimensions),activation="linear")(inputs)
    translationNetwork = layers.Dense((latentEncDimensions+latentDecDimensions)/2,activation="linear")(translationNetwork)
    #translationNetwork = layers.Dense(latentDecDimensions)(translationNetwork)
    outputs = layers.Dense(latentDecDimensions)(translationNetwork)
    
    return keras.Model([inputs],[outputs])

def createLatentToLatentOld(windowSize,latentEncDimensions,latentDecDimensions):

    inputs = layers.Input(shape=(windowSize, latentEncDimensions))
    
    # translationNetwork = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    # translationNetwork = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(translationNetwork)
    # translationNetwork = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(translationNetwork)
    translationNetwork = layers.Dense(latentEncDimensions*16,activation="relu")(inputs)
    translationNetwork = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(translationNetwork)
    translationNetwork = layers.Dense(latentDecDimensions*16,activation="relu")(translationNetwork)
    #translationNetwork = layers.Dense(latentDecDimensions)(translationNetwork)
    outputs = layers.Dense(latentDecDimensions)(translationNetwork)
    
    return keras.Model([inputs],[outputs])


############################################################################################
############################################################################################
############################################################################################
#EXPERIMENTAL
############################################################################################
############################################################################################
############################################################################################
def sample_normal(latent_dim, batch_size, window_size=None):
    shape = (batch_size, latent_dim) if window_size is None else (batch_size, window_size, latent_dim)
    return np.random.normal(size=shape) #set to normal distribution

def create_discriminator(latent_dim):
    input_layer = layers.Input(shape=(latent_dim,))
    disc = layers.Dense(128)(input_layer)
    disc = layers.ELU()(disc)
    disc = layers.Dense(64)(disc)
    disc = layers.ELU()(disc)
    disc = layers.Dense(1, activation="sigmoid")(disc)
    
    model = keras.Model(input_layer, disc)
    return model

def create_generator(latentFrom, latentTo):
    input_layer = layers.Input(shape=(latentFrom,))
    gen = layers.Dense(128)(input_layer)
    gen = layers.ELU()(gen)
    gen = layers.Dense(128)(gen)
    gen = layers.ELU()(gen)
    gen = layers.Dense(latentTo, activation = 'linear')(gen)
    
    model = keras.Model(input_layer, gen)
    return model

def create_discriminator(latent_dim, class_num = 1):
    input_layer = layers.Input(shape=(latent_dim,))
    disc = layers.Dense(256, activation = 'relu')(input_layer)
    disc = layers.Dense(128, activation = 'relu')(disc)
    disc = layers.Dense(64, activation = 'relu')(disc)
    disc = layers.Dense(1, activation='sigmoid')(disc)
    
    model = keras.Model(input_layer, disc)
    return model



############################################################################################
############################################################################################
############################################################################################
#DEPRACATED
############################################################################################
############################################################################################
############################################################################################

def custom_mse_weighted(y_true, y_pred):
    # emgMSE = K.abs((y_true-y_pred)/y_true)
    # mses=K.mean(emgMSE,axis=1)*100
    # return K.mean(mses)
    return K.mean(K.square(y_true - y_pred), axis=-1)

def customDoubleLoss(y_true,y_pred):
    return K.abs(y_pred[0] - y_true[0]) + K.abs(y_pred[1] - y_true[1]);

def customVAEloss(y_true,y_pred):
    reconstructionLoss = K.mean(K.square(y_true - y_pred), axis=-1)

def createLSTMNetwork(windowSize, inputDimensions, outputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions, ))
    
    encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder)

    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder)
    decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoder)
    outputs = layers.Dense(outputDimensions, activation="linear")(decoder)  

    # return keras.Model([inputs,latentSpace], [outputs, latentSpace, error])        
    return keras.Model(inputs, outputs)
    # return [np.empty(2),np.empty(1),np.empty(2)];
    
def createCNNFFTNetwork(windowSize, inputDimensions, outputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions, ))
    
    encoder1 = layers.Conv1D(4, (3), activation='relu', padding='same')(inputs)
    encoder1 = layers.MaxPooling1D((2), padding='same')(encoder1)
    encoder1 = layers.Conv1D(2, (3), activation='relu', padding='same')(encoder1)
    encoder1 = layers.MaxPooling1D((2), padding='same')(encoder1)
    
    encoder1 = layers.Dense(64, activation="relu")(encoder1)
    encoder1 = layers.Dense(16, activation="relu")(encoder1)
    encoder1 = layers.concatenate([encoder1,encoder1,encoder1,encoder1], axis=1)    

    inputs2 = layers.Input(shape=(windowSize, inputDimensions, ))
    
    encoder2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs2)
    encoder2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder2)
    encoder2 = layers.Dense(16, activation="relu")(encoder2)
    
    latentSpace = layers.concatenate([encoder1,encoder2], axis=2)    
    latentSpace = layers.Dense(4)(latentSpace)
    
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    decoder = layers.Dense(64, activation="relu")(decoder)
    decoder = layers.Bidirectional(layers.LSTM(4, return_sequences=True))(decoder)
    #decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)

    # x = layers.Conv1D(2, (3), activation='relu', padding='same')(x)
    # x = layers.UpSampling1D((2))(x)
    # x = layers.Conv1D(4, (3), activation='relu', padding='same')(x)
    # x = layers.UpSampling1D((2))(x)
    outputs = decoder
    
    # inputs2 = layers.Input(shape=(windowSize, inputDimensions, ))
    
    # encoder2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs2)
    # encoder2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder2)

    # decoder2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(encoder2)
    # outputs2 = layers.Dense(outputDimensions, activation="linear")(decoder2)  

    #outputs = layers.Conv1D(1, (3), activation='sigmoid', padding='same')(x)     
    return keras.Model([inputs,inputs2], [outputs])
    # return [np.empty(2),np.empty(1),np.empty(2)];

def createAutoencoderRepresentation(latentDimensions,windowSize, handDimensions):
    inputs = layers.Input(shape=(windowSize, handDimensions, ))
    
    encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder)
    encoder = layers.Dense(64, activation="relu")(encoder)
    encoder = layers.Dense(16, activation="relu")(encoder)
    latentSpace = layers.Dense(latentDimensions)(encoder)
    
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    decoder = layers.Dense(64, activation="relu")(decoder)
    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)
    decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoder)
    outputs = layers.Dense(handDimensions, activation="linear")(decoder)
    
    representation = layers.Dense(64, activation="relu")(latentSpace)
    representation = layers.Dense(16, activation="relu")(representation)
    representation = layers.Dense(latentDimensions, activation="relu")(representation)
    
    
    error = layers.Subtract()([inputs, outputs])

    # return keras.Model([inputs,latentSpace], [outputs, latentSpace, error])        
    return keras.Model(inputs, [outputs,representation])#, keras.Model(inputs, latentSpace), keras.Model(latentSpace, outputs)
    # return [np.empty(2),np.empty(1),np.empty(2)];
    
def createAutoencoderLSTMDoubleTraining(latentDimensions,windowSize, handDimensions, emgDimensions):
    inputsH = layers.Input(shape=(windowSize, handDimensions, )) 
    encoderH = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputsH)
    encoderH = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoderH)
    encoderH = layers.Dense(64, activation="relu")(encoderH)
    encoderH = layers.Dense(16, activation="relu")(encoderH)
    latentSpaceH = layers.Dense(latentDimensions,activation="linear")(encoderH)
    decoderH = layers.Dense(16, activation="relu")(latentSpaceH)
    decoderH = layers.Dense(64, activation="relu")(decoderH)
    decoderH = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoderH)
    decoderH = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoderH)
    outputsH = layers.Dense(handDimensions, activation="linear", name = "handOutput")(decoderH)

    # activity_regularizer = regularizers.l1(1e-2)

    inputsE = layers.Input(shape=(windowSize, emgDimensions, )) 
    encoderE = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputsE)
    encoderE = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoderE)
    encoderE = layers.Dense(64, activation="relu")(encoderE)
    encoderE = layers.Dense(16, activation="relu")(encoderE)
    latentSpaceE = layers.Dense(2,activation="linear")(encoderE)
    decoderE = layers.Dense(16, activation="relu")(latentSpaceE)
    decoderE = layers.Dense(64, activation="relu")(decoderE)
    decoderE = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoderE)
    decoderE = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoderE)
    outputsE = layers.Dense(emgDimensions, activation="linear", name = "emgOutput")(decoderE)
        
    latentSpaceError = layers.Subtract(name = "latentSpaceDiff")([latentSpaceH, latentSpaceE])
    
    return keras.Model([inputsH,inputsE], [outputsH,outputsE,latentSpaceError]), keras.Model(inputsH, latentSpaceH),keras.Model(latentSpaceH, outputsH), keras.Model(inputsE, latentSpaceE), keras.Model(latentSpaceE,outputsE)

def createAutoencoderLSTMRepeat(latentDimensions,windowSize, handDimensions):
    inputs = layers.Input(shape=(windowSize, handDimensions, ))
    
    x = layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(64, activation='relu', return_sequences=False))(x)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Dense(16, activation="linear")(x)
    latentSpace = layers.RepeatVector(windowSize)(x)
    # x = layers.Dense(16, activation="linear")(latentSpace)
    # x = layers.Dense(64, activation="linear")(x)
    x = layers.Bidirectional(layers.LSTM(64, activation='relu', return_sequences=True))(latentSpace)
    x = layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True))(x)
    outputs = layers.Dense(handDimensions, activation="linear")(x)
    
    error = layers.Subtract()([inputs, outputs])
        
    return keras.Model(inputs, [outputs, latentSpace, error])

def createAutoencoderCNN1(latentDimensions,windowSize, emgDimensions):
    inputs = layers.Input(shape=(windowSize, emgDimensions))
    
    # x = layers.Conv2D(filters=64, kernel_size=7, padding="same", strides=2, activation="relu") (inputs)
    # x = layers.Conv2D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu") (x)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Dense(16, activation="linear")(x)
    # latentSpace = layers.Dense(2)(x)
    # x = layers.Dense(16, activation="linear")(latentSpace)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Conv2DTranspose(filters=32, kernel_size=7, padding="same", strides=1, activation="relu") (x)
    # x = layers.Conv2DTranspose(filters=64, kernel_size=7, padding="same", strides=2, activation="relu") (x)
    # outputs = layers.Dense(emgDimensions, activation="linear")(x)
    
    x = layers.Conv1D(4, (3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D((2), padding='same')(x)
    x = layers.Conv1D(2, (3), activation='relu', padding='same')(x)
    x = layers.MaxPooling1D((2), padding='same')(x)
    
    x = layers.Dense(64, activation="linear")(x)
    x = layers.Dense(16, activation="linear")(x)
    latentSpace = layers.Dense(2)(x)
    x = layers.Dense(16, activation="linear")(latentSpace)
    x = layers.Dense(64, activation="linear")(x)

    x = layers.Conv1D(2, (3), activation='relu', padding='same')(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Conv1D(4, (3), activation='relu', padding='same')(x)
    x = layers.UpSampling1D((2))(x)
    outputs = layers.Conv1D(8, (3), activation='sigmoid', padding='same')(x)
    
    # error = layers.Subtract()([inputs, outputs])
    # error = inputs
        
    return keras.Model(inputs, [outputs, latentSpace])#, error])


def createAutoencoderCNN2(latentDimensions,windowSize, emgDimensions):
    inputs = layers.Input(shape=(windowSize, emgDimensions,1))
    
    # x = layers.Conv2D(filters=64, kernel_size=7, padding="same", strides=2, activation="relu") (inputs)
    # x = layers.Conv2D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu") (x)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Dense(16, activation="linear")(x)
    # latentSpace = layers.Dense(2)(x)
    # x = layers.Dense(16, activation="linear")(latentSpace)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Conv2DTranspose(filters=32, kernel_size=7, padding="same", strides=1, activation="relu") (x)
    # x = layers.Conv2DTranspose(filters=64, kernel_size=7, padding="same", strides=2, activation="relu") (x)
    # outputs = layers.Dense(emgDimensions, activation="linear")(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Reshape((40,32))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    latentSpace = layers.Dense(latentDimensions)(x)
    x = layers.Dense(16, activation="relu")(latentSpace)
    x = layers.Dense(64, activation="relu")(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Reshape((20,2,256))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    
    # error = layers.Subtract()([inputs, outputs])
    # error = inputs
        
    return keras.Model(inputs, outputs), keras.Model(inputs,latentSpace)

def createAutoencoderCNN3(latentDimensions,windowSize, emgDimensions):
    inputs = layers.Input(shape=(windowSize-1,windowSize-1, emgDimensions))
    
    # x = layers.Conv2D(filters=64, kernel_size=7, padding="same", strides=2, activation="relu") (inputs)
    # x = layers.Conv2D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu") (x)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Dense(16, activation="linear")(x)
    # latentSpace = layers.Dense(2)(x)
    # x = layers.Dense(16, activation="linear")(latentSpace)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Conv2DTranspose(filters=32, kernel_size=7, padding="same", strides=1, activation="relu") (x)
    # x = layers.Conv2DTranspose(filters=64, kernel_size=7, padding="same", strides=2, activation="relu") (x)
    # outputs = layers.Dense(emgDimensions, activation="linear")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.Flatten()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    latentSpace = layers.Dense(latentDimensions)(x)
    x = layers.Dense(16, activation="relu")(latentSpace)
    x = layers.Dense(64, activation="relu")(x)

    # x = layers.Reshape((25,25,64))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(8, (3, 3), activation='linear', padding='same')(x)
    
    # error = layers.Subtract()([inputs, outputs])
    # error = inputs
        
    return keras.Model(inputs, outputs), keras.Model(inputs,latentSpace)

def createAutoencoderDeepOnly(latentDimensions,windowSize,inputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    latentSpace = layers.Dense(4)(x)
    x = layers.Dense(16, activation="relu")(latentSpace)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(windowSize*inputDimensions, activation="relu")(x)
    outputs = layers.Reshape((windowSize,inputDimensions))(x)
    
    return keras.Model(inputs, outputs), keras.Model(inputs, latentSpace)
    
def createAutoencoderCNNStraight(latentDimensions,windowSize, emgDimensions, handDimensions):
    #FOR WAVELETS
    #inputs = layers.Input(shape=(windowSize-1,windowSize-1, emgDimensions))
    inputs = layers.Input(shape=(windowSize, emgDimensions, 1))
    
    # x = layers.Conv2D(filters=64, kernel_size=7, padding="same", strides=2, activation="relu") (inputs)
    # x = layers.Conv2D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu") (x)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Dense(16, activation="linear")(x)
    # latentSpace = layers.Dense(2)(x)
    # x = layers.Dense(16, activation="linear")(latentSpace)
    # x = layers.Dense(64, activation="linear")(x)
    # x = layers.Conv2DTranspose(filters=32, kernel_size=7, padding="same", strides=1, activation="relu") (x)
    # x = layers.Conv2DTranspose(filters=64, kernel_size=7, padding="same", strides=2, activation="relu") (x)
    # outputs = layers.Dense(emgDimensions, activation="linear")(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    # x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format = "channels_last")(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # x = layers.MaxPooling2D((4, 4), padding='same')(x)
    x = layers.Flatten()(x)
    
    # latentSpace = layers.Dense(latentDimensions*4, activation="relu")(x)
    # latentSpace = layers.Dense(16, activation="relu")(latentSpace)
    latentSpace = layers.Dense(latentDimensions)(x)
    # latentSpace = layers.Dense(16, activation="relu")(latentSpace)
    # latentSpace = layers.Dense(latentDimensions*4, activation="relu")(latentSpace)

    x = layers.Dense(40*4*32)(latentSpace)
    x = layers.Reshape((40,4,32))(x)
    # x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    # x = layers.UpSampling2D((4, 4))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same',data_format = "channels_last")(x)
    x = layers.UpSampling2D((2, 2))(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    # x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2DTranspose(1, (3, 3), activation='linear', padding='same')(x)
    
    # error = layers.Subtract()([inputs, outputs])
    # error = inputs
        
    return keras.Model(inputs, outputs), keras.Model(inputs, latentSpace)

def createAutoencoderCNNStraightToBinaryHand(latentDimensions,windowSize, emgDimensions, handDimensions):
    inputs = layers.Input(shape=(windowSize, emgDimensions, 1))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format = "channels_last")(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)

    # x = layers.Reshape((80,2*32))(x)
    latentSpace = layers.Dense(100, activation="relu")(x)
    outputs = layers.Dense(handDimensions, activation="sigmoid")(latentSpace)
        
    return keras.Model(inputs, outputs), keras.Model(inputs, latentSpace)

def createAutoencoderCNNStraightToHand(latentDimensions,windowSize, emgDimensions, handDimensions):
    inputs = layers.Input(shape=(windowSize-1,windowSize-1, emgDimensions))
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((4, 4), padding='same')(x)
    x = layers.Conv2D(16, (4, 4), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((4, 4), padding='same')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    latentSpace = layers.Dense(latentDimensions)(x)
    x = layers.Dense(16, activation="relu")(latentSpace)
    x = layers.Dense(32, activation="relu")(x)

    x = layers.RepeatVector(windowSize)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    outputs = layers.Dense(handDimensions, activation="linear")(x)  
        
    return keras.Model(inputs, outputs), keras.Model(inputs, latentSpace)

def createDiscriminator(windowSize,latentDimensions):
    inputs = layers.Input(shape=(latentDimensions,))
    disc = layers.Dense(256, activation = 'relu')(inputs)
    disc = layers.Dense(128, activation = 'relu')(disc)
    disc = layers.Dense(64, activation = 'relu')(disc)
    disc = layers.Dense(1, activation='sigmoid')(disc)
    
    model = keras.Model(inputs, disc)
    return model

def createRetrainingDiscriminator(windowSize,inputDimensions,encoder,decoder,discriminator):
    discriminator.trainable = False
    inputs = layers.Input(shape=(windowSize,inputDimensions))
    ae = encoder(inputs)
    disc = discriminator(ae)
    ae = decoder(ae)
    discriminator.trainable = False
    
    return keras.Model([inputs],[ae,disc]),encoder
      

def createVAESingle(latentDimensions,windowSize, inputDimensions):
    inputs = layers.Input(shape=(windowSize, inputDimensions, )) 
    encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    encoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder)
    encoder = layers.Dense(64, activation="relu")(encoder)
    encoder = layers.Dense(16, activation="relu")(encoder)
    
    z_mu = layers.Dense(latentDimensions)(encoder)
    z_log_var = layers.Dense(latentDimensions)(encoder)
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=K.shape(z_mean),mean=0.0, stddev=1.0)
        gDistribution = z_mean + K.exp(z_log_sigma) * epsilon
        return gDistribution
    
    latentSpace = layers.Lambda(sampling)([z_mu, z_log_var])
    
    # latentSpace = layers.Dense(2,activation="linear")(encoder_output)
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    decoder = layers.Dense(64, activation="relu")(decoder)
    decoder = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoder)
    decoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoder)
    outputs = layers.Dense(inputDimensions, activation="linear", name = "handOutput")(decoder)
    
    return keras.Model(inputs, outputs=[outputs,latentSpace])

def createVAEDouble(latentDimensions,windowSize, handDimensions, emgDimensions):
    inputsH = layers.Input(shape=(windowSize, handDimensions, )) 
    encoderH = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputsH)
    encoderH = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoderH)
    encoderH = layers.Dense(64, activation="relu")(encoderH)
    encoderH = layers.Dense(16, activation="relu")(encoderH)
    
    inputsE = layers.Input(shape=(windowSize, emgDimensions, )) 
    encoderE = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputsE)
    encoderE = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoderE)
    encoderE = layers.Dense(64, activation="relu")(encoderE)
    encoderE = layers.Dense(16, activation="relu")(encoderE)
    
    z_muH = layers.Dense(latentDimensions)(encoderH)
    z_log_varH = layers.Dense(latentDimensions)(encoderH)
    z_muE = layers.Dense(latentDimensions)(encoderE)
    z_log_varE = layers.Dense(latentDimensions)(encoderE)
    
    jointMu = layers.Average()([z_muH,z_muE])
    jointLogVar = layers.Average()([z_log_varH,z_log_varE])
    
    z_muH, z_log_varH = KLDivergenceLayer()([z_muH, z_log_varH])
    z_muE, z_log_varE = KLDivergenceLayer()([z_muE, z_log_varE])

    def samplingH(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=K.shape(z_mean),mean=0.0, stddev=0.5)
        gDistribution = z_mean + K.exp(z_log_sigma) * epsilon
        return gDistribution
    def samplingE(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=K.shape(z_mean),mean=0.0, stddev=0.1)
        gDistribution = z_mean + K.exp(z_log_sigma) * epsilon
        return gDistribution
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=K.shape(z_mean),mean=0.0, stddev=0.1)
        gDistribution = z_mean + K.exp(z_log_sigma) * epsilon
        return gDistribution    
    latentSpaceH = layers.Lambda(samplingH)([jointMu, jointLogVar])
    latentSpaceE = layers.Lambda(samplingE)([jointMu, jointLogVar])
    
    latentSpaceError = layers.Subtract(name = "latentSpaceDiff")([latentSpaceH, latentSpaceE])
    
    # latentSpaceH = layers.Dense(2,activation="linear")(encoder_outputH)
    decoderH = layers.Dense(16, activation="relu")(latentSpaceH)
    decoderH = layers.Dense(64, activation="relu")(decoderH)
    decoderH = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoderH)
    decoderH = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoderH)
    outputsH = layers.Dense(handDimensions, activation="linear", name = "handOutput")(decoderH)

    decoderE = layers.Dense(16, activation="relu")(latentSpaceE)
    decoderE = layers.Dense(64, activation="relu")(decoderE)
    decoderE = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoderE)
    decoderE = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoderE)
    outputsE = layers.Dense(emgDimensions, activation="linear", name = "emgOutput")(decoderE)
    # activity_regularizer = regularizers.l1(1e-2)
        

    # latentSpaceE = layers.Dense(2,activation="linear")(encoder_outputE)        
    #latentSpaceError = layers.Subtract(name = "latentSpaceDiff")([latentSpaceE, latentSpaceE])
    
    return keras.Model(inputs=[inputsH,inputsE], outputs=[outputsH,outputsE,latentSpaceError,latentSpaceH,latentSpaceE])#,keras.Model([inputsH], [latentSpaceH]),keras.Model(latentSpaceH, outputsH), keras.Model(inputsE, latentSpaceE), keras.Model(latentSpaceE,outputsE)
  #keras.Model(inputs=[inputsH,inputsE], outputs=[outputsH,outputsE,latentSpaceH,latentSpaceE]),

def createLossInjectionModel(windowSize,inputDimensions, latentDimensions, encoderComponent, decoderComponent):
    
    inputs = layers.Input(shape=(windowSize, inputDimensions, ))
    latentInputs = layers.Input(shape=(windowSize, latentDimensions, ))
    encoder = encoderComponent(inputs)
    translationIdentity,_ = MinSquaredLayer()([encoder, latentInputs])
    drop = layers.Dropout(0.1)(translationIdentity)
    decoder = decoderComponent(drop)
    
    return keras.Model(inputs=[inputs,latentInputs],outputs=[decoder]), keras.Model(inputs,encoder), decoder

def createNetworkInjectionModel(windowSize,inputDimensions, latentDimensions, encoderComponent, decoderComponent, decoderComponentH):
    # encoderComponent.trainable = False
    # decoderComponent.trainable = False

    inputs = layers.Input(shape=(windowSize, inputDimensions, ))
    latentInputs = layers.Input(shape=(windowSize, latentDimensions, ))
    encoder = encoderComponent(inputs)
    
    translationNetwork = layers.Dense(latentDimensions*16,activation="relu")(encoder)
    translationNetwork = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(translationNetwork)
    translationNetwork = layers.Dense(latentDimensions*16,activation="relu")(translationNetwork)
    translationNetwork = layers.Dense(latentDimensions)(translationNetwork)
    
    translationIdentity,_ = MinSquaredLayer(name="injectedLoss")([translationNetwork, latentInputs])
    
    decoder = decoderComponent(translationNetwork)
    decoder = layers.Layer(name = "emgOutput")(decoder)
    
    decoderH = decoderComponentH(translationNetwork)
    decoderH = layers.Layer(name = "handOutput")(decoderH)
    
    return keras.Model(inputs=[inputs,latentInputs],outputs=[decoder]), keras.Model(inputs,translationNetwork), keras.Model(translationNetwork,decoder)


def createAutoencoderConvLSTM(latentDimensions,windowSize, handDimensions):
    inputs = layers.Input(shape=(1,1,windowSize, handDimensions, ))
    
    encoder = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(inputs)
    encoder = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(encoder)
    encoder = layers.Dense(64, activation="relu")(encoder)
    encoder = layers.Dense(16, activation="relu")(encoder)
    latentSpace = layers.Dense(latentDimensions)(encoder)
    
    decoder = layers.Dense(16, activation="relu")(latentSpace)
    decoder = layers.Dense(64, activation="relu")(decoder)
    decoder = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(decoder)
    decoder = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(decoder)
    outputs = layers.Dense(handDimensions, activation="linear")(decoder)  
    
    error = layers.Subtract()([inputs, outputs])

    return keras.Model([inputs], [outputs, latentSpace, error])        
    # return keras.Model(inputs, outputs), keras.Model(inputs, latentSpace), keras.Model(latentSpace, outputs)
    # return [np.empty(2),np.empty(1),np.empty(2)];
    
class KLDivergenceLayer(layers.Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs
    
class MinSquaredLayer(layers.Layer):

    """ Identity transform layer that adds min squared to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MinSquaredLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        
        latentSpaceO, latentSpaceA = inputs
        mape = K.abs((latentSpaceO-latentSpaceA)/latentSpaceO)
        mape=K.mean(mape,axis=1)*1000
        # squaredDiff = K.square(latentSpaceO-latentSpaceA)
        # squaredDiff = K.mean(squaredDiff)
        self.add_loss(mape, inputs=inputs)
        return inputs
def createAutoencoderLSTMDavid(latent_dim, window_size, input_dim):
    input_layer = layers.Input(shape=(window_size, input_dim))
    
    code = layers.TimeDistributed(layers.Dense(64, activation='linear'))(input_layer)
    code = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input_layer)
    code = layers.BatchNormalization()(code)
    code = layers.ELU()(code)
    code = layers.Bidirectional(layers.LSTM(64))(code)
    code = layers.BatchNormalization()(code)
    code = layers.ELU()(code)
    
    latent_repr = layers.Dense(64)(code)
    latent_repr = layers.BatchNormalization()(latent_repr)
    latent_repr = layers.PReLU()(latent_repr)
    latent_repr = layers.Dense(latent_dim, activation='linear')(latent_repr)
    
    decode = layers.RepeatVector(1)(latent_repr)
    decode = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decode)
    decode = layers.ELU()(decode)
    decode = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decode)
    decode = layers.ELU()(decode)
    decode = layers.TimeDistributed(layers.Dense(64))(decode)
    decode = layers.ELU()(decode)
    decode = layers.TimeDistributed(layers.Dense(input_dim, activation='linear'))(decode)
    
    error = layers.Subtract()([input_layer, decode])
        
    return keras.Model(input_layer, [decode, latent_repr, error])

def createAutoencoderFromComponentsDense(windowSize,inputDimensions, latentDimensions, encoderComponent, decoderComponent):
    
    encoderComponent.trainable = False
    decoderComponent.trainable = False
    encoderComponent.input_shape
    inputs = layers.Input(shape=(windowSize, inputDimensions, ))
    encoder = encoderComponent(inputs)
    translationSpace = layers.Dense(latentDimensions)(encoder)
    decoder = decoderComponent(translationSpace)
    
    return keras.Model(inputs,decoder)

def createAutoencoderFromComponentsNoDense(windowSize,inputDimensions, latentDimensions, encoderComponent, decoderComponent):
    
    encoderComponent.trainable = False
    decoderComponent.trainable = False
    encoderComponent.input_shape
    inputs = layers.Input(shape=(windowSize, inputDimensions, ))
    encoder = encoderComponent(inputs)
    decoder = decoderComponent(encoder)
    
    return keras.Model(inputs,decoder)

def createVAE3(latentDimensions,windowSize, handDimensions, emgDimensions):
    inputsH = layers.Input(shape=(windowSize, handDimensions, )) 
    encoderH = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputsH)
    encoderH = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoderH)
    encoderH = layers.Dense(64, activation="relu")(encoderH)
    encoderH = layers.Dense(16, activation="relu")(encoderH)
    
    z_muH = layers.Dense(latentDimensions)(encoderH)
    z_log_varH = layers.Dense(latentDimensions)(encoderH)
    z_muH, z_log_varH = KLDivergenceLayer()([z_muH, z_log_varH])
    z_sigmaH = layers.Lambda(lambda t: K.exp(.5*t))(z_log_varH)
    # epsH = layers.InputLayer(input_tensor=K.random_normal(stddev=1,shape=(K.shape(inputsH)[0],K.shape(inputsH)[1], latentDimensions)))
    epsH = layers.Input(shape=(latentDimensions, )) 
    print(epsH.shape)
    print(z_sigmaH.shape)
    z_epsH = layers.Multiply()([z_sigmaH, epsH])
    latentSpaceH = layers.Add()([z_muH, z_epsH])

    # latentSpaceH = layers.Dense(2,activation="linear")(encoder_outputH)
    decoderH = layers.Dense(16, activation="relu")(latentSpaceH)
    decoderH = layers.Dense(64, activation="relu")(decoderH)
    decoderH = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoderH)
    decoderH = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoderH)
    outputsH = layers.Dense(handDimensions, activation="linear", name = "handOutput")(decoderH)

    # activity_regularizer = regularizers.l1(1e-2)

    inputsE = layers.Input(shape=(windowSize, emgDimensions, )) 
    encoderE = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputsE)
    encoderE = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoderE)
    encoderE = layers.Dense(64, activation="relu")(encoderE)
    encoderE = layers.Dense(16, activation="relu")(encoderE)
    
    z_muE = layers.Dense(latentDimensions)(encoderE)
    z_log_varE = layers.Dense(latentDimensions)(encoderE)
    
    z_muE, z_log_varE = KLDivergenceLayer()([z_muE, z_log_varE])
    z_sigmaE = layers.Lambda(lambda t: K.exp(.5*t))(z_log_varE)
    
    epsE = layers.Input(shape=(latentDimensions, )) 
    z_epsE = layers.Multiply()([z_sigmaE, z_sigmaE])
    latentSpaceE = layers.Add()([z_muE, z_epsE])

    # latentSpaceE = layers.Dense(2,activation="linear")(encoder_outputE)
    decoderE = layers.Dense(16, activation="relu")(latentSpaceE)
    decoderE = layers.Dense(64, activation="relu")(decoderE)
    decoderE = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoderE)
    decoderE = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoderE)
    outputsE = layers.Dense(emgDimensions, activation="linear", name = "emgOutput")(decoderE)
        
    #latentSpaceError = layers.Subtract(name = "latentSpaceDiff")([latentSpaceH, latentSpaceE])
    
    return keras.Model(inputs=[inputsH,inputsE,epsH,epsE], outputs=[outputsH,outputsE,latentSpaceE,latentSpaceH])#, keras.Model(inputsH, latentSpaceH)#,keras.Model(latentSpaceH, outputsH), keras.Model(inputsE, latentSpaceE), keras.Model(latentSpaceE,outputsE)
    # return keras.Model(inputs=[inputsH,inputsE,epsH], outputs=[outputsH,outputsE]), keras.Model(inputsH, latentSpaceH),keras.Model(latentSpaceH, outputsH), keras.Model(inputsE, latentSpaceE), keras.Model(latentSpaceE,outputsE)

def createVAE2(latentDimensions,windowSize, handDimensions, emgDimensions):
    inputsH = layers.Input(shape=(windowSize, handDimensions, )) 
    encoderH = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputsH)
    encoderH = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoderH)
    encoderH = layers.Dense(64, activation="relu")(encoderH)
    encoderH = layers.Dense(16, activation="relu")(encoderH)
    
    shape_before_flattenH = K.int_shape(encoderH)[1:]
    encoder_flattenH = layers.Flatten()(encoderH)
    
    encoder_muH = layers.Dense(units=latentDimensions, name="encoder_muH")(encoder_flattenH)
    encoder_log_varianceH = layers.Dense(units=latentDimensions, name="encoder_log_varianceH")(encoder_flattenH)
    
    encoder_mu_log_variance_modelH = models.Model(inputsH, (encoder_muH, encoder_log_varianceH), name="encoder_mu_log_variance_modelH")
    
    def sampling(mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + K.exp(log_variance/2) * epsilon
        return random_sample
    
    latentSpaceH = layers.Lambda(sampling, name="encoderH_output")([encoder_muH, encoder_log_varianceH])

    decoder_dense_layer1H = layers.Dense(units=np.prod(shape_before_flattenH), name="decoder_dense_1")(latentSpaceH)
    decoder_reshapeH = layers.Reshape(target_shape=shape_before_flattenH)(decoder_dense_layer1H)

    # latentSpaceH = layers.Dense(2,activation="linear")(encoder_outputH)
    decoderH = layers.Dense(16, activation="relu")(decoder_reshapeH)
    decoderH = layers.Dense(64, activation="relu")(decoderH)
    decoderH = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoderH)
    decoderH = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoderH)
    outputsH = layers.Dense(handDimensions, activation="linear", name = "handOutput")(decoderH)

    # activity_regularizer = regularizers.l1(1e-2)

    inputsE = layers.Input(shape=(windowSize, emgDimensions, )) 
    encoderE = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputsE)
    encoderE = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoderE)
    encoderE = layers.Dense(64, activation="relu")(encoderE)
    encoderE = layers.Dense(16, activation="relu")(encoderE)
    
    shape_before_flattenE = K.int_shape(encoderE)[1:]
    encoder_flattenE = layers.Flatten()(encoderE)
    
    encoder_muE = layers.Dense(units=latentDimensions, name="encoder_muE")(encoder_flattenE)
    encoder_log_varianceE = layers.Dense(units=latentDimensions, name="encoder_log_varianceE")(encoder_flattenE)
    
    encoder_mu_log_variance_modelE = models.Model(inputsE, (encoder_muE, encoder_log_varianceE), name="encoder_mu_log_variance_modelE")
    
    def sampling(mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + K.exp(log_variance/2) * epsilon
        return random_sample
    
    latentSpaceE = layers.Lambda(sampling, name="encoderE_output")([encoder_muE, encoder_log_varianceE])

    decoder_dense_layer1E = layers.Dense(units=np.prod(shape_before_flattenE), name="decoder_dense_1E")(latentSpaceE)
    decoder_reshapeE = layers.Reshape(target_shape=shape_before_flattenE)(decoder_dense_layer1E)
    
    # latentSpaceE = layers.Dense(2,activation="linear")(encoderE)
    decoderE = layers.Dense(16, activation="relu")(decoder_reshapeE)
    decoderE = layers.Dense(64, activation="relu")(decoderE)
    decoderE = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(decoderE)
    decoderE = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(decoderE)
    outputsE = layers.Dense(emgDimensions, activation="linear", name = "emgOutput")(decoderE)
        
    #latentSpaceError = layers.Subtract(name = "latentSpaceDiff")([latentSpaceH, latentSpaceE])
    
    return keras.Model([inputsH,inputsE], [outputsH,outputsE]), encoder_muH, encoder_log_varianceH, encoder_muE, encoder_log_varianceE, keras.Model(inputsH, latentSpaceH),keras.Model(latentSpaceH, outputsH), keras.Model(inputsE, latentSpaceE), keras.Model(latentSpaceE,outputsE)

def lossVAE(encoder_mu, encoder_log_variance):
    axisList = [1,2]
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=axisList)
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=axisList)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=axisList)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss + kl_loss
        return loss
    return vae_loss