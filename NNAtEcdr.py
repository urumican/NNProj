
# coding: utf-8

# In[ ]:

## CS519 Deep Learning Course Project
## Project: Deep learning for GWAS SNPs prioritization / prediction
## Method: deep auto-encoder
## Libraries: numpy, scipy, scikit-learn, patsy, keras, theano


# In[ ]:

## Command lines for environment setup and report generation

# setup scipy: pip install scipy
# setup scikit-learn: pip install --upgrade scikit-learn (we select the latest version, --upgrade)
# setup patsy: pip install patsy
# setup theano: pip install --upgrade theano
# setup keras: pip install keras

# create PDF report: ipython nbconvert --to pdf GWASTool.ipynb
##


# In[ ]:

## load packages & import libraries
import sys, os
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics #roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


###################################################################
# By import plotROC, call plotAutoEntocer() and plotPCA() to plot #
# corresponding ROC plot.                                         #
###################################################################
import plotROC


# get_ipython().magic(u'matplotlib inline')

pd.set_option('display.max_columns', 500)


# In[ ]:

## Load data (ucsc_metadata + TSS + GERP + eQTL-pvalue)

# Load and Concat training + test datasets
positive = pd.read_csv("data2/rsnps.ucsc_metadata.txt", sep='\t').fillna(0)
unlabeled = pd.read_csv("data2/csnps.ucsc_metadata.txt", sep='\t').fillna(0)

# Add TSS (Transcription Start Site) metadata to above
positiveTSS = pd.read_csv("data2/rsnps.tss_metadata.txt", sep='\t')
unlabeledTSS = pd.read_csv("data2/csnps.tss_metadata.txt", sep='\t')

positivePValue = pd.read_csv("data2/rsnps.pvalue_metadata.txt", sep='\t')
unlabeledPValue = pd.read_csv("data2/csnps.pvalue_metadata.txt", sep='\t')

positiveGerp = pd.read_csv("data2/rsnps_gerp_metadata.txt", sep='\t')
unlabeledGerp = pd.read_csv("data2/csnps_gerp_metadata_34k.txt", sep='\t')

positive = positive.merge(positiveTSS, on='name')
positive = positive.merge(positivePValue, on='name')
positive = positive.merge(positiveGerp, on='name')
unlabeled = unlabeled.merge(unlabeledTSS, on='name')
unlabeled = unlabeled.merge(unlabeledPValue, on='name')
unlabeled = unlabeled.merge(unlabeledGerp, on='name')

# positive.to_csv("rsnp.txt", sep='\t', encoding='utf-8', index=False)

# Add label
positive['label'] = 1
unlabeled['label'] = 0

alldata = pd.concat([positive, unlabeled]) # concat rows
print "Positive cases (%d) + Negative cases (%d) = Total (%d)" %     (len(positive), len(unlabeled), len(alldata))


# In[ ]:

# Set up design-matrix
# UCSC SNP142 Schema Notes: http://ucscbrowser.genap.ca/cgi-bin/hgTables?db=hg19&hgta_group=varRep&hgta_track=snp135&hgta_table=snp135&hgta_doSchema=describe+table+schema

# extract first two alleles from UCSC column 'alleles'
alleleFeature = alldata.alleles.str.split(',').tolist()
alleleFeature = [[] if x is np.nan else x[0:2] for x in alleleFeature]
alleleFeature = pd.DataFrame(alleleFeature, columns="majorAllele minorAllele".split())
alldata = alldata.merge(alleleFeature, left_index=True, right_index=True)

# TFBS: text-to-columns 
tfbsOneHot = alldata['tf_names'].str.get_dummies(sep=',') # 158 TFBS columns
alldata = alldata.merge(tfbsOneHot, left_index=True, right_index=True)

# Create R-formula for the fit
from patsy import dmatrices, dmatrix
formula = 'label ~ C(chrom) + C(majorAllele) + C(minorAllele) +     DHS_SourceCount + num_tfs + phastCons + tssDistance + P_Val + gerp + '
formula += " + ".join(tfbsOneHot.columns[1:])

# create data-matrix
y, X = dmatrices(formula, alldata, return_type="dataframe")
y = np.ravel(y)

print "#y:%d, #x.rows:%d, #x.cols:%d" % (len(y), len(X), len(X.columns))


# In[ ]:

X1 = X.as_matrix(columns=None)


# In[ ]:

# shuffle and split into (1) model building  and (2) model-comparison/holdout sets
holdoutFraction = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=holdoutFraction, random_state=1337)


# In[ ]:

print(X_test.shape)
print(y_test.shape)
#print(y_train.shape)
#print(np.size(y_train))
#print(np.size(X_test))


# # Baseline
# 
# Fully connected layers

# In[ ]:

import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T
import random
# Baseline
from keras.layers import containers
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, AutoEncoder, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


#random.seed(1)
#np.random.seed(1)

nb_epoch = 50
batch_size = 100
nb_labels = 2


X_train_narray = X_train.as_matrix(columns=None) #change the X_train from pandas.DataFrame to numpy.narray
X_test_narray = X_test.as_matrix(columns=None) #change the X_test from pandas.DataFrame to numpy.narray

y_train_narray = np_utils.to_categorical(y_train,nb_labels)
y_test_narray = np_utils.to_categorical(y_test, nb_labels)

model = Sequential()
model.add(Dense(50, input_dim = X_train_narray.shape[1], activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam')
# model.compile(loss='categorical_crossentropy', optimizer='sgd')

# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png')

#earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0)

print('Start training')
# model.fit(X_train_narray, y_train_narray, batch_size=batch_size, nb_epoch=nb_epoch,
#          show_accuracy=True, verbose=True, validation_data=(X_test, y_test_narray), callbacks=[earlyStopping])

model.fit(X_train_narray, y_train_narray, batch_size=batch_size, nb_epoch=nb_epoch, 
          show_accuracy=True, verbose=True, validation_data=(X_test_narray, y_test_narray), shuffle=True)

score = model.evaluate(X_test_narray, y_test_narray, show_accuracy=True, verbose=False)
print('Test accuracy:', score[1])


# In[ ]:

# Normalization for PCA
# X_train_mean = np.mean(X_train_narray, axis=0)
# X_train_std = np.std(X_train_narray, axis=0)
# X_train_norm = (X_train - X_train_mean)/(X_train_std)
X_train_norm = pd.DataFrame(X_train,copy=True)
X_train_norm['tssDistance'] = (X_train_norm['tssDistance'] - X_train_norm['tssDistance'].mean()) / X_train_norm['tssDistance'].std()
X_train_norm_narray = X_train_norm.as_matrix(columns=None)

X_test_norm = X_test
X_test_norm['tssDistance'] = (X_test_norm['tssDistance'] - X_test_norm['tssDistance'].mean()) / X_test_norm['tssDistance'].std()
X_test_norm_narray = X_test_norm.as_matrix(columns=None)



# In[ ]:

from sklearn.decomposition import PCA
import pickle

def PCAMyData(tranData, trainLabel, testData, testLabel):
    ## use following to plot
    pca = PCA(n_components=2)
    pca.fit(tranData)
    pcaResOfTestData = pca.transform(testData)
    
    ## use following for future
    pcaFuture = PCA(n_components=207)
    pcaFuture.fit(tranData)
    pcaResFuture = pcaFuture.transform(testData)

    colors = {0: 'g', 1:'m'}
    markers = {0: '*', 1: '+'}

    for idx in range(testData.shape[0]):
        point = pcaResOfTestData[idx]
        label = testLabel[idx]
        color = colors[label]
        marker = markers[label]
        line = plt.plot(point[0], point[1], color = color, marker = marker, markersize=2)
    ## end
    plt.show()
    
## end

PCAMyData(X_train_narray,y_train,X_test_narray,y_test)
PCAMyData(X_train_norm,y_train,X_test_norm,y_test)


# In[ ]:

# Stacked Autoencoder
# Train the autoencoder
# Source: https://github.com/fchollet/keras/issues/358
from keras.layers import containers
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, AutoEncoder, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

random.seed(3)
np.random.seed(3)

nb_epoch_pretraining = 10
batch_size_pretraining = 500

# Layer-wise pretraining
encoders = []
decoders = []
nb_hidden_layers = [X_train_narray.shape[1], 150, 2]

X_train_tmp = np.copy(X_train_norm_narray)
print('original X_train_tmp SIZE:',X_train_tmp.shape)

dense_layers = []

for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
    print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
    # Create AE and training
    ae = Sequential()
    if n_out >= 100:
        encoder = containers.Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='tanh'), Dropout(0.5)])
    else:
        encoder = containers.Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='tanh')])
    decoder = containers.Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='tanh')])
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    
    sgd = SGD(lr=2, decay=1e-6, momentum=0.0, nesterov=True)
    ae.compile(loss='mse', optimizer='adam')
    ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size_pretraining, nb_epoch=nb_epoch_pretraining, verbose = True, shuffle=True)
    # Store trainined weight and update training data
    encoders.append(ae.layers[0].encoder)
    decoders.append(ae.layers[0].decoder)
    
    X_train_tmp = ae.predict(X_train_tmp)    
    
    print('X_train_tmp SIZE:',X_train_tmp.shape)


##############
    
#End to End Autoencoder training
if len(nb_hidden_layers) > 2:
    full_encoder = containers.Sequential()
    for encoder in encoders:
        full_encoder.add(encoder)

    full_decoder = containers.Sequential()
    for decoder in reversed(decoders):
        full_decoder.add(decoder)

    full_ae = Sequential()
    full_ae.add(AutoEncoder(encoder=full_encoder, decoder=full_decoder, output_reconstruction=False))    
    full_ae.compile(loss='mse', optimizer='adam')

    print "Pretraining of full AE"
    full_ae.fit(X_train_norm_narray, X_train_norm_narray, batch_size=batch_size_pretraining, nb_epoch=nb_epoch_pretraining, verbose = True, shuffle=True)


# # Using pretrained AutoEncoder for Classification
# In principle the pretrained AutoEncoder could be used for classification as in the following code.

# In[ ]:

nb_epoch = 50
batch_size = 100

model = Sequential()
for encoder in encoders:
    model.add(encoder)

model.add(Dense(output_dim=nb_labels, activation='softmax'))

#y_train_cat = np_utils.to_categorical(train_subset_y, nb_labels)
#y_test_cat = np_utils.to_categorical(y_test, nb_labels)

model.compile(loss='categorical_crossentropy', optimizer='Adam')
score = model.evaluate(X_test_norm_narray, y_test_narray, show_accuracy=True, verbose=1)
print('Test score before fine turning:', score[0])
print('Test accuracy before fine turning:', score[1])
model.fit(X_train_norm_narray, y_train_narray, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, validation_data=(X_test_norm_narray, y_test_narray), shuffle=True)
score = model.evaluate(X_test_norm_narray, y_test_narray, show_accuracy=True, verbose=0)
print('Test score after fine turning:', score[0])
print('Test accuracy after fine turning:', score[1])


# In[ ]:

# plot ROC curve


# In[ ]:

parameters = [[1,2,3,4,5],['sgd','a','b','c','d'],[11,22,33,44,55]]


# In[ ]:

type(parameters[1][1])
type(parameters)


# In[ ]:

d = {'LA':'los angeles','SF':'san fransisco'}
print(d['LA'])


# ## Parameters tuning
# _Deep neural network usually need lots of work for tuning._
# 
# **parameters to tune:**
# * epoch number
# * optimizers
# * hidden layer number (optional)
# * neuron number each hidden layer
# * random seed

# In[ ]:

# TUNING
# modularization

# parameters: 
# 0. epoch number
# 1. optimizers
# 2. neuron number each hidden layer
# 3. X_train
# 4. y_train
# 5. X_test
# 6. y_test

# parameters:
# 'epoch'
# 'epoch_pretraining'
# 'optimizer'
# 'neuron'
# 'X_train'
# 'y_train'
# 'X_test'
# 'y_test'


# parameters = ['epoch number', 'optimizers', 'hidden layer number', 'neuron number each hidden layer']

def stackedAutoencoder(parameters):

    # Stacked Autoencoder
    # Train the autoencoder
    # Source: https://github.com/fchollet/keras/issues/358

    random.seed(3)
    np.random.seed(3)

    nb_epoch_pretraining = 10
    batch_size_pretraining = 500

    # Layer-wise pretraining
    encoders = []
    decoders = []
    nb_hidden_layers = [parameters['X_train'].shape[1], parameters['neuron'][0], parameters['neuron'][1]]

    X_train_tmp = np.copy(parameters['X_train'])
    print('original X_train_tmp SIZE:',X_train_tmp.shape)

    dense_layers = []

    for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
        print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
        # Create AE and training
        ae = Sequential()
        if n_out >= 100:
            encoder = containers.Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='tanh'), Dropout(0.5)])
        else:
            encoder = containers.Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='tanh')])
        decoder = containers.Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='tanh')])
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    
        sgd = SGD(lr=2, decay=1e-6, momentum=0.0, nesterov=True)
        ae.compile(loss='mse', optimizer=parameters['optimizer'])
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size_pretraining, nb_epoch=parameters['epoch_pretraining'], verbose = True, shuffle=True)
        # Store trainined weight and update training data
        encoders.append(ae.layers[0].encoder)
        decoders.append(ae.layers[0].decoder)
    
        X_train_tmp = ae.predict(X_train_tmp)    
    
        print('X_train_tmp SIZE:',X_train_tmp.shape)


    ##############
    
    #End to End Autoencoder training
    if len(nb_hidden_layers) > 2:
        full_encoder = containers.Sequential()
        for encoder in encoders:
            full_encoder.add(encoder)

        full_decoder = containers.Sequential()
        for decoder in reversed(decoders):
            full_decoder.add(decoder)

        full_ae = Sequential()
        full_ae.add(AutoEncoder(encoder=full_encoder, decoder=full_decoder, output_reconstruction=False))    
        full_ae.compile(loss='mse', optimizer=parameters['optimizer'])

        print "Pretraining of full AE"
        full_ae.fit(parameters['X_train'], parameters['X_train'], batch_size=batch_size_pretraining, nb_epoch=parameters['epoch_pretraining'], verbose = True, shuffle=True)
    
    #######################################
    nb_epoch = parameters['epoch']
    batch_size = 100

    model = Sequential()
    for encoder in encoders:
        model.add(encoder)

    model.add(Dense(output_dim=nb_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=parameters['optimizer'])
    score = model.evaluate(parameters['X_test'], parameters['y_test'], show_accuracy=True, verbose=0)
    print('Test score before fine turning:', score[0])
    print('Test accuracy before fine turning:', score[1])
    model.fit(parameters['X_train'], parameters['y_train'], batch_size=batch_size, nb_epoch=parameters['epoch'],
              show_accuracy=True, validation_data=(parameters['X_test'], parameters['y_test']), shuffle=True)
    score = model.evaluate(parameters['X_test'], parameters['y_test'], show_accuracy=True, verbose=0)
    print('Test score after fine turning:', score[0])
    print('Test accuracy after fine turning:', score[1])
    TestScore = score[0]
    TestAccuracy = score[1]
    return TestScore, TestAccuracy


# In[ ]:

# TUNING
# tune - test

# parameters: 
# 1. epoch number
# 2. optimizers
# 3. neuron number each hidden layer
# 4. X_train
# 5. y_train
# 6. X_test
# 7. y_test

from keras.layers import containers
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, AutoEncoder, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

# 'epoch'
# 'epoch_pretraining'
# 'optimizer'
# 'neuron'
# 'X_train'
# 'y_train'
# 'X_test'
# 'y_test'


# parameters = [50, 'sgd', [150,100], X_train_norm_narray, y_train_narray, X_test_norm_narray, y_test_narray]
parameters = {
    'epoch' : 50,
    'epoch_pretraining' :10,
    'optimizer' : 'adam',
    'neuron' : [150,100],
    'X_train' : X_train_norm_narray,
    'y_train' : y_train_narray,
    'X_test' : X_test_norm_narray,
    'y_test' : y_test_narray
}
[TestScore, TestAccuracy] = stackedAutoencoder(parameters)


# In[ ]:

# TUNING CYCLE
# form a dictionary of test parameter cases

# TUNING
# tune - test

# parameters: 
# 1. epoch number
# 2. optimizers
# 3. neuron number each hidden layer
# 4. X_train
# 5. y_train
# 6. X_test
# 7. y_test

from keras.layers import containers
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, AutoEncoder, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

# 'epoch'
# 'epoch_pretraining'
# 'optimizer'
# 'neuron'
# 'X_train'
# 'y_train'
# 'X_test'
# 'y_test'


####

## form tuning parameters dictionary
# epoch number is from 50 60 70 80 90 ... 180 190 200

epochs = [i for i in range(50,200,10)]
# epochs = [i for i in range(1,4)]
tuning_iterr = len(epochs)

parameters_epoch = {
    'epoch_pretraining': 10,
    'optimizer': 'adam' ,
    'neuron' : [150,100],
    'X_train' : X_train_norm_narray,
    'y_train' : y_train_narray,
    'X_test' : X_test_norm_narray,
    'y_test' : y_test_narray
}

#epochRecords = {}

epoch_Accuracy = np.zeros(tuning_iterr)
epoch_Score = np.zeros(tuning_iterr)
epoch_Record = np.ndarray(shape=(tuning_iterr, 3), dtype=float, order='F')
for iterr in range(0,tuning_iterr):
    parameters_epoch['epoch'] = epochs[iterr]
    [epoch_Score[iterr], epoch_Accuracy[iterr]] = stackedAutoencoder(parameters_epoch)
    epoch_Record[iterr] = [iterr,epoch_Score[iterr], epoch_Accuracy[iterr]]

# write the epoch_Record into text file
pickle.dump( epoch_Record, open( "epoch_Record.p", "wb" ) )
# test1 = pickle.load( open( "epoch_Record.p", "rb" ) )



# In[ ]:



