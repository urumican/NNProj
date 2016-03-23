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
