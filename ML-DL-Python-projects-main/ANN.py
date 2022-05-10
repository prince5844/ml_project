#PPP done
#Artificial Neural Network
'''
Ref parameter tuning: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
Ref call backs: https://machinelearningmastery.com/check-point-deep-learning-models-keras/
Ref TF serving: https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#exporting-a-model-with-tensorflow-serving
'''
#Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split as tts, cross_val_score as cvs, GridSearchCV as gs, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_config, Model
from keras.layers import Dense, Dropout
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier as kc
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder, tag_constants, signature_constants, signature_def_utils_impl
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
#importing dataset
dataset = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Churn_Modelling.csv')

'''To separate the dataset based on 1 of the values in a feature space, and a dataset based on the rest
idx = dataset['Geography'].str.contains('Spain', regex = False)
df1 = dataset[idx].sample(frac = 0.26, random_state = 5)
df2 = dataset[~idx]
'''
dataset.isnull().sum()#/len(dataset)*100 #checking how many missing values in each variable, unblock for % of missing values
input_variables = dataset.iloc[:, 3:13].values
output = dataset.iloc[:, 13].values
#Encoding categorical data and independent variable
labelEncoderX = LabelEncoder()
input_variables[:, 1] = labelEncoderX.fit_transform(input_variables[:, 1])
input_variables[:, 2] = labelEncoderX.fit_transform(input_variables[:, 2])
hotEncoder = OneHotEncoder(categorical_features = [1])
input_variables = hotEncoder.fit_transform(input_variables).toarray()
input_variables = input_variables[:, 1:]
#Feature Scaling
scaler = MinMaxScaler()
input_variables = scaler.fit_transform(input_variables)
#Splitting dataset into training and test set
xTrain, xTest, yTrain, yTest = tts(input_variables, output, test_size = 0.2, random_state = 0)

'''Building ANN with Keras'''

#Using TensorFlow distributed training by registering a TF session linked to a cluster with Keras
local_server = tf.train.Server.create_local_server()
session = tf.Session(local_server.target)
backend.set_session(session)

'''1st way of defining the ANN architecture'''

#Initializing ANN
classifier = Sequential()
#Adding input layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1))
#Adding more hidden layers
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
#Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''2nd way, by defining the ANN architecture as a definition'''

def create_model():
    classifier = Sequential()
    #Adding input layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(rate = 0.1))
    #Adding hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    #Adding output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    #Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = kc(build_fn = create_model, verbose = 1, batch_size = 128)

# Tuning for optimal hyper parameters using Grid Search
batch_Size = [8, 16, 32, 50, 64, 100, 128] #probing optimal batch size
epochs = [10, 50, 100, 150, 200] #probing optimal no of epochs
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] #probing for best optimizer
learn_rate = [0.001, 0.01, 0.1, 0.2 ,0.3] #probing for optimizer learning rate
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9] #probing for momentum
initialization = ['normal', 'zero', 'uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_uniform'] #probing for weight initialization mode
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] #probing for optimal activation
weights = [1, 2, 3, 4, 5] #dropout is best combined with a weight constraint such as the max norm constraint
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #probing for best dropout rate
no_of_neurons = [1, 5, 10, 15, 20, 25, 30] #probing for no of neurons in hidden layers

param_grid = dict(batch_size = batch_Size, epochs = epochs, optimizer = optimizers, learn_rate = learn_rate, momentum = momentum, init = initialization, activation = activation, weight_constraint = weights, dropout_rate = dropout_rate, neurons = no_of_neurons)
grid = gs(estimator = classifier, param_grid = param_grid, n_jobs = -1)

gSearch = grid.fit(input_variables, output)
best_params = gSearch.best_params_
best_accuracy = gSearch.best_score_

# summarize results
print("Best score: %f using params %s" % (gSearch.best_score_, gSearch.best_params_))
means = gSearch.cv_results_['mean_test_score']
stds = gSearch.cv_results_['std_test_score']
params = gSearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 10)
results = cvs(classifier, input_variables, output, cv = kfold)
print(results.mean())

'''Check point of ANN model improvements while training by max mode'''
filepath = 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]

# Fit the model
classifier.fit(xTrain, yTrain, validation_split = 0.33, epochs = 150, batch_size = 10, verbose = 0)

#Load weights from the saved checkpoint in case the above process is interrupted
classifier.load_weights('weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')

#Re-compiling the ANN based on the saved weights from above file
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print('Created model from the weights saved in Weights.Best.Hdf5')

#Re-fitting ANN to training set
classifier.fit(xTrain, yTrain, epochs = 100, batch_size = 10, callbacks = callbacks_list, validation_split = 0.33, verbose = 0)

#estimate accuracy on whole dataset using loaded weights
scores = classifier.evaluate(input_variables, output, verbose = 1)
print('%s: %.2f%%' % (classifier.metrics_names[1], scores[1] * 100))

'''Check point of best ANN model by max mode'''
filepath = 'weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]

#Fit the model
classifier.fit(xTrain, yTrain, validation_split = 0.33, epochs = 150, batch_size = 10, callbacks = callbacks_list, verbose = 1)

#Load weights from the saved checkpoint in case the above process is interrupted
classifier.load_weights('weights.best.hdf5')

#Re-compiling the ANN based on the saved weights from above file
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print('Created model from the weights saved in Weights.Best.Hdf5')

#Re-fitting ANN to training set
classifier.fit(xTrain, yTrain, epochs = 100, batch_size = 10, callbacks = callbacks_list, validation_split = 0.33, verbose = 0)

# estimate accuracy on whole dataset using loaded weights
scores = classifier.evaluate(input_variables, output, verbose = 1)
print('%s: %.2f%%' % (classifier.metrics_names[1], scores[1] * 100))

#Predicting the test set results
yPred = classifier.predict(xTest)
yPred = (yPred > 0.5)

#Predicting for new input
yNew = classifier.predict(scaler.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
yNew = (yNew > 0.5)

#Creating confusion matrix to find TP,TN,FP,FN
c_matrix = confusion_matrix(yTest, yPred)

#Evaluating the ANN
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = kc(build_fn = build_classifier, batch_size = 64, epochs = 100)
accuracies = cvs(estimator = classifier, X = xTrain, y = yTrain, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

'''Below is an attempt to export the ANN as TF Serving'''

#Exporting model with TensorFlow Serving
sess = tf.Session()
backend.set_session(sess)
backend.set_learning_phase(0)  # all new operations will be in test mode from now on

#serialize the model and get its weights, for quick re-building
config = classifier.get_config()
weights = classifier.get_weights()

#re-build a model where the learning phase is now hard-coded to 0
new_model = Model.from_config(config) #removed Sequential.from_config(config)
new_model.set_weights(weights)

#Exporting as TFS
export_path = 'D:\Programming Tutorials\Machine Learning\Machine Learning AZ - Hadllin De Ponteves\Part 8 - Deep Learning\Section 39 - Artificial Neural Networks (ANN)' # where to save the exported graph
export_version = 1 # version number (integer)
saver = tf.train.Saver(sharded = True)
model_exporter = exporter.Exporter(saver)
signature = predict_signature_def(input_tensor = classifier.input, scores_tensor = classifier.output)
model_exporter.init(sess.graph.as_graph_def(), default_graph_signature = signature)
model_exporter.export(export_path, tf.constant(export_version), sess)

'''
Another attempt: https://medium.com/@brianalois/simple-keras-trained-model-export-for-tensorflow-serving-23fa5dfeeecc
'''
session = tf.Session()
backend.set_session(session)
backend.set_learning_phase(0)
model_version = "2"
epoch = 100
from numpy.random import randn
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
X = pd.DataFrame(randn(4, 3),['A', 'B', 'C', 'D'], ['X', 'Y', 'Z'])
Y = np.array([[0], [1], [1], [0]])
classifier = Sequential()
classifier.add(Dense(8, input_dim = 2))
classifier.add(Activation('tanh'))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))
sgd = SGD(lr = 0.1)
classifier.compile(loss = 'binary_crossentropy', optimizer = sgd)
classifier.fit(X, Y, batch_size = 1, nb_epoch = epoch, verbose = 1)
x = classifier.input
y = classifier.output
prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"inputs" : x}, {"prediction" : y})
valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
if(valid_prediction_signature == False):
    raise ValueError("Error: Prediction signature not valid!")
builder = saved_model_builder.SavedModelBuilder('./' + model_version)
legacy_init_op = tf.group(tf.tables_initializer(), name = 'legacy_init_op')
builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature, }, legacy_init_op = legacy_init_op)
builder.save()