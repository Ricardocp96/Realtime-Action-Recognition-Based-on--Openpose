import argparse
import pandas as pd
import numpy as np
import mylib.data_preprocessing as dpp
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import  Callback
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras.models import load_model
from tensorflow.keras.layers import BatchNormalization 
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz 
import seaborn as sns


import itertools
#from keras.layers.normalization import BatchNormalization
#from keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import LeakyReLU
from sklearn.metrics import confusion_matrix
##For training the DNN model
if __name__ == "__main__":
    # Read dataset from command line
    key_word = "--dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(key_word, required=False, default='../data/skeleton_raw.csv')
    input = parser.parse_args().dataset
    
    # Loading training data
    try:
        raw_data = pd.read_csv(input, header=0)
    except:
        print("Dataset not exists.")
    # X: input, Y: output
    dataset = raw_data.values
    X = dataset[:, 0:36].astype(float)
    Y = dataset[:, 36]

    # Data pre-processing
    # X = dpp.head_reference(X)
    
    X_pp = []
    for i in range(len(X)):
        X_pp.append(dpp.pose_normalization(X[i]))
     
    X_pp = np.array(X_pp)
    

    # Encoder the class label to number
    # Converts a class vector (integers) to binary class matrix
    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(Y)
    matrix_Y = np_utils.to_categorical(encoder_Y)
    print(Y[0], ": ", encoder_Y[0])
    print(Y[650], ": ", encoder_Y[650])
    print(Y[1300], ": ", encoder_Y[1300])
    print(Y[1950], ": ", encoder_Y[1950])
    print(Y[2600], ": ", encoder_Y[2600])

    # Split into training and testing data
    # random_state:
    X_train, X_test, Y_train, Y_test = train_test_split(X_pp, matrix_Y, test_size=0.1, random_state=42)

    # Build DNN model with keras
    model = Sequential()
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=5, activation='softmax'))

    # Training
    # optimiser: Adam with learning rate 0.0001
    # loss: categorical_crossentropy for the matrix form matrix_Y
    # metrics: accuracy is evaluated for the model
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # batch_size: number of samples per gradient update
    # epochs: how many times to pass through the whole training set
    # verbose: show one line for every completed epoch
    r=model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=2, validation_data=(X_test, Y_test))
    model.summary()

    epochs = range(50) # 50 is the number of epochs
    train_acc = r.history['accuracy']
    valid_acc = r.history['val_accuracy']
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Categorical crossentropy loss')
    plt.legend()
    plt.show()
    plt.plot(epochs, train_acc, 'bo', label='Training Accuracy')
    #plt.plot(epochs, valid_acc, 'r', label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    loss_and_acc=model.evaluate(X_train,Y_train)
    print('loss = ' + str(loss_and_acc[0]))
    print('accuracy = ' + str(loss_and_acc[1]))





#ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
#x.set_title('Seaborn Confusion Matrix with labels\n\n')
#ax.set_xlabel('\nPredicted Values')
#ax.set_ylabel('Actual Values ')
#cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))

#plt.show()
        







# Callback class to visialize training progress


    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.loss = {'batch':[], 'epoch':[]}
            self.accuracy = {'batch':[], 'epoch':[]}
            self.val_loss = {'batch':[], 'epoch':[]}
            self.val_acc = {'batch':[], 'epoch':[]}

        def on_batch_end(self, batch, logs={}):
            self.loss['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))

        def on_epoch_end(self, batch, logs={}):
            self.loss['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))

        def loss_plot(self, loss_type):
            iters = range(len(self.loss[loss_type]))
            plt.figure()
        # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
            plt.plot(iters, self.loss[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
            # val_acc
               plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
               plt.plot(iters, self.val_loss[loss_type], 'k', label='Test loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.show()


    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix showing True Labe and predicted Label for Action recognition',
                          cmap=plt.cm.Blues):
        
         # This function prints and plots the confusion matrix.
        # Normalization can be applied by setting `normalize=True`
        
        
        
        if normalize:

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
           plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    # Save the trained model
    

# print('Test:')
#score, 
    score,accuracy = model.evaluate(X_test,Y_test,batch_size=32)
    print('Test Score:{:.3}'.format(score))
    print('Test accuracy:{:.3}'.format(accuracy))
# # confusion matrix
    Y_pred = model.predict(X_test)
    cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))
    np.set_printoptions(precision=2)
#
    his=LossHistory()
    plt.figure()
    class_names = ['kick', 'punch', 'squat', 'stand','wave']
    plot_confusion_matrix(cfm, classes=class_names, title='Confusion Matrix Showing True Label and the Predicted label For Action recognition')

    plt.show()
    his.loss_plot('epoch')
    model.summary()
