import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def confusion_matrix(true_target, predicted_target, n_classes=10):
    # Initialize the confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    
    if len(true_target.shape) == 2:
        # One-hot encoded labels
        true_target = np.argmax(true_target, axis=1)
    
    if len(predicted_target.shape) == 2:
        # One-hot encoded predictions
        predicted_target = np.argmax(predicted_target, axis=1)
    
    # Loop over each pair of true and predicted labels
    for i in range(len(true_target)):
        true_label = true_target[i]
        pred_label = predicted_target[i]
        
        # Increment the corresponding entry in the confusion matrix
        conf_matrix[true_label][pred_label] += 1
    
    return conf_matrix

def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    # set the random seed
    tf.keras.utils.set_random_seed(5368)
    
    #initializing the model
    model = tf.keras.Sequential([
    # convolutional layers
    tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), 
                           input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),

    
    # flatten and dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(10, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    
    #Activation layer
    tf.keras.layers.Activation('softmax')
])
    # configuring the model for Training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #Initializing the trainig process with epoch, batchsize, validation , verbose
    training_history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,verbose=1)
    
    # saveing the model in .h5 file
    model.save('model.h5')

    #ploting actual and predicted values on confusion matrix
    predicted_target = np.argmax(model.predict(X_test),axis = 1) 
    true_target = np.argmax(Y_test,axis = 1) 
    conf_matrix = confusion_matrix(true_target, predicted_target)

    # plot the confusion matrix
    plt.matshow(conf_matrix)
    plt.colorbar()
    plt.savefig('confusion_matrix.png')
    
    # return the trained model, training history, confusion matrix, and test set predictions
    return model, training_history, conf_matrix, predicted_target 
