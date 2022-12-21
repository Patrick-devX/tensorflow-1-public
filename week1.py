
from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.5):
            print('\nLoss is low so cancelling training!')
            self.model.stop_training = True


def build_model1(xs, ys):
    model = keras.Sequential([keras.layers.Dense(1, input_shape=([1]), name="layer1")])
    model.compile(optimizer="sgd", loss='mean_squared_error', metrics=['accuracy'])
    model.fit(xs, ys, epochs=5)
    return model

def build_model2(data, callbacks):
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation = tf.nn.relu , name="denselayer1"),
        keras.layers.Dense(10, activation=tf.nn.softmax, name="denselayer2")
    ])
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data['train_images'], data['train_labels'], epochs=9, callbacks=[callbacks])
    model.evaluate(data['test_images'], data['test_labels'])
    return model

def build_model3(data, callbacks):
    model = keras.Sequential([

        #Add Convolutions and MaxPooling
        keras.layers.Conv2D(63, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(63, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        # Same Layers as Before
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    #Print Model Summary
    model.summary()

    #Compile the model
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #Train the modle
    print(f'\nModel Training:')
    model.fit(data['train_images'], data['train_labels'], epochs=1)
    model.evaluate(data['test_images'], data['test_labels'])
    return model





if __name__ == "__main__" :
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    fmnist = keras.datasets.fashion_mnist
    #model = build_model1(xs, ys)
    #print(model.predict([10.0]))

    (train_images, train_labels), (test_images, test_labels) = fmnist.load_data()

    data = {'train_images': train_images,
            'train_labels': train_labels,
            'test_images': test_images,
            'test_labels': test_labels}
    index = 0

    #print(f'\nIMAGE LABEL:\n {train_labes[index]}')
    #print(f'\nIMAGE PIXEL ARRAY:\n {train_images[index]}')
    print(f'IMAGE SIZE: {train_images[index].shape}')
    print('BEFORE SCALE')
    print(f'Max PIXL VALUE: {train_images[index].max()}')
    print(f'Min PIXL VALUE: {train_images[index].min()}')

    train_images = train_images/ 255.0
    print('AFTER SCALE')
    print(f'Max PIXL VALUE: {train_images[index].max()}')
    print(f'Min PIXL VALUE: {train_images[index].min()}')

    'UNDERSTANDING SOFTMAX'
    inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
    print(inputs.shape)
    print(inputs)
    inputs = tf.convert_to_tensor(inputs)
    print(inputs)
    print(f'input to sofmax function: {inputs.numpy()}')
    outputs = tf.keras.activations.softmax(inputs)
    print(f'output to sofmax function: {outputs.numpy()}')

    #Get the index with highest value
    prediction = np.argmax(outputs)
    print(f'Class with the highest probabilit: {prediction}')

    callbacks = myCallback()
    # Train Model2
    # model2 = build_model2(data, callbacks)
    #classifications = model2.predict(data['test_images'])
    #print(classifications[0])
    print(data['test_labels'][0])

    model3 = build_model3(data, callbacks)
    f, ax = plt.subplots(3,4)

    FIRST_IMAGE = 0
    SECOND_IMAGE = 23
    THIRD_IMAGE = 28
    CONV_NUMBER = 1

    layers_outputs = [layer.output for layer in model3.layers]
    activation_model = tf.keras.models.Model(inputs=model3.input, outputs=layers_outputs)

    for x in range(0, 4):
        f1 = activation_model.predict(data['test_images'][FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        ax[0, x].imshow(f1[0, :, :, CONV_NUMBER], cmap='inferno')
        ax[0, x].grid(False)

        f2 = activation_model.predict(data['test_images'][SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        ax[1, x].imshow(f2[0, :, :, CONV_NUMBER], cmap='inferno')
        ax[1, x].grid(False)

        f3 = activation_model.predict(data['test_images'][THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        ax[2,x].imshow(f3[0, :, :, CONV_NUMBER], cmap='inferno')
        ax[2,x].grid(False)


