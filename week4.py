import os.path

from tensorflow import keras
import numpy as np
import tensorflow as tf
import google
import zipfile



def build_model3(data):
    model = keras.Sequential([
        #Add Convolutions and MaxPooling
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    #Print Model Summary
    model.summary()
    #Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    #Train the modle
    print(f'\nModel Training:')

    model.fit(train_generator,
              steps_per_epoch=8,
              epochs=15,
              validation_data=validation_generator,
              validation_steps=8,
              verbose=2)

    model.evaluate(data['test_images'], data['test_labels'])
    return model



if __name__ == "__main__" :
    #uplooded = google.colab.files.upload()

    #for fn in uplooded.keys():
        #path = '/content/' + fn
        #img = keras.preprocessing.image.load_im(path, targetsize=(300, 300))
        #x = keras.preprocessing.image.img_to_arry(img)
        #x = np.expand_dims(x, axis=0)

        #images = np.vstack([x])
        #classes = model.predict(images, batch_size=18)
        #print(classes[0])
        #if classes[0]>0.5:
            #print(fn+ 'is a human')
        #else:
            #print(fn + 'is a horses')

    #training hores picture
    train_horse_dir = os.path.join('./horse-or-human/train/horses')
    #training hores picture
    train_human_dir = os.path.join('./horse-or-human/train/humans')

    train_horse_names = os.listdir(train_horse_dir)
    print(train_horse_names[:10])
    train_human_names = os.listdir(train_human_dir)
    print(train_human_names[:10])

    print(f'total training horses images: {len(os.listdir(train_horse_dir))}')
    print(f'total training human images: {len(os.listdir(train_human_dir))}')







