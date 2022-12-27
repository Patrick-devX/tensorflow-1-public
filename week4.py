import os.path

from tensorflow import keras
import numpy as np
import tensorflow as tf
import google
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mping



def build_model(train_generator):
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
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    #Train the modle
    print(f'\nModel Training:')

    history = model.fit(train_generator,
              steps_per_epoch=8,
              epochs=15,
              verbose=1)
    return model, history



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

    nrows = 4
    ncols = 4
    pic_index = 10
    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)
    pic_index += 8

    nex_horse_pics = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
    nex_human_pics = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index - 8:pic_index]]

    for i, img_path in enumerate(nex_horse_pics+nex_human_pics):
        sp = plt.subplot(nrows, ncols, i+1)
        sp.axis('off')
        img = mping.imread(img_path)
        plt.imshow(img)
    plt.show()



    #All images will be rescaled by 1./255

    train_datagen = ImageDataGenerator(rescale=1/255)
    print(train_datagen)
    train_generator = train_datagen.flow_from_directory(
        directory='./horse-or-human/',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary')

    model, history = build_model(train_generator)
    print(f"The model reached the desired accuracy after {len(history.epoch)} epochs")







