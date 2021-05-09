import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile
import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import os
import random
import time
import math
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from engineModules import *

root = tk.Tk()

canvas = tk.Canvas(root, width=600, height=600)
canvas.grid(columnspan=3, rowspan=15)

#title
title = tk.Label(root, text="Reverse Image Search Engine", font="Raleway 22 bold")
title.grid(column=1, row=0)

#instructions
instructions = tk.Label(root, text="1. Realizar Feature Extraction", font="Raleway")
instructions.grid(columnspan=3, rowspan=1, column=0, row=1)

def featureExtraction():
    # file = askopenfile(parent=root, mode='rb', title="Choose a file", filetype=[("Image file", ".jpg")])
    # if file:
    #     queryImg = Image.open(file)
    model_architecture = 'resnet'
    model = model_picker(model_architecture)
    features = extract_features('datasets/product10k/chumpas/1005559.jpg', model)
    print(len(features))
    
    root_dir = 'datasets/product10k'
    filenames = sorted(get_file_list(root_dir))

    feature_list = []
    for i in range(len(filenames)):
        feature_list.append(extract_features(filenames[i], model))
    
    batch_size = 64
    datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = datagen.flow_from_directory(root_dir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode=None,
                                            shuffle=False)

    num_images = len(generator.filenames)
    num_epochs = int(math.ceil(num_images / batch_size))

    start_time = time.time()
    feature_list = []
    feature_list = model.predict_generator(generator, num_epochs)
    end_time = time.time()

    for i, features in enumerate(feature_list):
        feature_list[i] = features / norm(features)

    feature_list = feature_list.reshape(num_images, -1)

    print("Num images   = ", len(generator.classes))
    print("Shape of feature_list = ", feature_list.shape)
    print("Time taken in sec = ", end_time - start_time)

    filenames = [root_dir + '/' + s for s in generator.filenames]

    os.makedirs(os.path.dirname("datasets/data/features-product10k-resnet.pickle"),exist_ok=True)

    os.makedirs(os.path.dirname("datasets/data/filenames-product10k.pickle"), exist_ok=True)

    os.makedirs(os.path.dirname("datasets/data/class_ids-product10k.pickle"), exist_ok=True)

    pickle.dump(generator.classes, open('datasets/data/class_ids-product10k.pickle','wb'))
    pickle.dump(feature_list, open('datasets/data/features-product10k-resnet.pickle', 'wb'))
    pickle.dump(filenames, open('datasets/data/filenames-product10k.pickle','wb'))

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)

    train_generator = train_datagen.flow_from_directory(root_dir,
                                                        target_size=(IMG_WIDTH,
                                                                    IMG_HEIGHT),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=12345,
                                                        class_mode='categorical')
    model_finetuned = model_maker()
    model_finetuned.compile(loss='categorical_crossentropy',
                optimizer=tensorflow.keras.optimizers.Adam(0.001),
                metrics=['acc'])
    model_finetuned.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / batch_size),
        epochs=10)

    os.makedirs(os.path.dirname("datasets/model/"), exist_ok=True)

    model_finetuned.save('datasets/model/model-finetuned.h5')

    start_time = time.time()
    feature_list_finetuned = []
    feature_list_finetuned = model_finetuned.predict_generator(generator, num_epochs)
    end_time = time.time()

    for i, features_finetuned in enumerate(feature_list_finetuned):
        feature_list_finetuned[i] = features_finetuned / norm(features_finetuned)

    feature_list = feature_list_finetuned.reshape(num_images, -1)

    print("Num images   = ", len(generator.classes))
    print("Shape of feature_list = ", feature_list.shape)
    print("Time taken in sec = ", end_time - start_time)

    pickle.dump(feature_list,open('datasets/data/features-product10k-resnet-finetuned.pickle', 'wb'))

    print("---- Feature Extraction ended------")


#Feature Extraction button
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=lambda:featureExtraction(), font="Raleway", bg="#0d1117", fg="white", height=1, width=15)
browse_text.set("Start")
browse_btn.grid(column=1, row=2, rowspan=1)
root.mainloop()