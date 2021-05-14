import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile
import numpy as np
from numpy.linalg import norm
import pickle
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
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data


os.getcwd()


root = tk.Tk()

canvas = tk.Canvas(root, width=600, height=600)
canvas.grid(columnspan=3, rowspan=15)

#title
title = tk.Label(root, text="Reverse Image Search Engine", font="Raleway 22 bold")
title.grid(column=1, row=0)

#instructions
instructions = tk.Label(root, text="1. Realizar Feature Extraction", font="Raleway")
instructions.grid(columnspan=3, rowspan=1, column=0, row=1)
instructions1 = tk.Label(root, text="2. Mostrar Clusters de imagenes utilizando TSNE", font="Raleway")
instructions1.grid(columnspan=3, rowspan=1, column=0, row=4)
instructions2 = tk.Label(root, text="3. Ver las 10 imágenes más similares", font="Raleway")
instructions2.grid(columnspan=3, rowspan=1, column=0, row=7)

def featureExtraction():
    # file = askopenfile(parent=root, mode='rb', title="Choose a file", filetype=[("Image file", ".jpg")])
    # if file:
    #     queryImg = Image.open(file)
    model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                        pooling='max')
    features = extract_features('datasets/product10k/chumpas/1023627.jpg', model)
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


def similaritySearch():
    filenames = pickle.load(open('datasets/data/filenames-product10k.pickle', 'rb'))
    feature_list = pickle.load(open('datasets/data/features-product10k-resnet.pickle','rb'))
    class_ids = pickle.load(open('datasets/data/class_ids-product10k.pickle', 'rb'))

    num_images = len(filenames)
    num_features_per_image = len(feature_list[0])
    print("Number of images = ", num_images)
    print("Number of features per image = ", num_features_per_image)

    neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean').fit(feature_list)
    # random_index = 3
    # distances, indices = neighbors.kneighbors([feature_list[random_index]])
    # plt.imshow(mpimg.imread(filenames[random_index]), interpolation='lanczos')
    # plt.show()
    
    neighbors = NearestNeighbors(n_neighbors=len(feature_list),
                                algorithm='brute',
                                metric='euclidean').fit(feature_list)
    distances, indices = neighbors.kneighbors(feature_list)

    # Calculating some stats
    print("Median distance between all photos: ", np.median(distances))
    print("Max distance between all photos: ", np.max(distances))
    print("Median distance among most similar photos: ",
        np.median(distances[:, 2]))

    selected_features = feature_list[:]
    selected_class_ids = class_ids[:]
    selected_filenames = filenames[:]
# You can play with these values and see how the results change
    n_components = 2
    verbose = 1
    perplexity = 30
    n_iter = 1000
    metric = 'euclidean'

    time_start = time.time()
    tsne_results = TSNE(n_components=n_components,
                        verbose=verbose,
                        perplexity=perplexity,
                        n_iter=n_iter,
                        metric=metric).fit_transform(selected_features)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))


    show_tsne(tsne_results[:, 0], tsne_results[:, 1], selected_filenames)

def browseFile():
    file = askopenfile(parent=root, mode='rb', title="Choose a file", filetype=[("Image file", ".jpg")])
    if file:

        filenames = pickle.load(open('datasets/data/filenames-product10k.pickle', 'rb'))
        feature_list = pickle.load(open('datasets/data/features-product10k-resnet.pickle','rb'))
        class_ids = pickle.load(open('datasets/data/class_ids-product10k.pickle', 'rb'))
        neighbors = NearestNeighbors(n_neighbors=10,algorithm='ball_tree',metric='euclidean').fit(feature_list)

        model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                        pooling='max')

        input_shape = (224, 224, 3)
        img = image.load_img(file.name, target_size=(input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        test_img_features = model.predict(preprocessed_img, batch_size=1)

        _, indices = neighbors.kneighbors(test_img_features)

        def similar_images(indices):
            plt.figure(figsize=(15,10), facecolor='white')
            plotnumber = 1    
            for index in indices:
                if plotnumber<=len(indices) :
                    ax = plt.subplot(2,5,plotnumber)
                    plt.imshow(mpimg.imread(filenames[index]), interpolation='lanczos')            
                    plotnumber+=1
            plt.tight_layout()
            plt.show()

        print(indices.shape)
        print("hola")

        plt.imshow(mpimg.imread(file.name), interpolation='lanczos')
        plt.xlabel(file.name.split('.')[0] + '_Original Image',fontsize=20)
        plt.show()
        print('********* Predictions ***********')
        similar_images(indices[0])
        

#Feature Extraction button
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=lambda:featureExtraction(), font="Raleway", bg="#0d1117", fg="white", height=1, width=15)
browse_text.set("Start")
browse_btn.grid(column=1, row=2, rowspan=1)

#Clusters Button
cluster_text = tk.StringVar()
cluster_btn = tk.Button(root, textvariable=cluster_text, command=lambda:similaritySearch(), font="Raleway", bg="#0d1117", fg="white", height=1, width=15)
cluster_text.set("Start")
cluster_btn.grid(column=1, row=5, rowspan=1)

#Upload Button
upload_text = tk.StringVar()
upload_btn = tk.Button(root, textvariable=upload_text, command=lambda:browseFile(), font="Raleway", bg="#0d1117", fg="white", height=1, width=15)
upload_text.set("Browse")
upload_btn.grid(column=1, row=8, rowspan=1)


if os.path.isfile('datasets/data/class_ids-product10k.pickle') and os.path.isfile('datasets/data/features-product10k-resnet.pickle') and os.path.isfile('datasets/data/filenames-product10k.pickle') and os.path.isfile('datasets/data/features-product10k-resnet-finetuned.pickle'):
    #instructions
    filesFound = tk.Label(root, text="Features, Class_IDs and Filenames FOUND", font="Raleway 8")
    filesFound.grid(columnspan=3, rowspan=1, column=0, row=3)
else:
    filesFound = tk.Label(root, text="Features, Class_IDs and Filenames NOT FOUND", font="Raleway 8")
    filesFound.grid(columnspan=3, rowspan=1, column=0, row=3)
    
root.mainloop()






