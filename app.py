import os
import random
import time
import math
import numpy as np
from numpy.linalg import norm
from pathlib import Path

import tkinter as tk
from tkinter.filedialog import askopenfile

import pickle
from PIL import Image, ImageTk

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input                               
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from engineModules import *

#Variables constantes
ROOT_DIR = 'datasets/product10k'

def feature_extraction():

    """ Descripción de esta función """

    model = ResNet50(weights='imagenet', #Creación de la CNN ResNet50: 50 Layers Deep
                         include_top=False,
                         input_shape=(224, 224, 3),
                        pooling='max')

    filenames = sorted(get_file_list(ROOT_DIR)) #Dentro de esta lista se encuentran todos nombres de los archivos en el dataset, de manera ordenada.

    feature_list = []
    for i in range(len(filenames)):                                     #Se realiza la extracción de features a cada imagen cuyo nombre está en la lista de filenames
        feature_list.append(extract_features(filenames[i], model))      #y el resultado se agrega a la lista llamada feature_list
    
    batch_size = 64
    datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = datagen.flow_from_directory(ROOT_DIR,
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

    filenames = [ROOT_DIR + '/' + s for s in generator.filenames]

    #Creacion de archivos para almacenar los resultados obtenidos arriba.
    os.makedirs(os.path.dirname("datasets/data/features-product10k-resnet.pickle"),exist_ok=True)
    os.makedirs(os.path.dirname("datasets/data/filenames-product10k.pickle"), exist_ok=True)
    os.makedirs(os.path.dirname("datasets/data/class_ids-product10k.pickle"), exist_ok=True)

    #Escritura en los archivos recien creados.
    pickle.dump(generator.classes, open('datasets/data/class_ids-product10k.pickle','wb'))
    pickle.dump(feature_list, open('datasets/data/features-product10k-resnet.pickle', 'wb'))
    pickle.dump(filenames, open('datasets/data/filenames-product10k.pickle','wb'))

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)

    train_generator = train_datagen.flow_from_directory(ROOT_DIR,
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

  
    os.makedirs(os.path.dirname("datasets/model/"), exist_ok=True)  #Creacion del directorio para guardar una copia del modelo usado.
    model_finetuned.save('datasets/model/model-finetuned.h5')       #Se guarda el modelo en el directorio que se creo.

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

def similarity_search():

    """ Descripción de esta función """

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

def upload_and_search():

    """ Descripción de esta función """

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

os.getcwd()

root = tk.Tk()

#Set de las dimensiones de la ventana de la aplicación y también la cuadrícula o grid donde irán los elementos como texto y botones.
canvas = tk.Canvas(root, width=600, height=600)
canvas.grid(columnspan=3, rowspan=15)

#Titulo
title = tk.Label(root, text="Reverse Image Search Engine", font="Raleway 22 bold")
title.grid(column=1, row=0)

#Instrucciones y boton para la extraccion de features de las imagenes
feature_extraction_text = tk.Label(root, text="1. Realizar Feature Extraction", font="Raleway")
feature_extraction_text.grid(columnspan=3, rowspan=1, column=0, row=1)
feature_extraction_button_text = tk.StringVar()
feature_extraction_button_text.set("Start")
feature_extraction_button = tk.Button(root, textvariable=feature_extraction_button_text, command=lambda:feature_extraction(), font="Raleway", bg="#0d1117", fg="white", height=1, width=15)
feature_extraction_button.grid(column=1, row=2, rowspan=1)

#Instrucciones y boton para mostrar el agrupamiento de las fotos usando el algoritmo t-sne
show_clusters_text = tk.Label(root, text="2. Mostrar Clusters de imagenes utilizando TSNE", font="Raleway")
show_clusters_text.grid(columnspan=3, rowspan=1, column=0, row=4)
show_clusters_button_text = tk.StringVar()
show_clusters_button_text.set("Start")
show_clusters_button = tk.Button(root, textvariable=show_clusters_button_text, command=lambda:similarity_search(), font="Raleway", bg="#0d1117", fg="white", height=1, width=15)
show_clusters_button.grid(column=1, row=5, rowspan=1)

#Instrucciones y botón para subir una imagen de nuestro ordenador, y con base en ella, encontrar otras 10 similares en el dataset.
browse_image_text = tk.Label(root, text="3. Ver las 10 imágenes más similares", font="Raleway")
browse_image_text.grid(columnspan=3, rowspan=1, column=0, row=7)
browse_image_button_text = tk.StringVar()
browse_image_button_text.set("Browse")
browse_image_button = tk.Button(root, textvariable=browse_image_button_text, command=lambda:upload_and_search(), font="Raleway", bg="#0d1117", fg="white", height=1, width=15)
browse_image_button.grid(column=1, row=8, rowspan=1)    

#Si encuentra los archivos correspondientes a features list, filenames y class_ids, muestra el texto descriptivo de FOUND, de lo contrario muestra NOT FOUND
if os.path.isfile('datasets/data/class_ids-product10k.pickle') and os.path.isfile('datasets/data/features-product10k-resnet.pickle') and os.path.isfile('datasets/data/filenames-product10k.pickle') and os.path.isfile('datasets/data/features-product10k-resnet-finetuned.pickle'):
    filesFound = tk.Label(root, text="Features, Class_IDs and Filenames FOUND", font="Raleway 8")
    filesFound.grid(columnspan=3, rowspan=1, column=0, row=3)
else:
    filesFound = tk.Label(root, text="Features, Class_IDs and Filenames NOT FOUND", font="Raleway 8")
    filesFound.grid(columnspan=3, rowspan=1, column=0, row=3)

#Ejecutando siempre la aplicación hasta que el usuario la cierre.
root.mainloop()






