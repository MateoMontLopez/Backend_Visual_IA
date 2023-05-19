import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

"""
Parametros
"""
#Definimos parametros y dimensiones de las imagenes que la IA usara como referencia 
epocas=20
longitud, altura = 150, 150
batch_size = 32
pasos = 1000
validation_steps = 300
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 2
lr = 0.0004

#Preparamos nuestras imagenes para el entrenamiento indicandole parametros de visualizacion
entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Llamamos imagenes de entrenamiento
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical')

#Llamamos imagenes de validacion
validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical')


"""
Creamos nuestra Red Neuronal Convolucional
"""

cnn = Sequential()
#Primera capa de la Red Neuronal
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool)) #Segunda capa para reducir la cantidad de datos de la capa anterior

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same")) #Tercera Capa
cnn.add(MaxPooling2D(pool_size=tamano_pool)) #Cuarta Capa

#Capas de clasificacion
cnn.add(Flatten()) #Se aplana la informacion de la imagen obtenida de las capas anteriores
cnn.add(Dense(256, activation='relu')) #Quinta capa normal, recibe datos de la anterior
cnn.add(Dropout(0.5)) #Se apagan el 50% de las neuronas para que hallan varias rutas de aprendizaje
cnn.add(Dense(clases, activation='softmax')) #Sexta y ultima capa, clasificacion de imagenes

#Optimizacion del algoritmo
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

#Entrenamiento
cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch = pasos,
    epochs = epocas,
    validation_data = validacion_generador,
    validation_steps = validation_steps)

#Guardamos el proceso de entrenamiento para importarlo en programa de prediccion
target_dir = './modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
