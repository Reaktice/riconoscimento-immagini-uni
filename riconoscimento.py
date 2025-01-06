import tensorflow as tf  # Importa il modulo TensorFlow, che include strumenti per costruire e addestrare modelli di machine learning.
from tensorflow.keras.models import load_model  # Importa la funzione per caricare un modello salvato con Keras.
from tensorflow.keras.preprocessing import image  # Importa utilità per il preprocessing delle immagini.
import numpy as np  # Importa la libreria NumPy per lavorare con array multidimensionali.


# Carica il modello addestrato
model = load_model('dogs_vs_cats_model.h5')  # Carica il modello salvato dal file 'dogs_vs_cats_model.h5'.

# Predizione di una nuova immagine
img_path = 'img/cane.jpg'  # Cambia questo con il percorso della tua immagine. Assicurati di avere l'immagine nella directory specificata.
img = image.load_img(img_path, target_size=(150, 150))  # Carica l'immagine e ridimensiona a 150x150 pixel, come il modello si aspetta.
img_array = image.img_to_array(img) / 255.0  # Converte l'immagine in un array NumPy e normalizza i valori dei pixel a [0, 1].
img_array = np.expand_dims(img_array, axis=0)  # Aggiunge una dimensione extra in modo che l'array corrisponda alla forma attesa dal modello (batch_size, height, width, channels).

prediction = model.predict(img_array)  # Utilizza il modello per fare una predizione sull'immagine preprocessata.
if prediction[0] > 0.5:  # Se la predizione è maggiore di 0.5, il modello predice che l'immagine è un gatto.
    print("È un gatto!")  # Stampa "È un gatto!" se la predizione è maggiore di 0.5.
else:  # Se la predizione è 0.5 o inferiore, il modello predice che l'immagine è un cane.
    print("È un cane!")  # Stampa "È un cane!" se la predizione è 0.5 o inferiore.
