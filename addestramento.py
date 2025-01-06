import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Creazione delle cartelle per il dataset
base_dir = 'dataset'  # Directory principale per il dataset
train_dir = os.path.join(base_dir, 'train')  # Directory per il set di addestramento
validation_dir = os.path.join(base_dir, 'validation')  # Directory per il set di validazione

# Definisci i parametri del modello
input_shape = (150, 150, 3)  # Dimensioni dell'immagine in input (larghezza, altezza, canali di colore)
batch_size = 32  # Numero di immagini da processare in un batch
epochs = 20  # Numero di epoche per l'addestramento del modello

# Prepara i generatori di dati
train_datagen = ImageDataGenerator(rescale=1./255)  # Generatore per normalizzare le immagini di addestramento
val_datagen = ImageDataGenerator(rescale=1./255)  # Generatore per normalizzare le immagini di validazione

train_generator = train_datagen.flow_from_directory(
    train_dir,  # Directory contenente le immagini di addestramento
    target_size=(150, 150),  # Dimensioni a cui ridimensionare tutte le immagini
    batch_size=batch_size,  # Numero di immagini da processare in un batch
    class_mode='binary'  # Modalità di classificazione binaria (cane o gatto)
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,  # Directory contenente le immagini di validazione
    target_size=(150, 150),  # Dimensioni a cui ridimensionare tutte le immagini
    batch_size=batch_size,  # Numero di immagini da processare in un batch
    class_mode='binary'  # Modalità di classificazione binaria (cane o gatto)
)

# Costruisci il modello
model = Sequential([  # Inizializza un modello sequenziale
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # Strato convoluzionale con 32 filtri e kernel 3x3
    MaxPooling2D((2, 2)),  # Strato di pooling con finestra 2x2
    Conv2D(64, (3, 3), activation='relu'),  # Strato convoluzionale con 64 filtri e kernel 3x3
    MaxPooling2D((2, 2)),  # Strato di pooling con finestra 2x2
    Conv2D(128, (3, 3), activation='relu'),  # Strato convoluzionale con 128 filtri e kernel 3x3
    MaxPooling2D((2, 2)),  # Strato di pooling con finestra 2x2
    Flatten(),  # Aggiunge uno strato di flattening per convertire i dati 2D in un vettore 1D
    Dropout(0.5),  # Strato di dropout per prevenire l'overfitting, con tasso del 50%
    Dense(512, activation='relu'),  # Strato completamente connesso con 512 unità e funzione di attivazione ReLU
    Dense(1, activation='sigmoid')  # Strato di output con una singola unità e funzione di attivazione sigmoide (per classificazione binaria)
])

# Compila il modello
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Utilizza l'ottimizzatore Adam, la funzione di perdita binary_crossentropy e l'accuratezza come metrica

# Addestra il modello
history = model.fit(
    train_generator,  # Dati di addestramento
    epochs=epochs,  # Numero di epoche per addestrare il modello
    validation_data=validation_generator  # Dati di validazione
)

# Salva il modello
model.save('dogs_vs_cats_model.h5')  # Salva il modello addestrato in un file