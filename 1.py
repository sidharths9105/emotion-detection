import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense ,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


directory = 'D:/train'
target_size = (48,48)
batch_size = 32
class_mode = 'categorical'
rescale = 1./255


datagen = ImageDataGenerator(
    rescale=rescale,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    directory=directory,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=class_mode,
    shuffle=True,
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    directory=directory,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=class_mode,
    shuffle=True,
    subset='validation'
)

# print(train_generator.class_indices)

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7,activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    
)

model.save('emotion_detect_model.h5')

model.summary()



# Plot training & validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
