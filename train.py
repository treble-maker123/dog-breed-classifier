from keras import applications
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import ModelCheckpoint

from sklearn.datasets import load_files

from PIL import ImageFile  
ImageFile.LOAD_TRUNCATED_IMAGES = True # otherwise exception is thrown

import numpy as np

train_path = 'dogImages/train'
valid_path = 'dogImages/valid'
test_path = 'dogImages/test'
model_output = "saved_models/dog-classifier-with-data-aug.hdf5"

batch_size = 16

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

print("####################\n SETTING UP MODEL\n####################")

base = applications.ResNet50(include_top=False, weights='imagenet')
for layer in base.layers:
    layer.trainable = False

output = GlobalAveragePooling2D(name="average")(base.output)
output = Dense(133, activation='softmax', name="final")(output)

model = Model(inputs=base.inputs, outputs=output)
model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath=model_output, 
                               verbose=1, save_best_only=True)

print("Done!")

print("####################\n LOADING DATA\n####################")

# load train, test, and validation datasets
train_files, train_targets = load_dataset(train_path)
valid_files, valid_targets = load_dataset(valid_path)
test_files, test_targets = load_dataset(test_path)

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

print("Done!")

print("####################\n SETTING UP DATA GEN\n####################")

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_datagen = datagen.flow(
                        train_tensors,
                        train_targets,
                        batch_size=batch_size)

valid_datagen = datagen.flow(
                        valid_tensors,
                        valid_targets,
                        batch_size=batch_size)

print("Done!")

print("####################\n TRAINING\n####################")

model.fit_generator(train_datagen,
    train_tensors.shape[0]//batch_size,
    epochs=20, verbose=1, callbacks=[checkpointer],
    validation_data=valid_datagen,
    validation_steps=valid_tensors.shape[0]//batch_size)

model.load_weights(model_output)

predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)