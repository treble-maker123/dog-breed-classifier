from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import load_model

import numpy as np
import cv2

class Classifier:
    '''
    A class that uses ResNet-50 and a custom model to determine whether an image contains a dog
    as well as its breed.
    '''

    def __init__(self):
        self.__model = load_model('dog_classifier/model/classifier.model.hdf5')
        self.__dog_names = np.load('dog_classifier/dog-names.npy')
        self.__face_cascade = cv2.CascadeClassifier('dog_classifier/model/haarcascade_frontalface_alt.xml')

    def determine_breed(self, path_or_img):
        '''
        Determine the breed of dog that is contained in the supplied image.

        @param path_or_img: path to the local image as a string or a PIL Image object
        @return: an array of tuple containing the name of the breed and the probability.
        '''
        tensor = preprocess_input(self.__transform(path_or_img))
        bottleneck_feature = ResNet50(weights='imagenet', include_top=False).predict(tensor)
        prediction = self.__model.predict(bottleneck_feature)
        top_five = self.__top_five(prediction)

        value_sum = sum([ value for index, value in top_five])
        return [ (self.__dog_names[index], str(round((value/value_sum)*100, 2))) for index, value in top_five]
    
    def is_human(self, path_or_img):
        '''
        Whether the image supplied contains a human

        @param path_or_img: path to the local image as a string or a PIL Image object
        @return: True if the image contains a dog, False if otherwise
        '''
        if (isinstance(path_or_img, str)):
            img = cv2.imread(path_or_img, 0)
        else:
            img = np.array(path_or_img)
       
        faces = self.__face_cascade.detectMultiScale(img)
        return len(faces) > 0

    def is_dog(self, path_or_img):
        '''
        Whether the image supplied contains a dog

        @param path_or_img: path to the local image as a string or a PIL Image object
        @return: True if the image contains a dog, False if otherwise
        '''
        resnet = ResNet50(weights="imagenet")
        img = preprocess_input(self.__transform(path_or_img))
        prediction = np.argmax(resnet.predict(img))
        return (prediction <= 268) and (prediction >= 151)

    def __transform(self, path_or_img):
        '''
        Transforms a PIL Image into a 4D tensor of shape [1, 224, 224, 3]
        '''
        if (isinstance(path_or_img, str)):
            img = image.load_img(path_or_img, target_size=(224, 224))
        else:
            img = path_or_img.resize((224, 224))

        array = image.img_to_array(img)
        return np.expand_dims(array, axis=0)

    def __top_five(self, prediction):
        '''
        Returns the top five most likely candidates and their probability based on the prediction array.

        @param prediction: a 2D array of shape [1, 133] that contains the results of prediction.
        @return: five tuples of (index, value) in a list containing the top five predictions.
        '''
        top = 5
        indices = [0 for i in range(0, top)]
        values = [float('-inf') for i in range(0, top)]

        for index, value in enumerate(prediction[0]):
            smallest = np.argmin(values)
            if value > values[smallest]:
                indices[smallest] = index
                values[smallest] = value
        
        return list(zip(indices, values))