This repository contains code for training a dog breed classifier. A demo of this app can be found [here](https://www.ziqiangguan.com/projects/dog-classifier).


This model is built on top of the [ResNet-50](https://arxiv.org/abs/1512.03385) developed by Kaiming He et al. A GlobalAveragePooling2D layer and a fully-connected layer with softmax activation is added as trainable layers. The model is then trained with 6680 additional images of 133 breeds of dogs.


The original training (current demo) did not incorporate data augmentation, and the accuracy from validating on the test set was around 80%. The train.py script does have data augmentation code, and I will train a new model with it in the near future and update the demo with the new model.


If you would like to try this out yourself, feel free to reach out to me for the dog images.