#importing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import opendatasets as od

od.download("https://www.kaggle.com/datasets/antobenedetti/animals")  # download using opendatasets
train_path = 'animals/animals/train'  # specify the train path
# ImageDataGenerator for data augmentation and preprocessing
# rescales the pixel values to [0,1], shear shifts a bit for random distortion, zoom range zooms in to help mdoel learn different ranges
#horizontal flip randomly flips it, validation split sets aside 20% of data for validation (unbiased evulation of a model fit), (training: used to fitmodel)
datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                             validation_split=0.2)
#data that fits model, target size is dimensions, batch_size-> every batch contains 32 images -> class_mode->basically data = a vector ex: [0,0,1] = cat, [0,1,0] = dog, [1,0,0] = lion
train_generator = datagen.flow_from_directory(train_path, target_size=(512, 512), batch_size=32,
                                              class_mode='categorical', subset='training')

#essentially the same thing except this time it is validation
validation_generator = datagen.flow_from_directory(train_path, target_size=(512, 512), batch_size=32,
                                                   class_mode='categorical', subset='validation')

# build the CNN model
#sequential method <- takes in layers as an input and builds a linear queue of layers
#first line inputs 32 inputs, size = (3,3), (so Conv slides over 3x3 so ) regions, input, 512x512 3 channels for RGB, relu is conventional
#Conv2D is a type of Convolution layer -> recognies patters in an image. This one in specific 2D takes up all the pixel values and adds then all resulting in one single "pixel" with that wholes image data
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(512, 512, 3), activation='relu'),
    #this takes the maximum from the above processin, for each 2x2 region
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #Does the same thing again, essentially makes the process easier as its shrinking the image to its most important points
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #.flatten() turns everything into a vector making it easier to pass into the final layer
    tf.keras.layers.Flatten(),
    #this is a layer where every neuron is conencted to all previous layer's neurons, basically our decision layer
    tf.keras.layers.Dense(128, activation='relu'),
    #regularization technique used to prevent overfitting <- randomly sets a fraction of inputs to zero to prevent network from relying to much on specific neurons
    #encourages better learning
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 output classes, for 5 animals 'softmax' is math
])

# compile the model
model.compile(optimizer='adam',  # random gradient descent
              loss='categorical_crossentropy', #common loss metric that measures beteweened predicted and true
              metrics=['accuracy'])
#metrics -> basically it's what is being tracked, accuracy in this case refers to proportion  of correctly classified data.

# train the model
history = model.fit(train_generator, #training
                    epochs=10, #iteration through the data
                    validation_data=validation_generator) #validation

# evulate the model
test_loss, test_acc = model.evaluate(validation_generator) #evulate returns two floating point integers that are the test_loss and test_accuarcy of the training
print(f'Test accuracy: {test_acc}')





