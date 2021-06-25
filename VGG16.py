
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import time
from tensorflow.compat.v1.keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Input, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator

flags = tf.compat.v1.flags
flags.DEFINE_string("dataset", "BookCover30", "The dataset on which training is to be done.")
flags.DEFINE_integer("num_epochs_1", "10", "Number of epochs 1.")
flags.DEFINE_integer("num_epochs_2", "30", "Number of epochs 2.")
flags.DEFINE_integer("batch_size", "64", "Batch Size.")
flags.DEFINE_integer("num_cores", "128", "Number of cores to use while training.")
FLAGS = flags.FLAGS

num_cores = FLAGS.num_cores
num_GPU = 1
num_CPU = 20

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

#config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
K.set_session(session)


input_map_file = FLAGS.dataset + "_map.txt"
file = open(input_map_file, 'r')
image_labels_list = [line.strip() for line in file]
file.close()

num_classes = len(image_labels_list)

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def preprocess_image(path):
    im = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
    im = im - vgg_mean
    return im[:, ::-1] # RGB to BGR

def load_data(dataset):
    input_exp_file = dataset + "_map.txt"
    
    ext_file = open(input_exp_file, 'r')
    image_labels_list = [line.strip() for line in ext_file]    
    ext_file.close()
    
    label_map = {}
    for i in range(len(image_labels_list)):
        label_map[image_labels_list[i]] = i
    
    train_file =  dataset + "files.txt"
    test_file =  dataset + "_testfiles.txt"
    
    ext_file = open(train_file, 'r')
    train_list = [line.strip() for line in ext_file]    
    ext_file.close()
    
    ext_file = open(test_file, 'r')
    test_list = [line.strip() for line in ext_file]    
    ext_file.close()
    
    x_train = []
    y_train = []
        
    num_of_files_recorded = len(train_list)
        
    for j in range(num_of_files_recorded):
            image = cv2.resize(cv2.imread(train_list[j]), (224, 224)).astype(np.float32)
            image = image - vgg_mean
            image = image[:, ::-1] # RGB to BGR
            x_train.append(image)
            lists = train_list[j].split('/')
            label = lists[-2]
            y_train.append(label_map[label])
            
    x_train = np.array(x_train)
    x_train = x_train.transpose(0, 2, 3, 1)
    x_train = x_train.transpose(0, 2, 3, 1)
    y_train = np.array(y_train)
    
    x_test = []
    y_test = []
    
    num_of_files_recorded = len(test_list)
    
    for j in range(num_of_files_recorded):
        image = cv2.resize(cv2.imread(test_list[j]), (224, 224)).astype(np.float32)
        image = image - vgg_mean
        image = image[:, ::-1] # RGB to BGR
        x_test.append(image)
        lists = test_list[j].split('/')
        label = lists[-2]
        y_test.append(label_map[label])
    
    x_test = np.array(x_test)
    x_test = x_test.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_test = np.array(y_test)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
        
        #plt.imshow(x_train[5])
        #plt.imshow(x_test[1225])

    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    
    if K.image_dim_ordering() == 'th':
        x_train = np.array([cv2.resize(img.transpose(1,2,0), (224,224)).transpose(2,0,1) for img in x_train[:,:,:,:]])
        x_test = np.array([cv2.resize(img.transpose(1,2,0), (224,224)).transpose(2,0,1) for img in x_test[:,:,:,:]])
    else:
        x_train = np.array([cv2.resize(img, (224,224)) for img in x_train[:,:,:,:]])
        x_test = np.array([cv2.resize(img, (224,224)) for img in x_test[:,:,:,:]])

    # Transform targets to keras compatible format
    y_train = np_utils.to_categorical(y_train[:], num_classes)
    y_test = np_utils.to_categorical(y_test[:], num_classes)
    
    return (x_train, y_train), (x_test, y_test)
    
# initialize the model
initial_model = VGG16(weights="imagenet", include_top=True)

#finetuning
x = Dense(num_classes, activation='softmax')(initial_model.layers[-2].output)
model = Model(initial_model.input, x)

# we freeze the other layers 
for layer in initial_model.layers: layer.trainable=False

opt = Adam(lr=0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=3,
#                validation_data=valid_batches, nb_val_samples=valid_batches.nb_sample)

print(model.summary())

X_train, Y_train, X_valid, Y_valid = load_data(FLAGS.dataset)

batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs_1

hist1 = model.fit(X_train, Y_train,
                 batch_size=batch_size,
                 nb_epoch=num_epochs,
                 shuffle=True,
                 verbose=1,
                 validation_data=(X_valid, Y_valid))

outputFileName = FLAGS.dataset + time.strftime("%Y%m%d-%H%M%S")
f = open(outputFileName + '.txt', 'w')
print >> f, hist1.history

for layer in model.layers[:10]:
   layer.trainable = False
for layer in model.layers[10:]:
   layer.trainable = True

opt = SGD(lr=10e-5)
model.compile(optimizer=opt,
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=20,
#                validation_data=valid_batches, nb_val_samples=valid_batches.nb_sample)

num_epochs = FLAGS.num_epochs_2

hist2 = model.fit(X_train, Y_train,
                 batch_size=batch_size,
                 nb_epoch=num_epochs,
                 shuffle=True,
                 verbose=1,
                 validation_data=(X_valid, Y_valid))

result = model.evaluate(X_valid, Y_valid, batch_size=batch_size, verbose=1)

result = np.array(result)
model.save("model_" + outputFileName + ".h5")
print >> f, hist2.history
print >> f, result
f.close()

#image = preprocess_image(path)
#prediction = model.predict(np.reshape(image, (1, 224, 224, 3)))
#print(prediction)
