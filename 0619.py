import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

# Load datasets
data_train = tf.keras.utils.image_dataset_from_directory(
    'dataset/train', image_size=(256, 256), batch_size=32)
data_test = tf.keras.utils.image_dataset_from_directory(
    'dataset/test', image_size=(256, 256), batch_size=32)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# Normalize the data
def preprocess(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

data_train = data_train.map(preprocess)
data_test = data_test.map(preprocess)

# Extract data from tf.data.Dataset
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for img, label in dataset.unbatch():
        images.append(img.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

# Convert datasets to numpy arrays
X_train, y_train = dataset_to_numpy(data_train)
X_test, y_test = dataset_to_numpy(data_test)

# Setup KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True)

# Define the model using Depthwise Separable Convolution
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

# Perform KFold cross-validation
fold_no = 1
for train_index, val_index in kfold.split(X_train):
    train_data = tf.data.Dataset.from_tensor_slices((X_train[train_index], y_train[train_index])).batch(32)
    val_data = tf.data.Dataset.from_tensor_slices((X_train[val_index], y_train[val_index])).batch(32)
    
    model = create_model()
    
    logdir = f'logs/fold_{fold_no}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    hist = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])
    
    # Plotting the loss
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle(f'Loss for Fold {fold_no}', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    
    # Plotting the accuracy
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle(f'Accuracy for Fold {fold_no}', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    
    fold_no += 1

# Calculate computational complexity (FLOPs)
# Creating a model without data augmentation for FLOPs calculation
def create_model_no_augmentation():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

flops_model = create_model_no_augmentation()
flops_model.build(input_shape=(None, 256, 256, 3))

concrete_function = tf.function(lambda x: flops_model(x))
concrete_function = concrete_function.get_concrete_function(tf.TensorSpec([1, 256, 256, 3], tf.float32))

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(concrete_function)

# Get the graph definition
graph_def = frozen_func.graph.as_graph_def()

# Calculate FLOPs
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    print(f'Total FLOPs: {flops.total_float_ops}')

# Evaluate the model on the test dataset
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
for batch in test_data.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print('Precision:', pre.result().numpy())
print('Recall:', re.result().numpy())
print('Accuracy:', acc.result().numpy())
