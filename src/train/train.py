import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory


def train_test_split(datadir, img_width=224, img_height=224, val_split=.3, seed=42):
    train = image_dataset_from_directory(
        datadir,
        image_size=(img_width, img_height),
        validation_split=val_split,
        label_mode='categorical',
        subset='training',
        batch_size=32,
        seed=seed
    )
    val = image_dataset_from_directory(
        datadir,
        image_size=(img_width, img_height),
        validation_split=val_split,
        label_mode='categorical',
        subset='training',
        batch_size=32,
        seed=seed
    )
    return train, val


DEF_IMG_SIZE = {
    'Xception': (299, 299),
    'ResNet': (224, 224),
    'MobileNet': (224, 224),
    'Inception': (299, 299),
    'DenseNet': (224, 224)
}

APPLICATIONS = {
    'Xception': keras.applications.Xception,
    'ResNet': keras.applications.ResNet152V2,
    'MobileNet': keras.applications.MobileNetV3Small,
    'Inception': keras.applications.InceptionV3,
    'DenseNet': keras.applications.DenseNet121
}


class TransferLearningNet():

    def __init__(
        self,
        model_name,
        n_classes) -> None:
        self.model_builder = APPLICATIONS[model_name]
        self.n_classes = n_classes
        # self.img_size = img_size
        self.img_size = DEF_IMG_SIZE[model_name]

    # TODO properties
    
    # TODO add a customized interface to add custom layers
    def build(self,
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.Recall()]):
        model = self.model_builder(
            include_top=False,
            input_shape=self.img_size
        )
        flat = keras.layers.Flatten()(model.layers[-1].output)
        classification = keras.layers.Dense(1024, activation='relu')(flat)
        output = keras.layers.Dense(self.n_classes, activation='softmax')(classification)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, restore_best_weights=True),
            keras.callbacks.TensorBoard(),
            keras.callbacks.CSVLogger('training.log')
        ]
        
        model = keras.models.Model(
            inputs=model.inputs,
            output=output
        )
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        return model
    

def build_tl_2(
    model_builder,
    n_classes,
    train,
    val,
    logfile='training.csv',
    img_width=224,
    img_height=224,
    n_channels=3,
    epochs=10,
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[
        keras.metrics.CategoricalAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall()]):
    model = model_builder(
        include_top=False,
        input_shape=(
            img_width,
            img_height,
            n_channels))
    flat = keras.layers.Flatten()(model.layers[-1].output)
    classification = keras.layers.Dense(
        1024, activation='relu')(flat)
    output = keras.layers.Dense(
        n_classes, activation='softmax')(classification)
    model = keras.models.Model(
        inputs=model.inputs,
        outputs=output)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',
            min_delta=0.01,
            patience=round(epochs/10),
            restore_best_weights=True),
        keras.callbacks.TensorBoard(),
        keras.callbacks.CSVLogger(logfile)
    ]
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
    history = model.fit(
        train,
        epochs=epochs,
        batch_size=8,
        callbacks=callbacks,
        validation_data=val)
    return model, history


def build_transfer_learning(
    model_builder,
    n_classes,
    train_data,
    val_data,
    img_width=224,
    img_height=224,
    channels=3,
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()]):
    data_augmentation = keras.Sequential(
        [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.1),]
    )
    base_model = model_builder(
        weights='imagenet',
        input_shape=(img_width, img_height, channels),
        include_top=False
    )
    base_model.trainable=False
    # TODO add data augmentation
    inputs = tf.keras.Input(shape=(img_width, img_height, channels))
    x = data_augmentation(inputs)  # Apply random data augmentation
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(n_classes)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    epochs = 20         # TODO parametrize
    history = model.fit(train_data, epochs=epochs, validation_data=val_data)
    return base_model, model, history


def build_fine_tuning(
    base_model,
    model,
    train, 
    val_data,
    epochs=10,
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()]):
    base_model.trainable = True
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    history = model.fit(train, epochs=epochs, validation_data=val_data)
    return model, history
