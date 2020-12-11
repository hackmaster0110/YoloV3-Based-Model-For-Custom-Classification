
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from models import *
import Dataset as dataset
from settings import Config
from matplotlib import pyplot as plt
from IPython.display import clear_output
import seaborn as sns

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        sns.lineplot(self.x, self.losses, label="loss")
        sns.lineplot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()
        
PLOT_LOSSES = PlotLosses()


def main(dataset_path=Config.DATASET_PATH,val_dataset=Config.DATASET_PATH,classes=Config.CLASSES_TXT,size=Config.SIZE,epochs=Config.EPOCHS,batch_size=Config.BATCH_SIZE,learning_rate=Config.LEARNING_RATE,num_classes=Config.NUM_CLASSES):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    model = YoloV3Tiny(size, training=True,classes=num_classes)

    train_dataset = dataset.load_tfrecord_dataset(dataset_path, classes, size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(lambda x, y: (dataset.transform_images(x, size),dataset.transform_targets(y, yolo_tiny_anchors, yolo_tiny_anchor_masks, size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_tfrecord_dataset(val_dataset, classes, size)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, yolo_tiny_anchors, yolo_tiny_anchor_masks, size)))


    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = [YoloLoss(yolo_tiny_anchors[mask], classes=num_classes)
            for mask in yolo_tiny_anchor_masks]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(Config.DEBUG))

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=Config.PATIENCE, verbose=Config.VERBOSE),
        ModelCheckpoint(Config.WEIGHTS,
                        verbose=Config.VERBOSE, save_best_only=True,save_weights_only=True),
        TensorBoard(log_dir=Config.TENSORBOARD_LOGS),
        PLOT_LOSSES
    ]

    history = model.fit(train_dataset,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=val_dataset)

if __name__ == '__main__':
    main()
