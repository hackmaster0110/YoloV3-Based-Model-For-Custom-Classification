from collections import namedtuple

""" 
This model is based on YoloV3, usage of MICROSOFT VOTT TOOL for annotating and creating tfrecords for both train and validating is recommended to ensure same format 
# CHANNELS (integer) 1 - for greyscale or 3 for Color image
# Size can be 416 or 320 or 608

class_names.txt must contain one class name in each line in same class number order

"""
configKeys = namedtuple('Constants', ['DATASET_PATH', 'VAL_DATASET', 'CLASSES_TXT',
                                      'SIZE', 'EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE', 'NUM_CLASSES', 'CHANNELS', 'WEIGHTS', 'YOLO_MAX_BOXES', 'YOLO_IOU_THRESHOLD', 'YOLO_SCORE_THRESHOLD', 'PATIENCE', 'VERBOSE', 'DEBUG', 'CLASS_MUTUALLY_EXCLUSIVE','TENSORBOARD_LOGS'])
Config = configKeys('path_train_tfrecords/', 'path_validation_tfrecords/', 'path_to_class_names.txt',
                    416, 1000, 8, 1E-3, 2, 1, './output/checkpoints/weights.tf', 50, 0.5, 0.5, 10, 1, False, True,'tensorboardlogs/')
