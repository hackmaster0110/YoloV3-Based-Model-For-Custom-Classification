import time
import cv2
import numpy as np
import tensorflow as tf
from models import *
from Dataset import transform_images
from utils import draw_outputs
from settings import Config





def detectAndLocate(image_path,output_path,classes_file = Config.CLASSES_TXT):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3Tiny(classes=Config.NUM_CLASSES)

    yolo.load_weights(Config.WEIGHTS).expect_partial()
    class_names = [c.strip() for c in open(classes_file).readlines()]

    img_raw = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
    # img_to_draw = cv2.resize(cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR),(Config.SIZE,Config.SIZE))
    img_to_draw = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    if Config.CHANNELS == 1:
        img_raw = tf.image.rgb_to_grayscale(img_raw)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, Config.SIZE)

    # t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    # t2 = time.time()
    # print('time: {}'.format(t2 - t1))

    # print('detections:')
    # for i in range(nums[0]):
    #     print(('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
    #                                        np.array(scores[0][i]),
    #                                        np.array(boxes[0][i]))))

    img = draw_outputs(img_to_draw, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path, img)

