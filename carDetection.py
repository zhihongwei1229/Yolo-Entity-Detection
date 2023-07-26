from PIL import Image
from keras import backend as K
from keras.models import load_model
from utils import (read_classes, read_anchors, yolo_eval, predict)
from model.model import yolo_head


def image_resize(image_file_name, new_image_file_name):
    target_image = Image.open("images/" + image_file_name)
    if target_image.size != (1280, 720):
        target_image = target_image.resize((1280, 720))
    target_image.save("images/" + new_image_file_name)


def image_rotate(image_file_name, new_image_file_name, rotate_angle):
    target_image = Image.open("images/" + image_file_name)
    target_image = target_image.rotate(rotate_angle)
    target_image.save("images/" + new_image_file_name)


session = K.get_session()
class_names = read_classes("source/coco_classes.txt")
anchors = read_anchors("source/yolo_anchors.txt")

image_shape = (720., 1280.)
yolo_model = load_model("source/yolo.h5")
# yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

image = Image.open('images/car_sample.jpeg')
# file_name = 'car_sample.jpeg'
# processed_file_name = 'car_sample_processed.jpeg'
# image_resize(file_name, processed_file_name)
# image_rotate(processed_file_name, processed_file_name, -90)

out_scores, out_boxes, out_classes = predict(session, yolo_model, scores, boxes, classes, class_names, "test.jpg")
# out_scores, out_boxes, out_classes = predict(
# session, yolo_model, scores, boxes, classes, class_names, processed_file_name)


