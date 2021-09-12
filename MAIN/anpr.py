import datetime
from flask import request
from flask import Flask, render_template, Response
app = Flask(__name__)
import pyrebase
import os
import tensorflow as tf
import cv2
from PIL import ImageGrab
import numpy as np
import easyocr
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

home = 'A:\projects\ANPR'  # Project Directory
os.chdir(home)
CUSTOM_MODEL_NAME = 'my_ssd_mobnet_model'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
path_to_images = os.path.join('Datasets', 'licence')

paths ={
    'WORKSPACE_PATH': os.path.join('Tensorflow'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models','tfmodels'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'annotations'),
    'IMAGE_PATH': path_to_images,
    'MODEL_PATH': os.path.join('Tensorflow','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'models','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-7')).expect_partial()


# -----------------MAIN----------------- #
detection_threshold = 0.7
region_threshold = 0.6


# FILTER TEXT
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]

    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


# OCR FUNCTION
def ocr_it(image, detections, detection_threshold, region_threshold):
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)

        text = filter_text(region, ocr_result, region_threshold)

        print(text)
        return text


# SAVE IMAGE
def save_results(text):

    for i in text:
        i=i.replace(" ", "")
        img_name = '{}.txt'.format(i)
        img_name = os.path.join(home, "MAIN", "Detected", img_name)
        with open(img_name, 'a') as t:
            t.write(str(datetime.datetime.now())+"\n")


print(os.getcwd())


config = {
    'apiKey': "AIzaSyAt9WY7A1woCyXEXu7UvTTgSlCan2caEOw",
    'authDomain': "vehicle-tega.firebaseapp.com",
    'databaseURL': "https://vehicle-tega-default-rtdb.firebaseio.com",
    'projectId': "vehicle-tega",
    'storageBucket': "vehicle-tega.appspot.com",
    'messagingSenderId': "288424776498",
    'appId': "1:288424776498:web:f27430dce1105e7bb33d2b",
    'measurementId': "G-35Q4WN14B3",
    "serviceAccount": "A:\projects\ANPR\MAIN\service.json"
    }


fire_base = pyrebase.initialize_app(config)
auth = fire_base.auth()
database = fire_base.database()


def add_new_plate(new):
    print(new)
    data= {'Name':new['Name'], 'Phone':new['Phone'], 'Address':new['Address']}
    database.child(new["number"]).set(data)


def get_number_info(num):
    value = database.get().val()
    value = value[num]
    return value['Name'], value['Address'], value['Phone']


cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def gen_frames():  # generate frame by frame from camera
    while cap.isOpened():
    ##while True:
        ret, frame = cap.read()
        ##frame =  ImageGrab.grab(bbox=(0, 100, 500,500))
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

        try:
            text =ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
            save_results(text)

        except Exception as e:
            print(e)

        ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/get_info', methods=['POST', 'GET'])
def get_info():
    try:
        number = str(request.args.get('myfile'))
        number = number.split('.')[0]
        name, addr, phone = get_number_info(number)

        return '''<html><head><title>PLATE INFO</title></head><body> 
        <h1>NAME: {}</h1> 
        <h1>NUMBER: {}</h1> 
        <h1>ADDRESS: {}</h1> 
        <h1>PHONE: {}</h1> 
        </body></html>'''.format(name, number, addr, phone)
    except:
        pass


@app.route('/add_new', methods=['POST', 'GET'])
def add_new():
    new = {"Phone":request.form["Phone"], "Address":request.form['Address'], "Name":request.form["Name"],"number":request.form["number"] }
    add_new_plate(new)
    return "ADDED!"


@app.route('/anpr')
def anpr():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)