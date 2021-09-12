import os
import wget

home = 'A:\projects\ANPR'  # Project Directory
os.chdir(home)
labels = [{'name': 'licence', 'id': 1}]
CUSTOM_MODEL_NAME = 'my_ssd_mobnet_model'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
path_to_images = os.path.join('Datasets', 'licence')
paths = {

    'WORKSPACE_PATH': os.path.join('Tensorflow'),
    'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models', 'tfmodels'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'annotations'),
    'IMAGE_PATH': path_to_images,
    'MODEL_PATH': os.path.join('Tensorflow', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'models', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('Tensorflow', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('Tensorflow', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('Tensorflow', 'protoc')
}

files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

for path in paths.values():
    if not os.path.exists(path):
        os.system('mkdir {}'.format(path))

if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    os.system('git clone https://github.com/tensorflow/models {}'.format(paths['APIMODEL_PATH']))

# PROTOC and TensorFlow Object Detection Download
url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"

# Download protoc
wget.download(url)
# Move and unzip
os.system('move protoc-3.15.6-win64.zip {}'.format(paths['PROTOC_PATH']))
os.system('cd {} && tar -xf protoc-3.15.6-win64.zip'.format(paths['PROTOC_PATH']))
os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))

os.system(
    'cd Tensorflow/models/tfmodels/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install')
os.system('cd Tensorflow/models/tfmodels/research/slim && pip install -e .')

VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders',
                                   'model_builder_tf2_test.py')
# Verify Installation
os.system('pip install tensorflow && python {}'.format(VERIFICATION_SCRIPT))

wget.download(PRETRAINED_MODEL_URL)
os.system('move {} {}'.format(PRETRAINED_MODEL_NAME + '.tar.gz', paths['PRETRAINED_MODEL_PATH']))
os.system('cd {} && tar -zxvf {}'.format(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))

# create label map

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
    f.write('\tname:\'{}\'\n'.format(label['name']))
    f.write('\tid:{}\n'.format(label['id']))
    f.write('}\n')

# create tfrecords (test and train data) using nicknochnack script
os.system('git clone https://github.com/nicknochnack/GenerateTFRecord {}'.format(paths['SCRIPTS_PATH']))
os.system("python {} -x {} -l {} -o {}".format(files['TF_RECORD_SCRIPT'], os.path.join(paths['IMAGE_PATH'], 'train'),
                                               files['LABELMAP'],
                                               os.path.join(paths['ANNOTATION_PATH'], 'train.record')))
os.system("python {} -x {} -l {} -o {}".format(files['TF_RECORD_SCRIPT'], os.path.join(paths['IMAGE_PATH'], 'test'),
                                               files['LABELMAP'],
                                               os.path.join(paths['ANNOTATION_PATH'], 'test.record')))

# copy model config file from downloded pre-trained model intomy own model folder
os.system('copy ' + os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME,
                                 'pipeline.config') + " " + os.path.join(paths['CHECKPOINT_PATH']))

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# get my model config and edit it: Update Config For Transfer Learning
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
print(config)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# editing
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME,
                                                                 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
    os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
    os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

# saving
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    print('NEW CONFIG:', config_text)
    f.write(config_text)

# TRAIN THE MODEL
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=30000".format(TRAINING_SCRIPT,
                                                                                              paths['CHECKPOINT_PATH'],
                                                                                              files['PIPELINE_CONFIG'])
os.chdir(home)
os.system(command)