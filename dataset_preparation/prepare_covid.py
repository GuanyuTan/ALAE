# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Create a tfrecords for MNIST. """
import sys
import os
sys.path.append(os.getcwd())
import random
import tensorflow as tf
import tqdm
from defaults import get_cfg_defaults
from utils import get_main_directory
from subprocess import call
import imageio



def compile(cfg, logger):
    # initial test using part of the data I have. all of covid and half of normal ct scans
    # thinking of what is the best way to do so
    # I'll generate train and test files by iterating through the original files
    data_path = os.path.join(get_main_directory(), "data/")
    dataset_name = "COVID-19_Radiography_Dataset"
    data_list = ["COVID", "Normal"]
    count = []
    if not os.path.exists(data_path):
        print("------Downloading Dataset from Kaggle------")
        os.system(f"kaggle datasets download tawsifurrahman/covid19-radiography-database -p {data_path} --unzip")
    if not os.path.exists(os.path.dirname(cfg.DATASET.PATH)):
        os.makedirs(os.path.dirname(cfg.DATASET.PATH))
    for data_dir in data_list:
        root, dirs, files = next(
            os.walk(os.path.join(data_path, dataset_name, data_dir), topdown=False))
        count.append(len(files))
        random.shuffle(files)
        files = files[:min(count)]
        images = []
        for file in tqdm.tqdm(files):
            image = imageio.imread(root + "/" + file)
            # put images in dict
            images.append(image)
        split = 0.85
        train, test = images[:int(len(images)*0.85)
                             ], images[int(len(images)*0.85):]

        part_path = cfg.DATASET.PATH % 299
        tfr_writer = tf.io.TFRecordWriter(part_path)
        for image in train:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))
            }))
            tfr_writer.write(ex.SerializeToString())
        tfr_writer.close()

        part_path = cfg.DATASET.PATH_TEST % 299
        tfr_writer = tf.io.TFRecordWriter(part_path)
        for image in test:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))
            }))
            tfr_writer.write(ex.SerializeToString())
        tfr_writer.close()


im_size = 256
cfg = get_cfg_defaults()
compile(cfg, None)
# directory = os.path.dirname(path)

# os.makedirs(directory, exist_ok=True)

# # setting folds = 1 for now, not using k-cross fold
# folds = 1
# # get root and file names for train


# im = 0
# i = 0
# # iterate through filenames
# for file in tqdm.tqdm(files):
#     image = imageio.imread(root+"/"+file)
#     # put images in dict
#     images.append((1 if ("COVID" in file) else 0,image))
#     # record label
#     labels.append(1 if ("COVID" in file) else 0)
#
# tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
#
# part_path = cfg.DATASET.PATH % (256, 0)
#
# tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
# random.shuffle(images)
# for label, image in images:
#     ex = tf.train.Example(features=tf.train.Features(feature={
#         'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
#         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
#         'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))
#         }))
#     tfr_writer.write(ex.SerializeToString())
# tfr_writer.close()
