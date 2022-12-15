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

import imageio
from defaults import get_cfg_defaults
import sys
import logging
from net import *
import numpy as np
import argparse
import os
import tensorflow as tf
import random
import dlutils
import pandas as pd
import shutil
import torch
import cv2

def compile(cfg, logger):
    # initial test using part of the data I have. all of covid and half of normal ct scans
    # thinking of what is the best way to do so
    # I'll generate train and test files by iterating through the original files
    data_path = "~/data/fyp_covid"
    covid_metadata = pd.read_csv(data_path+"/COVID.metadata.csv")
    normal_metadata = pd.read_csv(data_path+"/Normal.metadata.csv")

    # Imma just gonna compile it manually
    # First stack them together
    metadata = pd.concat([covid_metadata,normal_metadata.sample(frac=0.5)])
    metadata["Path"] = np.where(metadata["FILE NAME"].str.contains("COVID"),)
    metadata["class"] =np.where(metadata["FILE NAME"].str.contains("COVID"),"covid","normal")
    metadata = metadata.sample(frac=1).reset_index(drop=True)
    train_df = metadata.sample(frac=0.7).reset_index(drop=True)
    condition = metadata["FILE NAME"].isin(train_df["FILE NAME"])
    test_df = metadata.drop(metadata[condition].index)
    train_df.to_csv("/home/jaws/data/fyp_covid/data_1/train.csv")
    test_df.to_csv("/home/jaws/data/fyp_covid/data_1/test.csv")
    if not os.path.isdir("/home/jaws/data/fyp_covid/data_1"):
        # create our first data directory
        train_dir = "/home/jaws/data/fyp_covid/data_1/train"
        test_dir = "/home/jaws/data/fyp_covid/data_1/test"
        os.mkdir("/home/jaws/data/fyp_covid/data_1")
        os.mkdir("/home/jaws/data/fyp_covid/data_1/train")
        os.mkdir("/home/jaws/data/fyp_covid/data_1/test")
        for file_name in train_df["FILE NAME"]:
            if "COVID" in file_name:
                shutil.copyfile("/home/jaws/data/fyp_covid/COVID"+file_name, train_dir)
            if "NORMAL" in file_name:
                shutil.copyfile("/home/jaws/data/fyp_covid/COVID"+file_name, train_dir)
        for file_name in test_df["FILE NAME"]:
            if "COVID" in file_name:
                shutil.copyfile("/home/jaws/data/fyp_covid/COVID"+file_name, test_dir)
            if "NORMAL" in file_name:
                shutil.copyfile("/home/jaws/data/fyp_covid/COVID"+file_name, test_dir)

    

def prepare_covid(cfg, logger, train):
    im_size = 256

    if train:
        path = cfg.DATASET.PATH
    else:
        path = cfg.DATASET.PATH_TEST

    directory = os.path.dirname(path)

    os.makedirs(directory, exist_ok=True)

    # setting folds = 1 for now
    folds = 1

    if not train:
        folds = 1

    image_folds = [[] for _ in range(folds)]

    root, dirs, files = next(os.walk("/home/jaws/data/fyp_covid/data_1/train", topdown=False))
    count = len(files)

    count_per_fold = count // folds

    names = []
    labels = []
    images = {}
    im = 0
    i = 0
    for file in tqdm.tqdm(files):
        image = imageio.imread(root+"/"+file)
        images[file] = image
        labels.append(1 if ("COVID" in file) else 0)
        im+=1
        if im == len(files):
            output = open(f"covid_data_fold_{i}.pkl","wb")
            pickle.dump(list(images.values()),output)
            i+=1
            im = 0
            images.clear()
    if train:
        part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL, 0)
    else:
        part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL, 0)
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
    tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)

    random.shuffle(images)

    for label, image in images:
        ex = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
        tfr_writer.write(ex.SerializeToString())
    tfr_writer.close()

    

        if train:
            for j in range(3):
                images_down = []

                for image, label in zip(images, labels):
                    h = image.shape[1]
                    w = image.shape[2]
                    image = torch.tensor(np.asarray(image, dtype=np.float32)).view(1, 1, h, w)

                    image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8)

                    image_down = image_down.view(1, h // 2, w // 2).numpy()
                    images_down.append(image_down)

                part_path = cfg.DATASET.PATH % (5 - j - 1, i)
                tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
                for image, label in zip(images_down, labels):
                    ex = tf.train.Example(features=tf.train.Features(feature={
                        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
                    tfr_writer.write(ex.SerializeToString())
                tfr_writer.close()

                images = images_down


def run():
    parser = argparse.ArgumentParser(description="ALAE. prepare mnist")
    parser.add_argument(
        "--config-file",
        default="configs/mnist.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    random.seed(0)

    dlutils.download.mnist()
    mnist = dlutils.reader.Mnist('mnist', train=True, test=False).items
    random.shuffle(mnist)

    mnist_images = np.stack([x[1] for x in mnist])
    mnist_labels = np.stack([x[0] for x in mnist])

    prepare_mnist(cfg, logger, mnist_images, mnist_labels, train=False)
    prepare_mnist(cfg, logger, mnist_images, mnist_labels, train=True)


if __name__ == '__main__':
    run()
