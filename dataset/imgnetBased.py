# -*- coding:utf-8 -*
"""
@version: 0.0
@Author by Ggao
@Mail: ggao_liming@qq.com
@File: imgnetBased.py
@time: 2017-07-29
"""
import tensorflow as tf
import os
import logging
import cv2
from tqdm import tqdm
import time

class DataReader(object):
    def __init__(self, annotation_file='', images_path='', tf_record_path=None):
        assert os.path.exists(annotation_file), 'label file not exit {0}'.format(annotation_file)
        assert os.path.exists(images_path), 'image path not exit {0}'.format(images_path)
        if tf_record_path is None:
            cur_dir_path = os.path.dirname(__file__)
            self.tf_record_path = os.path.join(cur_dir_path, 'TFRecodr')
            if not os.path.exists(self.tf_record_path):
                logging.info('create tf record save path {0}'.format(self.tf_record_path))
                os.mkdir(self.tf_record_path)
        else:
            self.tf_record_path = tf_record_path
        if os.path.isdir(self.tf_record_path):
            self.tf_record_path = os.path.join(self.tf_record_path, 'input_data.tfrecord')
        if not os.path.exists(self.tf_record_path):
            DataReader.create_tf_record_data(
                annotation_file, images_path, self.tf_record_path
            )
        logging.info('TF record path {0}'.format(self.tf_record_path))

    @staticmethod
    def create_tf_record_data(annotation_file, img_path, record_file, max_num=-1):
        """
        generate tf frecord file
        :param anno_file: annotation file 
        :param img_path: image store file
        :param record_file: tf record save path
        :param max_num: max image number, -1 for all images
        :return: 
        """
        logging.info('Start to generate record file {0}'.format(record_file))
        writer = tf.python_io.TFRecordWriter(record_file)
        annotation_lines = open(annotation_file).readlines()
        start_time = time.time()
        for a_index in tqdm(range(len(annotation_lines)), desc='Transforming Image Data'):
            anno = annotation_lines[a_index]
            if max_num != -1 and max_num < a_index:
                break
            anno = anno.strip().split()
            img_name = anno[0].split('/')[-1]
            img_name = os.path.join(img_path, img_name)
            img_type = anno[1]
            img = cv2.imread(img_name)
            if img.shape[0] == 0 or img.shape[1] == 0 or img.shape[2] != 3:
                continue
            img = cv2.resize(img, (224, 224))
            img_raw = img.tobytes()
            label = int(img_type)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })
            )
            writer.write(example.SerializeToString())
        writer.close()
        logging.info('Image Data transform finish time cost:{0}'.format(time.time() - start_time))
        pass

    def read_and_decode(self):
        """
        load tf record in queue
        :return: 
        """
        logging.info('Read data form path {0}'.format(self.tf_record_path))
        filename_queue = tf.train.string_input_producer([self.tf_record_path])
        reader = tf.TFRecordReader()
        _, example = reader.read(filename_queue)
        features = tf.parse_single_example(
            example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)
            }
        )
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, (224, 224, 3))
        img = tf.cast(img, tf.float32) # * (1. / 255) - 0.5
        label = tf.cast(features['label'], tf.int32)
        logging.info('Load data info {0} {1}'.format(img, label))
        return img, label

    def get_batch_data(self, batch_size=5, num_threads=4, capacity=5000, min_deque=1000):
        """
        generate train batch data from tf queue data
        :param batch_size: 
        :param num_threads: 
        :param capacity: 
        :param min_deque: 
        :return: 
        """
        if min_deque > capacity:
            min_deque = int(capacity/10)
        single_img, single_label = self.read_and_decode()
        img_batch, label_batch = tf.train.shuffle_batch(
            [single_img, single_label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_deque
        )
        logging.info('Batch data info {0} {1}'.format(img_batch, label_batch))
        return img_batch, label_batch
    pass
