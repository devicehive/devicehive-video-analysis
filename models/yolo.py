# Copyright (C) 2017 DataArt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf

from models.base import BaseModel
from utils import yolo, general


class YoloBaseModel(BaseModel):
    """Yolo base model class."""

    _anchors = None
    _labels = None

    def __init__(self, input_shape, checkpoint_path, score_threshold=.3,
                 iou_threshold=.4):
        self._checkpoint_path = checkpoint_path
        self._meta_graph_location = self._checkpoint_path+'.meta'
        self._input_shape = input_shape

        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._sess = None
        self._eval_ops = None

        self.colors = None

    @property
    def anchors(self):
        if self._anchors is None:
            raise AttributeError(
                '"{}" must define "_anchors"'.format(self.__class__.__name__))
        return self._anchors

    @property
    def labels(self):
        if self._labels is None:
            raise AttributeError(
                '"{}" must define "_labels"'.format(self.__class__.__name__))
        return self._labels

    def _evaluate(self, matrix):
        raw_inp = self._sess.graph.get_tensor_by_name('normalization/input:0')
        out = self._sess.graph.get_tensor_by_name('normalization/output:0')
        eval_inp = self._sess.graph.get_tensor_by_name('evaluation/input:0')

        # TODO: We can merge normalization with other OPs, but we need to
        # redefine input tensor for this. Anyway this works faster then
        # normalizing input data with python.
        normalized = self._sess.run(out, feed_dict={raw_inp: matrix})
        return self._sess.run(self._eval_ops, feed_dict={eval_inp: normalized})

    def init(self):
        self._sess = tf.Session()
        self.colors = general.generate_colors(len(self.labels))

        saver = tf.train.import_meta_graph(
            self._meta_graph_location, clear_devices=True,
            import_scope='evaluation'
        )
        saver.restore(self._sess, self._checkpoint_path)

        eval_inp = self._sess.graph.get_tensor_by_name('evaluation/input:0')
        eval_out = self._sess.graph.get_tensor_by_name('evaluation/output:0')
        
        with tf.name_scope('normalization'):
            inp = tf.placeholder(tf.float32, self._input_shape, name='input')
            inp = tf.image.resize_images(inp, eval_inp.get_shape()[1:3])
            inp = tf.expand_dims(inp, 0)
            tf.divide(inp, 255., name='output')

        with tf.name_scope('postprocess'):
            outputs = yolo.head(eval_out, self.anchors, len(self.labels))
            self._eval_ops = yolo.evaluate(
                outputs, self._input_shape[0:2],
                score_threshold=self._score_threshold,
                iou_threshold=self._iou_threshold)

        self._sess.run(tf.global_variables_initializer())

    def close(self):
        self._sess.close()

    def evaluate(self, matrix):
        boxes, scores, classes = self._evaluate(matrix)
        objects = []
        for num, box in enumerate(boxes):
            top, left, bottom, right = box
            objects.append({
                'box': {
                    'top': top,
                    'left': left,
                    'bottom': bottom,
                    'right': right
                },
                'score': scores[num],
                'class': classes[num],
                'class_name': self.labels[classes[num]],
                'color': self.colors[classes[num]]
            })
        return objects


class Yolo9kModel(YoloBaseModel):

    _anchors = [[0.77871, 1.14074], [3.00525, 4.31277], [9.22725, 9.61974]]

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('checkpoint_path', 'data/yolo9k/yolo9000_model.ckpt')
        self._names_path = kwargs.pop('names_path', 'data/yolo9k/yolo9k.names')

        super(Yolo9kModel, self).__init__(*args, **kwargs)

    def init(self):
        with open(self._names_path) as f:
            self._labels = f.read().splitlines()

        super(Yolo9kModel, self).init()
