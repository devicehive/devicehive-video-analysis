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


FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_float('score_threshold', .3, 'Score threshold.')
tf.flags.DEFINE_float('iou_threshold', .4, 'Intersection over union threshold.')


class YoloBaseModel(BaseModel):
    """Yolo base model class."""

    _checkpoint_path = None
    _names_path = None
    _anchors = None
    labels = None

    def __init__(self, input_shape):
        self._meta_graph_location = self._checkpoint_path+'.meta'
        self._input_shape = input_shape

        self._score_threshold = FLAGS.score_threshold
        self._iou_threshold = FLAGS.iou_threshold
        self._sess = None
        self._raw_inp = None
        self._raw_out = None
        self._eval_inp = None
        self._eval_ops = None

        self.colors = None

    def _evaluate(self, matrix):
        # TODO: We can merge normalization with other OPs, but we need to
        # redefine input tensor for this. Anyway this works faster then
        # normalizing input data with python or openCV or numpy.
        normalized = self._sess.run(self._raw_out,
                                    feed_dict={self._raw_inp: matrix})
        return self._sess.run(self._eval_ops,
                              feed_dict={self._eval_inp: normalized})

    def init(self):
        if bool(self.labels) == bool(self._names_path):
            raise AttributeError(
                'Model must define either "labels" or "names path" not both.')

        if self._names_path:
            with open(self._names_path) as f:
                self.labels = f.read().splitlines()

        if not self._anchors:
            raise AttributeError('Model must define "_anchors".')

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
            raw_inp = tf.placeholder(tf.float32, self._input_shape,
                                     name='input')
            inp = tf.image.resize_images(raw_inp, eval_inp.get_shape()[1:3])
            inp = tf.expand_dims(inp, 0)
            raw_out = tf.divide(inp, 255., name='output')

        with tf.name_scope('postprocess'):
            outputs = yolo.head(eval_out, self._anchors, len(self.labels))
            self._eval_ops = yolo.evaluate(
                outputs, self._input_shape[0:2],
                score_threshold=self._score_threshold,
                iou_threshold=self._iou_threshold)

        self._raw_inp = raw_inp
        self._raw_out = raw_out
        self._eval_inp = eval_inp

        self._sess.run(tf.global_variables_initializer())

    def close(self):
        self._sess.close()

    def evaluate(self, matrix):
        objects = []
        for box, score, class_id in zip(*self._evaluate(matrix)):
            top, left, bottom, right = box
            objects.append({
                'box': {
                    'top': top,
                    'left': left,
                    'bottom': bottom,
                    'right': right
                },
                'score': score,
                'class': class_id,
                'class_name': self.labels[class_id],
                'color': self.colors[class_id]
            })
        return objects


class Yolo2Model(YoloBaseModel):

    _checkpoint_path = 'data/yolo2/yolo_model.ckpt'
    _names_path = 'data/yolo2/yolo2.names'
    _anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
                [7.88282, 3.52778], [9.77052, 9.16828]]
