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


def boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.concat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ], axis=-1)


def filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = tf.argmax(box_scores, axis=-1)
    box_class_scores = tf.reduce_max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.
    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.
    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(
        tf.Variable(anchors, dtype=tf.float32, name='anchors'),
        [1, 1, 1, num_anchors, 2])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = tf.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = tf.range(0, conv_dims[0])
    conv_width_index = tf.range(0, conv_dims[1])
    conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

    conv_width_index = tf.tile(tf.expand_dims(conv_width_index, 0),
                              [conv_dims[0], 1])
    conv_width_index = tf.reshape(tf.transpose(conv_width_index), [-1])
    conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
    conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = tf.cast(conv_index, feats.dtype)

    feats = tf.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), feats.dtype)

    box_xy = tf.nn.sigmoid(feats[..., :2])
    box_wh = tf.exp(feats[..., 2:4])
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.nn.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


def evaluate(yolo_outputs, image_shape, max_boxes=10, score_threshold=.6,
             iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = filter_boxes(boxes, box_confidence,
                                          box_class_probs,
                                          threshold=score_threshold)

    # Scale boxes back to original image shape.
    image_shape = tf.cast(image_shape, tf.float32)
    image_dims = tf.concat([image_shape, image_shape], axis=0)
    image_dims = tf.expand_dims(image_dims, 0)
    boxes = boxes * image_dims

    max_boxes_tensor = tf.Variable(max_boxes, dtype=tf.int32, name='max_boxes')
    nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor,
                                             iou_threshold=iou_threshold)
    boxes = tf.gather(boxes, nms_index)
    scores = tf.gather(scores, nms_index)
    classes = tf.gather(classes, nms_index)
    return tf.cast(tf.round(boxes), tf.int32), scores, classes
