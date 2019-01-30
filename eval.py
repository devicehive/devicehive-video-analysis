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


import time
import logging.config
import cv2
import pafy
import tensorflow as tf
import os

from models import yolo
from log_config import LOGGING
from utils.general import format_predictions, find_class_by_name, is_url

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('detector')
FLAGS = tf.flags.FLAGS


def evaluate(_):
    win_name = 'Detector'
    cv2.namedWindow(win_name)

    video = FLAGS.video

    if is_url(video):
        videoPafy = pafy.new(video)
        video = videoPafy.getbest(preftype="mp4").url
        cam = cv2.VideoCapture(video)
    elif os.path.isfile(video):
        cam = cv2.VideoCapture(video)
    else:
        cam = cv2.VideoCapture(0)
		
    if not cam.isOpened():
        raise IOError('Can\'t open "{}"'.format(FLAGS.video))

    source_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    source_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

    model_cls = find_class_by_name(FLAGS.model_name, [yolo])
    model = model_cls(input_shape=(source_h, source_w, 3))
    model.init()

    frame_num = 0
    start_time = time.time()
    fps = 0
    try:
        while True:
            ret, frame = cam.read()

            if not ret:
                logger.info('Can\'t read video data. Potential end of stream')
                return

            predictions = model.evaluate(frame)

            for o in predictions:
                x1 = o['box']['left']
                x2 = o['box']['right']

                y1 = o['box']['top']
                y2 = o['box']['bottom']

                color = o['color']
                class_name = o['class_name']

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                (test_width, text_height), baseline = cv2.getTextSize(
                    class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
                cv2.rectangle(frame, (x1, y1),
                              (x1+test_width, y1-text_height-baseline),
                              color, thickness=cv2.FILLED)
                cv2.putText(frame, class_name, (x1, y1-baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

            end_time = time.time()
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time

            # Draw additional info
            frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
            cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            logger.info(frame_info)

            cv2.imshow(win_name, frame)

            if predictions:
                logger.info('Predictions: {}'.format(
                    format_predictions(predictions)))

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord('q'):
                break

            # Take screenshot
            if key == ord('s'):
                cv2.imwrite('frame_{}.jpg'.format(time.time()), frame)

            frame_num += 1

    finally:
        cv2.destroyAllWindows()
        cam.release()
        model.close()


if __name__ == '__main__':
    tf.flags.DEFINE_string('video', '0', 'Path to the video file.')
    tf.flags.DEFINE_string('model_name', 'Yolo2Model', 'Model name to use.')

    tf.app.run(main=evaluate)
