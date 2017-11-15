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


import argparse
import logging.config
import cv2
import time

from models import yolo
from log_config import LOGGING


logging.config.dictConfig(LOGGING)
logger = logging.getLogger('detector')


def find_class_by_name(name, modules):
    modules = [getattr(m, name, None) for m in modules]
    return next(c for c in modules if c)


def evaluate(model_name, video, **model_kwargs):
    win_name = 'Detector'
    cv2.namedWindow(win_name)

    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        logger.error('Can\'t open "{}"'.format(video))
        return

    source_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    source_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

    model_cls = find_class_by_name(model_name, [yolo])
    model = model_cls(input_shape=(source_h, source_w, 3), **model_kwargs)
    model.init()

    frame_num = 0
    start_time = time.time()
    fps = 0
    try:
        while True:
            ret, frame = cam.read()

            if not ret:
                logger.warning('Can\'t read video data')
                continue

            predictions = model.evaluate(frame)

            for o in predictions:
                x1 = o['box']['left']
                x2 = o['box']['right']

                y1 = o['box']['top']
                y2 = o['box']['bottom']

                color = o['color']
                class_name = o['class_name']

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                (test_width, text_height), _ = cv2.getTextSize(
                    class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(frame, (x1, y1), (x1+test_width, y1-text_height),
                              color, thickness=cv2.FILLED)
                cv2.putText(frame, class_name, (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            end_time = time.time()
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time

            cv2.putText(frame, 'FPS: {:.2f}'.format(fps),
                        (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 0), 1)

            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_num += 1
            logger.info('Frame: {}, FPS: {:.2f}'.format(frame_num, fps))

    finally:
        cv2.destroyAllWindows()
        cam.release()
        model.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='CLI for video evaluation.')
    ap.add_argument('-v', '--video', default=0, help='Path to the video file.')
    ap.add_argument('-m', '--model_name', default='Yolo9kModel',
                    metavar='MODEL', help='Model name to use.')

    evaluate(**vars(ap.parse_args()))
