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


import cv2
import json
import time
import threading
import logging.config
from Tkinter import *
from devicehive_webconfig import Server, Handler

from models.yolo import Yolo2Model
from utils.general import format_predictions, format_notification, format_person_prediction
from web.routes import routes
from log_config import LOGGING

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('detector')


class DeviceHiveHandler(Handler):

    _device = None

    def handle_connect(self):
        self._device = self.api.put_device(self._device_id)
        self._device.send_notification("Begin", {"Initial":"start"})
        super(DeviceHiveHandler, self).handle_connect()

    def send(self, data):
        if isinstance(data, str):
            notification = data
        else:
            try:
                notification = json.dumps(data)
            except TypeError:
                notification = str(data)

        self._device.send_notification("predictions", {"notifications":notification})


class Daemon(Server):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, cv2.COLOR_LUV2LBGR]

    _detect_frame_data = None
    _detect_frame_data_id = None
    _cam_thread = None
    _surgery_meta = None

    def __init__(self, *args, **kwargs):
        super(Daemon, self).__init__(*args, **kwargs)
        self._detect_frame_data_id = 0
        self._cam_thread = threading.Thread(target=self._cam_loop, name='cam')
        self._cam_thread.setDaemon(True)
        # self._surgery_meta = initialData

    def _on_startup(self):
        self._cam_thread.start()

    def _cam_loop(self):
        logger.info('Start camera loop')
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise IOError('Can\'t open "{}"'.format(0))

        source_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        source_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

        model = Yolo2Model(input_shape=(source_h, source_w, 3))
        model.init()

        start_time = time.time()
        frame_num = 0
        fps = 0
        try:
            while self.is_running:
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

                self._detect_frame_data_id = frame_num
                _, img = cv2.imencode('.jpg', frame, self.encode_params)
                self._detect_frame_data = img

                if predictions:
                    formatted = format_predictions(predictions)
                    logger.info('Predictions: {}'.format(formatted))
                    self._send_dh(format_person_prediction(predictions))

                frame_num += 1

        finally:
            cam.release()
            model.close()

    def _send_dh(self, data):
        if not self.dh_status.connected:
            logger.error('Devicehive is not connected')
            return
        print("sending")
        self.deviceHive.handler.send(data)

    def get_frame(self):
        return self._detect_frame_data, self._detect_frame_data_id


class Widget():
    
    # self.hospital_field = None
    # self.doctor_field = None
    # self.patient_field = None
    # self.procedure_field = None
    # self.instrument_packets_field = None

    def __init__(self):
        # self.server = None

        self.window = Tk()
        self.window.title("Assist-MD Capture")
        self.window.geometry('500x200')

        self.hospital_field = "Memorial Hospital"
        self.doctor_field = "Dr. Humphreys"
        self.patient_field = "David Roster"
        self.procedure_field = "Appendectomy"
        self.instrument_packets_field = 1

        self.hospital_lbl = Label(self.window, text="Hospital Name: ")
        self.hospital_lbl.grid(column=0, row=0)
        v = StringVar(self.window, value=self.hospital_field)
        self.hospital_txt = Entry(self.window,textvariable=v,width=20)
        self.hospital_txt.grid(column=1, row=0)

        self.doctor_lbl = Label(self.window, text="Doctor's Name: ")
        self.doctor_lbl.grid(column=0, row=1)
        v = StringVar(self.window, value=self.doctor_field)
        self.doctor_txt = Entry(self.window,textvariable=v,width=20)
        self.doctor_txt.grid(column=1, row=1)

        self.patient_lbl = Label(self.window, text="Patient's Name: ")
        self.patient_lbl.grid(column=0, row=2)
        v = StringVar(self.window, value=self.patient_field)
        self.patient_txt = Entry(self.window,textvariable=v,width=20)
        self.patient_txt.grid(column=1, row=2)

        self.procedure_lbl = Label(self.window, text="Procedure: ")
        self.procedure_lbl.grid(column=0, row=3)
        v = StringVar(self.window, value=self.procedure_field)
        self.procedure_txt = Entry(self.window,textvariable=v,width=20)
        self.procedure_txt.grid(column=1, row=3)

        self.instruments_lbl = Label(self.window, text="Number of Packets: ")
        self.instruments_lbl.grid(column=0, row=4)
        v = StringVar(self.window, value=self.instrument_packets_field)
        self.instruments_txt = Entry(self.window,textvariable=v,width=20)
        self.instruments_txt.grid(column=1, row=4)

        self.startButtonState = ACTIVE
        self.stopButtonState = DISABLED
        
        self.start_btn = Button(
            self.window, 
            text="Start Capture", 
            command=self.startClicked,
            state=self.startButtonState)
        self.start_btn.grid(column=2, row=0)

        self.stop_btn = Button(
            self.window, 
            text="Stop Capture", 
            command=self.stopClicked,
            state=self.stopButtonState)
        self.stop_btn.grid(column=2, row=1)

    def startClicked(self):
        print("starting server")
        self.hospital_field = self.hospital_txt.get()
        self.doctor_field = self.doctor_txt.get()
        self.patient_field = self.patient_txt.get()
        self.procedure_field = self.patient_txt.get()
        self.instrument_packets_field = self.instruments_txt.get()

        Initial = {
            "type": "start",
            "data": {
                "hospital": self.hospital_field,
                "doctor": self.doctor_field,
                "patient": self.patient_field,
                "procedure": self.procedure_field,
                "packets": self.instrument_packets_field
            }
        }

        self.stop_btn["state"] = ACTIVE
        self.start_btn["state"] = DISABLED

        self.server = Daemon(DeviceHiveHandler, routes=routes, is_blocking=False)
        self.server.start()

        while not self.server.dh_status.connected:
            # Wait till DH connection is ready
            time.sleep(.001)
            
        self.server.deviceHive.handler.send(Initial)
        

    def stopClicked(self):
        self.stop_btn["state"] = DISABLED
        self.start_btn["state"] = ACTIVE
        print('stop clicked')
        self.server.stop()
        self.window.destroy()

    def create_widget(self):
    
        self.window.mainloop()

if __name__ == '__main__':
    prog = Widget()
    prog.create_widget()
    
