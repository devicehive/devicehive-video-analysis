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

from subprocedure_sequence import demo_instruments
from models.yolo import Yolo2Model
from utils.general import format_predictions, format_notification, format_person_prediction
from web.routes import routes
from log_config import LOGGING

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('detector')


class DeviceHiveHandler(Handler):

    _device = None
    _surgery_meta = None
    _payload = None

    def handle_connect(self):
        self._device = self.api.put_device(self._device_id)
        super(DeviceHiveHandler, self).handle_connect()

    def send(self, data):
        if data["type"] == "start":
            self._surgery_meta = data["data"]
            return

        data = {
            "meta": self._surgery_meta,
            "data": data
        }
        print("NFDJKF:NDKFNDK:FN:")
        print(data['data']["predictions"]["AR-10000"])
        if isinstance(data, str):
            notification = data
        # else:
            # notification = json.dumps(data, encoding='UTF-8')
            # try:
            #     print("JSON")
            #     notification = json.dumps(data)
            # except TypeError:
            #     print("STRING")
            #     notification = str(data)

        print("note: ", type(data))
        print("note data: ", type(data["data"]))# ["predictions"]["AR-10000"]
        # {"0":{"1":1}}
        self._device.send_notification("instruments", {"notification":data})


class Daemon(Server):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, cv2.COLOR_LUV2LBGR]

    _detect_frame_data = None
    _detect_frame_data_id = None
    _cam_thread = None
    

    def __init__(self, *args, **kwargs):
        super(Daemon, self).__init__(*args, **kwargs)
        self._detect_frame_data_id = 0
        self._cam_thread = threading.Thread(target=self._cam_loop, name='cam')
        self._cam_thread.setDaemon(True)

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
                    self._send_dh(format_notification(predictions))

                frame_num += 1

        finally:
            cam.release()
            model.close()

    def _send_dh(self, data):
        # set information about the surgery
        if not self.dh_status.connected:
            logger.error('Devicehive is not connected')
            return

        
        self.deviceHive.handler.send(data)

    def get_frame(self):
        return self._detect_frame_data, self._detect_frame_data_id


class Widget():
    

    def __init__(self):
        # self.server = None

        self.window = Tk()
        self.window.title("Assist-MD Capture")
        self.window.geometry('500x400')

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

        # self.demo1_lbl = Label(self.window, text="Instrument set 1: ")
        # self.demo1_lbl.grid(column=0, row=6)
        # self.var1 = IntVar()
        # self.demo1_chk = Checkbutton(self.window, command=self.checkBoxClicked, variable=self.var1)
        # self.demo1_chk.grid(column=1, row=6)

        # self.demo2_lbl = Label(self.window, text="Instrument set 2: ")
        # self.demo2_lbl.grid(column=0, row=7)
        # self.var2 = IntVar()
        # self.demo2_chk = Checkbutton(self.window, command=self.checkBoxClicked, text="male", variable=self.var2)
        # self.demo2_chk.grid(column=1, row=7)

        # self.demo3_lbl = Label(self.window, text="Instrument set 3: ")
        # self.demo3_lbl.grid(column=0, row=8)
        # self.var3 = IntVar()
        # self.demo3_chk = Checkbutton(self.window, command=self.checkBoxClicked, text="male", variable=self.var3)
        # self.demo3_chk.grid(column=1, row=8)

        self.radiovar = IntVar()

        self.demo1_lbl = Label(self.window, text="Instrument set 1: ")
        self.demo1_lbl.grid(column=0, row=6)
        self.demo1_chk = Radiobutton(
            self.window, 
            text="Option 1", 
            variable=self.radiovar, 
            value=1,
            command=self.checkBoxClicked)
        self.demo1_chk.grid(column=1, row=6)

        self.demo2_lbl = Label(self.window, text="Instrument set 2: ")
        self.demo2_lbl.grid(column=0, row=7)
        self.demo2_chk = Radiobutton(
            self.window, 
            text="Option 2", 
            variable=self.radiovar, 
            value=2,
            command=self.checkBoxClicked)
        self.demo2_chk.grid(column=1, row=7)

        self.demo3_lbl = Label(self.window, text="Instrument set 3: ")
        self.demo3_lbl.grid(column=0, row=8)
        self.demo3_chk = Radiobutton(
            self.window, 
            text="Option 3", 
            variable=self.radiovar, 
            value=3,
            command=self.checkBoxClicked)
        self.demo3_chk.grid(column=1, row=8)

        

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

        self.instru_lbl = Label(self.window, text="Instrument In Use: ")
        self.instru_lbl.grid(column=0, row=9)
        self.instruments_in_use_box = Text(self.window, height=6, width=20)
        self.instruments_in_use_box.grid(column=0, row=10)
        self.instruments_in_use = None
   

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

    def checkBoxClicked(self):
        self.instruments_in_use = demo_instruments["instruments"][self.radiovar.get()-1]
        print("The instruments are: ", self.instruments_in_use)

        self.instruments_in_use_box.delete('1.0', END)
        for instrument in self.instruments_in_use:
            self.instruments_in_use_box.insert(END, instrument + '\n')
            print(instrument)
        


    def create_widget(self):
    
        self.window.mainloop()

if __name__ == '__main__':
    prog = Widget()
    prog.create_widget()
    
