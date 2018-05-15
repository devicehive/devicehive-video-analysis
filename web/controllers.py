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
from six.moves import http_client
from devicehive_webconfig.base import Controller, BaseController


class Events(Controller):
    def get(self, handler, *args, **kwargs):
        response = self.render_template('stream.html')

        handler.send_response(http_client.OK)
        handler.send_header('Content-type', 'text/html')
        handler.end_headers()
        handler.wfile.write(response.encode())


class Stream(BaseController):
    def get(self, handler, *args, **kwargs):
        handler.send_response(http_client.OK)
        c_type = 'multipart/x-mixed-replace; boundary=--mjpegboundary'
        handler.send_header('Content-Type', c_type)
        handler.end_headers()
        prev = None
        while handler.server.server.is_running:
            data, frame_id = handler.server.server.get_frame()
            if data is not None and frame_id != prev:
                prev = frame_id
                handler.wfile.write(b'--mjpegboundary\r\n')
                handler.send_header('Content-type', 'image/jpeg')
                handler.send_header('Content-length', str(len(data)))
                handler.end_headers()
                handler.wfile.write(data)
            else:
                time.sleep(0.025)
