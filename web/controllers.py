import time
from six.moves import http_client
from dh_webconfig.base import Controller, BaseController


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
