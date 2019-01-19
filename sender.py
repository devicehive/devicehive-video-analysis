from devicehive import Handler
from devicehive import DeviceHive


class SenderHandler(Handler):

    def __init__(self, api, device_id='brian-68278',
                 accept_command_name='accept_notifications',
                 num_notifications=10):
        Handler.__init__(self, api)
        self._device_id = device_id
        self._accept_command_name = accept_command_name
        self._num_notifications = num_notifications
        self._device = None

    def _send_notifications(self):
        for num_notification in range(self._num_notifications):
            notification = '%s-notification' % num_notification
            self._device.send_notification("name", {"name:",notification})
            print('Sending notification "%s"' % notification)
        self.api.disconnect()

    def handle_connect(self):
        self._device = self.api.get_device(self._device_id)
        self._device.send_command(self._accept_command_name)
        print('Sending command "%s"' % self._accept_command_name)
        self._device.subscribe_update_commands([self._accept_command_name])

    def handle_command_update(self, command):
        if command.status == 'accepted':
            print('Command "%s" accepted' % self._accept_command_name)
            self._send_notifications()


url = 'http://playground.devicehive.com/api/rest'
refresh_token = 'eyJhbGciOiJIUzI1NiJ9.eyJwYXlsb2FkIjp7ImEiOlsyLDMsNCw1LDYsNyw4LDksMTAsMTEsMTIsMTUsMTYsMTddLCJlIjoxNTYzNDMyMTE4ODUxLCJ0IjowLCJ1IjozOTI2LCJuIjpbIjM4ODEiXSwiZHQiOlsiKiJdfX0.ck11Ghnum4fsvs3YWEFWNh5Tulp2in_OSYsoQgYEu9g'
dh = DeviceHive(SenderHandler)
dh.connect(url, refresh_token=refresh_token)