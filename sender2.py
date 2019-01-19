from devicehive import DeviceHiveApi


url = 'http://playground.devicehive.com/api/rest'
refresh_token = 'eyJhbGciOiJIUzI1NiJ9.eyJwYXlsb2FkIjp7ImEiOlsyLDMsNCw1LDYsNyw4LDksMTAsMTEsMTIsMTUsMTYsMTddLCJlIjoxNTYzNDMyMTE4ODUxLCJ0IjowLCJ1IjozOTI2LCJuIjpbIjM4ODEiXSwiZHQiOlsiKiJdfX0.ck11Ghnum4fsvs3YWEFWNh5Tulp2in_OSYsoQgYEu9g'
device_hive_api = DeviceHiveApi(url, refresh_token=refresh_token)
device_id = 'brian-68278'
device = device_hive_api.put_device(device_id)
device.name = 'brian-68278'
device.data = {'key': 'value'}
device.save()
devices = device_hive_api.list_devices()
for device in devices:
    print('Device: %s, name: %s, data: %s' % (device.id, device.name,
                                              device.data))
    device.remove()