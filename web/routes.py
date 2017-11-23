from web.controllers import Events, Stream

routes = [
    (r'^/events/$', Events),
    (r'^/stream/$', Stream),
]
