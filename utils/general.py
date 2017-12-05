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


import colorsys
from six.moves.urllib.parse import urlparse


GOLDEN_RATIO = 0.618033988749895
NOTIFICATION_KEYS = ('class_name', 'score')


def generate_colors(n, max_value=255):
    colors = []
    h = 0.1
    s = 0.5
    v = 0.95
    for i in range(n):
        h = 1 / (h + GOLDEN_RATIO)
        colors.append([c*max_value for c in colorsys.hsv_to_rgb(h, s, v)])

    return colors


def format_predictions(predicts):
    return ', '.join('{class_name}: {score:.2f}'.format(**p) for p in predicts)


def format_notification(predicts):
    result = []
    for p in predicts:
        result.append({key: p[key] for key in NOTIFICATION_KEYS})

    return result

def find_class_by_name(name, modules):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def is_url(path):
    try:
        result = urlparse(path)
        return result.scheme and result.netloc and result.path
    except:
        return False
