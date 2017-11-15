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
import random


def generate_colors(n, max_value=255):
    colors = []
    base = 2 / n

    for i in range(n):
        h = i * base
        s = 1 - h
        l = 0.5 + (random.random() - 0.5) / 5
        colors.append([c*max_value for c in colorsys.hls_to_rgb(h, l, s)])

    return colors
