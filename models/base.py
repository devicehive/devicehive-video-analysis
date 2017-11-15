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


class BaseModel(object):
    """Base model class."""

    def init(self):
        raise NotImplementedError

    def evaluate(self, task):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
