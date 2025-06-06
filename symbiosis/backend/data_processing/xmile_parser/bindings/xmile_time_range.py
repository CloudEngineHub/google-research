# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

import dataclasses
import xmile_globals

__NAMESPACE__ = "http://iseesystems.com/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileTimeRange:

  class Meta:
    name = "time_range"
    namespace = xmile_globals.ISEE_NAMESPACE

  type_value: str = dataclasses.field(
      metadata={
          "name": "type",
          "type": "Attribute",
          "required": True,
      }
  )
