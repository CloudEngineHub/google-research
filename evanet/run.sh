#!/bin/bash
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


set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tensorflow
pip install -r evanet/requirements.txt

wget -P evanet/data/ https://storage.googleapis.com/gresearch/evanet/data/label_map.txt
wget -P evanet/data/ https://storage.googleapis.com/gresearch/evanet/data/v_CricketShot_g04_c01_flow.npy
wget -P evanet/data/ https://storage.googleapis.com/gresearch/evanet/data/v_CricketShot_g04_c01_rgb.npy

python -m evanet.run_evanet
