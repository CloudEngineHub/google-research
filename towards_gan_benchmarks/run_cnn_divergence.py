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

"""Runs the CNN divergence on two toy distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import towards_gan_benchmarks.lib.flags
import towards_gan_benchmarks.lib.nn_divergence

lib = towards_gan_benchmarks.lib


def main(_):
  flags = lib.flags.Flags()
  lib.nn_divergence.set_flags(flags)

  def real_gen():
    while True:
      yield np.random.randint(0, 246, (flags.batch_size, 32, 32, 3), 'int32')

  def fake_gen():
    while True:
      yield np.random.randint(10, 256, (flags.batch_size, 32, 32, 3), 'int32')

  result = lib.nn_divergence.run(flags, real_gen(), fake_gen())
  print(result)


if __name__ == '__main__':
  tf.app.run()
