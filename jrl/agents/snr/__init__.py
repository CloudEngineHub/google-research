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

"""SNR agent."""

from acme.agents.jax.sac.config import target_entropy_from_env_spec

from jrl.agents.snr import config
from jrl.agents.snr import networks
from jrl.agents.snr.builder import SNRBuilder
from jrl.agents.snr.learning import SNRLearner
from jrl.agents.snr.networks import apply_policy_and_sample
from jrl.utils.agent_utils import RLComponents


class SNRRLComponents(RLComponents):

  def __init__(self, spec, create_data_iter_fn):
    self._spec = spec
    self._config = config.SNRConfig(
        target_entropy=target_entropy_from_env_spec(spec))
    # self._config = config.SNRConfig(
    #     target_entropy=0.,
    #     # entropy_coefficient=10.)
    #     entropy_coefficient=1.)
    self._create_data_iter_fn = create_data_iter_fn

  def make_builder(self):
    return SNRBuilder(
        config=self._config,
        make_demonstrations=self._create_data_iter_fn)

  def make_networks(self):
    return networks.make_networks(
        self._spec,
        actor_hidden_layer_sizes=self._config.actor_network_hidden_sizes,
        critic_hidden_layer_sizes=self._config.critic_network_hidden_sizes,
        num_critics=self._config.num_critics,)

  def make_behavior_policy(self, network):
    return networks.apply_policy_and_sample(network, eval_mode=False)

  def make_eval_behavior_policy(self, network):
    return networks.apply_policy_and_sample(network, eval_mode=True)
