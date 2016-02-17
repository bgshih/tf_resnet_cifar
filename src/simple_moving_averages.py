# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Maintain moving averages of parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
import tensorflow as tf


# TODO(touts): switch to variables.Variable.
def assign_moving_average(variable, value, decay, name=None):
    with ops.op_scope([variable, value, decay], name, "AssignMovingAvg") as name:
        with ops.device(variable.device):
            decay = ops.convert_to_tensor(1.0 - decay, name="decay")
            if decay.dtype != variable.dtype.base_dtype:
                decay = math_ops.cast(decay, variable.dtype.base_dtype)
            return state_ops.assign_sub(variable, (variable - value) * decay,
                                        name=name)


class SimpleMovingAverage(object):

    def __init__(self, name="SimpleMovingAverage"):
        self._num_updates = tf.Variable(tf.zeros([1]), name='num_updates', trainable=False)
        self._name = name
        self._averages = {}
        self._num_updates_op = tf.assign_add(self._num_updates, tf.constant(1.0, shape=[1]), name='num_updates_op')

    def apply(self, var_list=None):
        # TODO(touts): op_scope
        if var_list is None:
            var_list = variables.trainable_variables()
        for var in var_list:
            if var.dtype.base_dtype not in [dtypes.float32, dtypes.float64]:
                raise TypeError(
                    "The variables must be float or double: %s" % var)
            if var in self._averages:
                raise ValueError(
                    "Moving average already computed for: %s" % var)

            # For variables: to lower communication bandwidth across devices we keep
            # the moving averages on the same device as the variables. For other
            # tensors, we rely on the existing device allocation mechanism.
            if isinstance(var, variables.Variable):
                avg = slot_creator.create_slot(
                    var, var.initialized_value(), self._name,
                    colocate_with_primary=True)
            else:
                avg = slot_creator.create_zeros_slot(
                    var, self._name, colocate_with_primary=(var.op.type == "Variable"))
            self._averages[var] = avg

        with ops.name_scope(self._name) as scope:
            decay = self._num_updates / (self._num_updates + 1)
            updates = []
            updates.append(self._num_updates_op)
            for var in var_list:
                updates.append(assign_moving_average(
                    self._averages[var], var, decay))
            return control_flow_ops.group(*updates, name=scope)

    def average(self, var):
        """Returns the `Variable` holding the average of `var`.

        Args:
          var: A `Variable` object.

        Returns:
          A `Variable` object or `None` if the moving average of `var`
          is not maintained..
        """
        return self._averages.get(var, None)

    def average_name(self, var):
        """Returns the name of the `Variable` holding the average for `var`.

        The typical scenario for `ExponentialMovingAverage` is to compute moving
        averages of variables during training, and restore the variables from the
        computed moving averages during evaluations.

        To restore variables, you have to know the name of the shadow variables.
        That name and the original variable can then be passed to a `Saver()` object
        to restore the variable from the moving average value with:
          `saver = tf.train.Saver({ema.average_name(var): var})`

        `average_name()` can be called whether or not `apply()` has been called.

        Args:
          var: A `Variable` object.

        Returns:
          A string: the name of the variable that will be used or was used
          by the `ExponentialMovingAverage class` to hold the moving average of
          `var`.
        """
        return var.op.name + "/" + self._name
