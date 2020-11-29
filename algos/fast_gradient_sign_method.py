# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    random_sphere,
    projection,
    check_and_transform_label_format,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class FastGradientSignMethod(EvasionAttack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity norm (and is known as the "Fast
    Gradient Sign Method"). Developers of ART implemented Goodfellow method by extending its attack to other norms.

    I used some of ART's infa, as the attack takes an ART's estimator as input and relaying on some ART datatypes, but my implementation
    does not have any extension and is a representation of the FGSM method introduced in paper.

    I have also consulted the tutorial of Google on FGSM: https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

    Paper link: https://arxiv.org/abs/1412.6572
    """

    # Input params of ART
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    # Please refer to ART regarding the input setup of their estimator, we have consulted Dr. Ray that it is ok toimport an estimator as our "victim" model.
    # We have only use a subset of params, you may find the full version here: https://adversarial-robustness-toolbox.readthedocs.io/en/stable/modules/attacks/evasion.html#fast-gradient-method-fgm
    # Although for the implemented param, their function are the same.

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: int = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        targeted: bool = True,
        batch_size: int = 32,
        minimal: bool = False,
    ):
        super().__init__(estimator=estimator)
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self._targeted = targeted
        self.batch_size = batch_size
        self.minimal = minimal


    # This method is usually the trigger method for ART for generating adversarial examples. We rerwite the code to be FGSM, buy
    # we kept the same method name for 1) consistency purposes; 2) in case some ART defense performance evaluation will internally call it.

    def generate(self, x: np.ndarray, aimed_target: np.ndarray = None) -> np.ndarray:
        x_label = get_labels_np_array(self.estimator.predict(x, batch_size = self.batch_size))

        # adv_x = self._compute(x, x_label, self.eps,)

        adv_x = x.copy()

        window_slice_amt = int(np.ceil(float(x.shape[0]) / self.batch_size))

        # (28, 28, 1)
        # if self.targeted:
        #     adv_x = self._minimal_perturbation(x, aimed_target)
        # else:
        if aimed_target is None:
            for batch_i in range(window_slice_amt):
                batch_start_i = batch_i * self.batch_size
                batch_end_i = (batch_i + 1) * self.batch_size
                batch_end_i = min(batch_end_i, x.shape[0])

                current_batch = adv_x[batch_start_i: batch_end_i]
                current_batch_label = x_label[batch_start_i: batch_end_i]

                batch_grad = self.estimator.loss_gradient(current_batch, current_batch_label)
                # batch_grad = self.estimator.loss_gradient(current_batch, target_label)
                batch_perturb = np.sign(batch_grad) #sign

                # current_batch += self.eps * batch_perturb
                current_batch += self.eps * batch_perturb
                # adv_batch = np.clip(current_batch, -1, 1) # tf.clip_by_value()
                adv_batch = current_batch # tf.clip_by_value()

                adv_x[batch_start_i: batch_end_i] = adv_batch
        else:
            aimed_target = np.array([aimed_target * len(x)])
            target_label = get_labels_np_array(self.estimator.predict(aimed_target, batch_size = self.batch_size))

            for batch_i in range(window_slice_amt):
                batch_start_i = batch_i * self.batch_size
                batch_end_i = (batch_i + 1) * self.batch_size
                batch_end_i = min(batch_end_i, x.shape[0])

                current_batch = adv_x[batch_start_i: batch_end_i]
                current_batch_label = x_label[batch_start_i: batch_end_i]

                # batch_grad = self.estimator.loss_gradient(current_batch, current_batch_label)
                batch_grad = self.estimator.loss_gradient(current_batch, target_label)
                batch_perturb = np.sign(batch_grad) #sign

                # current_batch += self.eps * batch_perturb
                current_batch -= self.eps * batch_perturb
                # adv_batch = np.clip(current_batch, -1, 1) # tf.clip_by_value()
                adv_batch = current_batch # tf.clip_by_value()

                adv_x[batch_start_i: batch_end_i] = adv_batch
        return adv_x

