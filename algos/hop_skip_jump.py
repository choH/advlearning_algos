# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
"""
Base on ART's infra and _compute_update() structure, a parial implementation of the HopSkipJump attack from Jianbo et al. (2019).
| Paper link: https://arxiv.org/abs/1904.02144
"""
# assume norm = np.inf and target = false

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class HopSkipJump(EvasionAttack):


    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "norm",
        "max_iter",
        "max_eval",
        "init_eval",
        "init_size",
        "curr_iter",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        targeted: bool = False,
        norm: Union[int, float, str] = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Create a HopSkipJump attack instance.

        :param classifier: A trained classifier.
        :param targeted: Should the attack target one specific class.
        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.
        :param max_iter: Maximum number of iterations.
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :param init_eval: Initial number of evaluations for estimating gradient.
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self._targeted = targeted
        self.norm = norm
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.curr_iter = 0
        self.batch_size = 1
        self.verbose = verbose
        self._check_params()
        self.curr_iter = 0
        self.clip_min, self.clip_max = self.estimator.clip_values

        # Set binary search threshold
        self.theta = 0.01 / np.prod(self.estimator.input_shape)

    def generate(self, x: np.ndarray, target = None, **kwargs) -> np.ndarray:


        # The following block enclosed by ###### are ART's infra on this attack method. Which means ART's developed implemented it, I read into the code and customized them to my need, but I didn't came up with them.

        ########################################################################

        target = check_and_transform_label_format(target, self.estimator.nb_classes)

        start = 0
        # Get clip_min and clip_max from the classifier or infer them from data
        # self.clip_min, self.clip_max = self.estimator.clip_values

        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        # Prediction from the initial adversarial examples if not None
        x_adv_init = init_preds = [None] * len(x)

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)


        ########################################################################

        for i, x_v in enumerate(tqdm(x_adv, desc="Hop Skip Jump attack progress", disable=not self.verbose)): # ART's TQDM progress bar
            self.curr_iter = start


            # First, create an initial adversarial sample
            initial_sample = self._initial_perbs(x_v, preds[i], init_preds[i], x_adv_init[i])
            # initial_sample = self._initial_perbs(x, y_p, init_pred, adv_init, clip_min, clip_max)
            # If an initial adversarial example is not found, then return the original image

            if initial_sample is None:
                x_adv[i] = x_v
            else:
                x_adv[i] = self._boundary_attack(initial_sample[0], x_v, initial_sample[1])

        return x_adv

    # Try first direction, if not adv try B (init_size) number of random vectors to see if adv.
    # If find adv among random vectors (target), find the midpoint between original to the target
    # If midpoint still adv, keep finding midpoint to closing to the original side.
    # Until midpoint because non adv, we make the pervious iteration to be the attack adv.

    # ART's required output being Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]. Used ART's variable to couple with their _compute_update().
    def _initial_perbs(self, x: np.ndarray, x_pred: int, init_pred: int, adv_init: np.ndarray):
        random_vectors = np.random.RandomState()
        initial_sample = None

        print('initial sample started')

        # If the first direction is good
        if adv_init is not None and init_pred != x_pred:
            return adv_init.astype(ART_NUMPY_DTYPE), x_pred

        # True different vectors (within initial size)
        for _ in range(self.init_size):
            random_img = random_vectors.uniform(self.clip_min, self.clip_max, size=x.shape).astype(x.dtype)

            random_class = np.argmax(
                self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1,
            )[0]

            if random_class != x_pred:
                # Binary search to reduce the l2 distance to the original image
                random_img = self._binary_search(
                    current_sample=random_img,
                    original_sample=x,
                    target=x_pred,
                    thrl=0.001, ###
                    L2_flag = True
                )
                initial_sample = random_img, x_pred

                print("Found adv candidate example with different label, proceed to this direction.")
                break
        else:
            print("Attack failed due to having no example with different label within the eps ball.")


        return initial_sample

    def _boundary_attack(self, current_sample: np.ndarray, original_sample: np.ndarray, target: int):

        # Walking around the boundary to find the minimum norm.
        for _ in range(self.max_iter):
            eps_step = 0.1
            L2_coe = 2.0

            delta = None
            if self.curr_iter == 0:
                delta = eps_step * (self.clip_max - self.clip_min)
            else:
                dist = np.max(abs(original_sample - current_sample))
                delta = np.prod(self.estimator.input_shape) * self.theta * dist # Just sign?

            current_sample = self._binary_search(current_sample, original_sample, target)


            # Next compute the number of evaluations and compute the update
            # Bound by max_eval in init.
            # Authors use max_iter=64, max_eval=10000, init_eval=100 as minimum norm decrese observed
            # need large number to get out.
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)
            update = self._compute_update(current_sample, num_eval, delta, target) # ART's method
            dist = np.max(abs(original_sample - current_sample))
            eps = L2_coe * dist / np.sqrt(self.curr_iter + 1)


            adv_success = False
            while not adv_success:
                eps /= 2.0
                sample_candidate = current_sample + eps * update

                wraped_sample_candidate = sample_candidate[None, ...]
                clipped_sample_candidate = np.clip(wraped_sample_candidate, self.clip_min, self.clip_max)
                samples_pred = np.argmax(self.estimator.predict(clipped_sample_candidate, batch_size=self.batch_size), axis=1)
                adv_success = samples_pred != target
                # adv_success = adv_success[0]

            # make sample_candidate current
            current_sample = np.clip(sample_candidate, self.clip_min, self.clip_max)
            self.curr_iter += 1

        return current_sample

    def _binary_search(self, current_sample, original_sample, target, thrl = None, L2_flag = False):

        # First set upper and lower bounds as well as the threshold for the binary search
        lower_bound = 0
        if L2_flag:
            upper_bound = 1
        else:
            upper_bound = np.max(abs(original_sample - current_sample))

        if thrl is None:
            thrl = np.minimum(upper_bound * self.theta, self.theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > thrl:
            # Mid point between current and target
            mid_point = (upper_bound + lower_bound) / 2.0
            # Get representation
            mid_point_sample = np.clip(current_sample, original_sample - mid_point, original_sample + mid_point)

            wraped_mid_point_sample = mid_point_sample[None, ...]
            clipped_samples = np.clip(wraped_mid_point_sample, self.clip_min, self.clip_max)
            sample_pred = np.argmax(self.estimator.predict(clipped_samples, batch_size=self.batch_size), axis=1)
            adv_success = sample_pred != target
            adv_success = adv_success[0]

            # Update range base on the new prediction
            lower_bound = np.where(adv_success == 0, mid_point, lower_bound)
            upper_bound = np.where(adv_success == 1, mid_point, upper_bound)

        clipped_mid_point_sample = np.clip(current_sample, original_sample - mid_point, original_sample + mid_point)
        return clipped_mid_point_sample

# The following block enclosed by ###### are written by ART's developer, it has something to do with their storage structure and thet implemented upon authos' source code but not paper. Tried to messed around with it, no success. Although I did customized a bit to only the portion I need.

# eval_samples
# Eq.(14).

################################################################################

    def _compute_update(
        self,
        current_sample: np.ndarray,
        num_eval: int,
        delta: float,
        target: int,
    ) -> np.ndarray:

        # Generate random noise
        rnd_noise_shape = [num_eval] + list(self.estimator.input_shape)

        rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)

        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / np.sqrt(
            np.sum(rnd_noise ** 2, axis=tuple(range(len(rnd_noise_shape)))[1:], keepdims=True,)
        )
        eval_samples = np.clip(current_sample + delta * rnd_noise, self.clip_min, self.clip_max)
        rnd_noise = (eval_samples - current_sample) / delta

        # Compute gradient: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        satisfied = self._if_attack_success(samples=eval_samples, target=target)

        # Bugfix 2: reshape 280 to (10, 1, 1, 1)
        # print(satisfied.shape)
#         (10,)
# (10,)
# (10,)
# (10,)
# (10,)
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.estimator.input_shape)) - 1.0
        f_val = f_val.astype(ART_NUMPY_DTYPE)

        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)

        # Compute update
        result = np.sign(grad)

        return result

################################################################################

    def _if_attack_success(self, samples, target):
        samples = np.clip(samples, self.clip_min, self.clip_max)
        preds = np.argmax(self.estimator.predict(samples, batch_size=self.batch_size), axis=1)
        result = preds != target

        return result

