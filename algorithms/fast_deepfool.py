from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional, TYPE_CHECKING
from tqdm import trange

import logging
import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, is_probability

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)

class FastDeepFool(EvasionAttack):
    """
    This class implements the DeepFool evasion attack, from Moosavi-Dezfooli et al. (2015): https://arxiv.org/abs/1511.04599
    """
    
    attack_params = EvasionAttack.attack_params + [
        "total_iter",
        "epsilon",
        "total_grad",
        "batch_size",
        "show_prog"
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self, 
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        total_iter: int = 100,
        epsilon: float = 1e-6,
        total_grad: int = 10,
        batch_size: int = 1,
        show_prog: bool = True,
    ) -> None:
        """
        :param classifier: An existing, trained classifier
        :param total_iter: Total number of iterations
        :param epsilon: Overshoot correctional value
        :param total_grad: Total number of class gradients to compute
        :param batch_size: Size of each batch
        :param show_prog: Toggles whether to show progress bars
        """

        super().__init__(estimator=classifier)
        self.total_iter = total_iter
        self.epsilon = epsilon
        self.total_grad = total_grad
        self.batch_size = batch_size
        self.show_prog = show_prog
        self.validate_params()

        if self.estimator.clip_values is None:
            logger.warning(
                "Adversarial perturbations will be defaulted to be scaled to input values"
                " in the range of 0 to 1, due to 'clip_values' attribute being None."
            )
        
    def generate(self, inputs: np.ndarray, labels: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Function creates adversarial examples, then returns them in a numpy array
        
        :param inputs: Array with original inputs (to be attacked)
        :param labels: Array with original labels (to be predicted)
        :return: Array containing adversarial examples
        """

        adv_inputs = inputs.astype(ART_NUMPY_DTYPE)
        predictions = self.estimator.predict(inputs, batch_size=self.batch_size)

        if(is_probability(predictions[0])):
            logger.warning(
                "Targeted model should output logits, not probabilities for predictions."
            )
        
        # Determine class labels for gradients
        use_grad_subset, labels_set = self.define_class_labels(predictions)
        sorter = np.arange(len(labels_set))

        # Calculate perturbation with batch
        for batch_nb in trange(int(np.ceil(adv_inputs.shape[0] / float(self.batch_size))), desc="DeepFool", disable=not self.show_prog):
            batch_idx_1, batch_idx_2 = batch_nb * self.batch_size, (batch_nb + 1) * self.batch_size
            batch = adv_inputs[batch_idx_1:batch_idx_2].copy()

            # Predictions for batch
            f_batch, fk_hat = self.batch_predict(predictions, batch_idx_1, batch_idx_2)

            # Gradient for batch
            grads = self.batch_gradient(batch, use_grad_subset, labels_set)

            # Gets current predictions
            active_idxs = np.arange(len(batch))
            step = 0
            while (active_idxs.size > 0) and (step < self.total_iter):
                # Difference in gradients and predictions for selected predictions
                labels_idxs = sorter[np.searchsorted(labels_set, fk_hat, sorter=sorter)]
                grad_dif = grads - grads[np.arange(len(grads)), labels_idxs][:, None]
                f_dif = f_batch[:, labels_set] - f_batch[np.arange(len(f_batch)), labels_idxs][:, None]

                # Select coordinate and compute perturbation
                r_var = self.perturbation(adv_inputs, labels_set, labels_idxs, grad_dif, f_dif)

                # Add new perturbation to clip result
                if self.estimator.clip_values is not None:
                    batch[active_idxs] = np.clip(
                        batch[active_idxs] 
                        + r_var[active_idxs] * (self.estimator.clip_values[1] - self.estimator.clip_values[0]), 
                        self.estimator.clip_values[0], 
                        self.estimator.clip_values[1],
                    )
                else:
                    batch[active_idxs] += r_var[active_idxs]
                
                # Recalculate prediction
                f_batch = self.estimator.predict(batch)
                fk_i_hat = np.argmax(f_batch, axis=1)

                # Recalculate gradient
                grads = self.batch_gradient(batch, use_grad_subset, labels_set)

                # Check if misclassification has occured
                active_idxs = np.where(fk_i_hat == fk_hat)[0]

                step += 1
            
            # Apply overshoot parameters
            adv_inputs[batch_idx_1:batch_idx_2] = self.overshoot(adv_inputs, batch_idx_1, batch_idx_2, batch, batch_nb)

            if self.estimator.clip_values is not None:
                np.clip(
                    adv_inputs[batch_idx_1:batch_idx_2],
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
                    out=adv_inputs[batch_idx_1:batch_idx_2],
                )
        
        logger.info(
            "DeepFool attack success rate: %.2f%%",
            100 * compute_success(self.estimator, inputs, labels, adv_inputs, batch_size=self.batch_size),
        )

        return adv_inputs
            
    def define_class_labels(self, predictions):
        """
        Function determines the class labels to compute the gradients

        :param predictions: Predictions for examples
        :return: Array of class labels for computing the gradients
        """
        use_grad_subset = self.total_grad < self.estimator.nb_classes
        if use_grad_subset:
            grad_labels = np.argsort(-predictions, axis=1)[:, : self.total_grad]
            return use_grad_subset, np.unique(grad_labels)
        else:
            return use_grad_subset, np.arange(self.estimator.nb_classes)
    
    def batch_predict(self, predictions, idx_1, idx_2):
        """
        Function determines predictions for current batch

        :param predictions: Predictions for examples
        :param idx_1: First index of slice
        :param idx_2: Second index of slice
        :return: Predictions
        """
        f_batch = predictions[idx_1:idx_2]
        fk_hat = np.argmax(f_batch, axis=1)
        return f_batch, fk_hat
    
    def batch_gradient(self, batch, use_grad_subset, labels_set):
        """
        Function determines the gradients for the current batch

        :param batch: Current batch
        :param use_gradient_subset: Toggles whether the entire or a part of the gradient should be used
        :param labels_set: Class labels
        :return: Array of the gradients
        """
        if use_grad_subset:
            # Gradients for predicted top classes
            grads = np.array([self.estimator.class_gradient(batch, label=_) for _ in labels_set])
            grads = np.squeeze(np.swapaxes(grads, 0, 2), axis=0)
            return grads
        else:
            # Gradients for all classes
            grads = self.estimator.class_gradient(batch)
            return grads
    
    def perturbation(self, adv_inputs, labels_set, labels_idxs, grad_dif, f_dif):
        """
        Function selects the coordinate and computes the perturbation

        :param adv_inputs: Adversarial inputs
        :param labels_set: Set of class labels
        :param labels_idxs: Array of label indices
        :param grad_dif: Gradient differences for best selected predicitions
        :param f_dif: Prediction differences for best selected predicitions
        :return: Calculated perturbation
        """
        small = 10e-8

        norm = np.linalg.norm(grad_dif.reshape(len(grad_dif), len(labels_set), -1), axis=2) + small
        val = np.abs(f_dif) / norm
        val[np.arange(len(val)), labels_idxs] = np.inf
        l_var = np.argmin(val, axis=1)
        abs1 = abs(f_dif[np.arange(len(f_dif)), l_var])
        d_dif = grad_dif[np.arange(len(grad_dif)), l_var].reshape(len(grad_dif), -1)
        pow1 = pow(np.linalg.norm(d_dif, axis=1), 2,) + small

        r_var = abs1 / pow1
        r_var = r_var.reshape((-1,) + (1,) * (len(adv_inputs.shape) -1))
        r_var = r_var * grad_dif[np.arange(len(grad_dif)), l_var]

        return r_var
    
    def overshoot(self, adv_inputs, idx_1, idx_2, batch, batch_nb):
        """
        Function applies overshooting parameter

        :param adv_inputs: adversarial inputs
        :param idx_1: First slice index
        :param idx_2: Second slice index
        :param batch: The current batch
        :return: Updated inputs
        """
        inputs1 = adv_inputs[idx_1:idx_2]
        self.epsilon += 0.01 / (batch_nb + 1)
        inputs2 = (1 + self.epsilon) * (batch - adv_inputs[idx_1:idx_2])
        return inputs1 + inputs2
    
    def validate_params(self) -> None:
        if not isinstance(self.total_iter, (int, np.int)) or self.total_iter <= 0:
            raise ValueError("Total iterations must be positive.")
        if not isinstance(self.total_grad, (int, np.int)) or self.total_grad <= 0:
            raise ValueError("Total number of class gradients must be positive.")
        
        if self.epsilon < 0:
            raise ValueError("Overshooting parameter must be non-negative.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        
        if not isinstance(self.show_prog, bool):
            raise ValueError("Arg for 'show_prog' must be of type bool.")
