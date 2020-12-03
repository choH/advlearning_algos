from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import product
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from six import string_types
from tqdm import tqdm

import numpy as np
from scipy._lib._util import check_random_state
from scipy.optimize.optimize import _status_message
from scipy.optimize import OptimizeResult, minimize

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

from cma import CMAEvolutionStrategy, CMAOptions

import logging
logger = logging.getLogger(__name__)

class PixelThreshold(EvasionAttack):
    attack_params = EvasionAttack.attack_params + ["threshold", "ev_strat", "targeted", "verbose"]
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        threshold: Optional[int],
        ev_strat: int,
        targeted: bool,
        verbose: bool = True,
    ) -> None:
        """
        :param classifier: Trained classifier.
        :param threshold: Threshold value of attack
        :param ev_strat: Selects evolutionary strategy: CMAES (0) or DE (1)
        :param targeted: Toggles whether attack is targeted or untargeted
        :param verbose: Print messages of ES, show progress bars.
        """
        super().__init__(estimator=classifier)

        self._project = True
        self.type_attack = -1
        self.threshold = threshold
        self.ev_strat = ev_strat
        self._targeted = targeted
        self.verbose = verbose

        PixelThreshold.validate_params(self)

        if self.estimator.channels_first:
            self.init_img(-2, -1, -3)
        else:
            self.init_img(-3, -2, -1)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, total_iter: int = 100, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in a numpy array.

        :param x: Input examples
        :param y: Target class lables
        :param total_iter: Total number of iterations
        :return: Numpy array of adversarial examples
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes, return_one_hot=False)

        # Format target labels
        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` must be provided for targeted attack.")
            y = np.argmax(self.estimator.predict(x), axis=1)
        else:
            if len(y.shape) > 1:
                y = np.argmax(y, axis=1)
        
        # Format inputs
        if np.max(x) <= 1:
            x = x * 255.0

        if self.threshold is None:
            logger.info("Attacking with minimal perturbation.")

        # Generate adversarial inputs
        adv_x_best = []
        for image, target_class in tqdm(zip(x, y), desc="Pixel threshold", disable=not self.verbose):
            if self.threshold is None:
                self.min_th = 127
                start, end = 1, 127
                while True:
                    image_result: Union[List[np.ndarray], np.ndarray] = []
                    threshold = (start + end) // 2
                    success, test_image_result = self.attack(image, target_class, threshold, total_iter)
                    if image_result or success:
                        image_result = test_image_result
                    if success:
                        end = threshold - 1
                        self.min_th = threshold
                    else:
                        start = threshold + 1
                    if end < start:
                        if isinstance(image_result, list) and not image_result:
                            image_result = image
                        break
            else:
                success, image_result = self.attack(image, target_class, self.threshold, total_iter)

            adv_x_best += [image_result]

        adv_x_best = np.array(adv_x_best)

        if np.max(x) <= 1:
            x = x / 255.0

        if y is not None:
            y = to_categorical(y, self.estimator.nb_classes)

        logger.info(
            "Success rate of Attack: %.2f%%", 100 * compute_success(self.estimator, x, y, adv_x_best, self.targeted, 1),
        )
        return adv_x_best

    def bounds(self, img: np.ndarray, limit) -> Tuple[List[list], list]:
        """
        Define the bounds for the image `img` within the limits `limit`.

        :param img: The example image being bounded
        :param limit: Limits to bound the image
        :return: The bounds of the image within the limits
        """
        # Internal function
        def bound_limit(value):
            return np.clip(value - limit, 0, 255), np.clip(value + limit, 0, 255)

        minbounds, maxbounds, bounds, initial = [], [], [], []

        img_dims = product(range(img.shape[-3]), range(img.shape[-2]), range(img.shape[-1]))
        for i, j, k in img_dims:
            temp = img[i, j, k]
            initial += [temp]
            bound = bound_limit(temp)
            if self.ev_strat == 0:
                minbounds += [bound[0]]
                maxbounds += [bound[1]]
            else:
                bounds += [bound]
        if self.ev_strat == 0:
            bounds = [minbounds, maxbounds]

        return bounds, initial

    def attack(
        self, image: np.ndarray, target_class: np.ndarray, limit: int, total_iter: int
    ) -> Tuple[bool, np.ndarray]:
        """
        Attack given image with the threshold limit for target_class, which is true label for
        untargeted attack and targeted label for targeted attack.
        """
        bounds, initial = self.bounds(image, limit)

        # CMAES attack selection
        adv_x = self.adv_inputs(image, target_class, bounds, initial, limit, total_iter)

        if self.attack_completed(adv_x, image, target_class):
            return True, self.perturbate(adv_x, image)[0]
        else:
            return False, image
    
    def adv_inputs(self, image, target_class, bounds, initial, limit, total_iter):
        """
        Function prepares adversarial inputs depending on selected strategy
        """

        def predict_func(x):
            """
            Function wraps call for classifier's predicition function
            """
            preds = self.estimator.predict(self.perturbate(x, image))[:, target_class]

            if not self.targeted:
                return preds
            else:
                return 1 - preds
        
        def callback_func(x, converge=None):
            """
            Function traces back attack
            """
            if self.ev_strat == 0:
                if self.attack_completed(x.result[0], image, target_class):
                    raise Exception("Attack completed early.")
            else:
                return self.attack_completed(x, image, target_class)

        # CMAES attack selection
        if self.ev_strat == 0:
            opts = CMAOptions()
            if not self.verbose:
                opts.set("verbose", -9)
                opts.set("verb_disp", 40000)
                opts.set("verb_log", 40000)
                opts.set("verb_time", False)

            opts.set("bounds", bounds)

            if self.type_attack == 0:
                std = 63
            else:
                std = limit

            strategy = CMAEvolutionStrategy(initial, std / 4, opts)

            try:
                strategy.optimize(
                    predict_func,
                    maxfun=max(1, 400 // len(bounds)) * len(bounds) * 100,
                    callback=callback_func,
                    iterations=1,
                )
            except Exception as exception:
                if self.verbose:
                    print(exception)

            return strategy.result[0]
        # Differential evolution attack selection
        else:
            strategy = differential_evolution(
                predict_func,
                bounds,
                disp=self.verbose,
                totaliter=total_iter,
                popsize=max(1, 400 // len(bounds)),
                recombination=1,
                atol=-1,
                callback=callback_func,
                polish=False,
            )
            return strategy.x
    
    def attack_completed(self, adv_x, x, target_class):
        """
        Checks whether the given perturbation `adv_x` for the image `img` is successful.
        """
        pred_class = np.argmax(self.estimator.predict(self.perturbate(adv_x, x))[0])

        return bool(
            (self.targeted and pred_class == target_class) or
            (not self.targeted and pred_class != target_class)
        )
    
    def init_img(self, row, col, chan):
        self.img_rows = self.estimator.input_shape[row]
        self.img_cols = self.estimator.input_shape[col]
        self.img_channels = self.estimator.input_shape[chan]
    
    def perturbate(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Perturbs the given image `img` with the given perturbation `x`.
        """
        return img
    
    def validate_params(self) -> None:
        if self.threshold is not None:
            if self.threshold <= 0:
                raise ValueError("Perturbation size must be positive.")

        if not isinstance(self.ev_strat, int):
            raise ValueError("Flag `ev_strat` must be of type int.")

        if not isinstance(self.targeted, bool):
            raise ValueError("Flag `targeted` must be of type bool.")

        if not isinstance(self.verbose, bool):
            raise ValueError("Flag `verbose` must be of type bool.")

        if not isinstance(self.verbose, bool):
            raise ValueError("Argument `verbose` must be of type bool.")


class PixelAttack(PixelThreshold):
    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        threshold: Optional[int] = None,
        ev_strat: int = 0,
        targeted: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        :param classifier: Trained classifier.
        :param threshold: Threshold value of attack
        :param ev_strat: Selects evolutionary strategy: CMAES (0) or DE (1)
        :param targeted: Toggles whether attack is targeted or untargeted
        :param verbose: Print messages of ES, show progress bars.
        """
        super().__init__(classifier, threshold, ev_strat, targeted, verbose)
        self.type_attack = 0

    def perturbate(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Perturbs the given image `img` with the given perturbation `x`.
        """
        if x.ndim < 2:
            x = np.array([x])

        imgs = np.tile(img, [len(x)] + [1] * (x.ndim + 1))
        x = x.astype(int)
        for adv, image in zip(x, imgs):
            split = np.split(adv, len(adv) // (2 + self.img_channels))
            for pixel in split:
                x_pos, y_pos, *rgb = pixel
                if self.estimator.channels_first:
                    image[:, x_pos % self.img_rows, y_pos % self.img_cols] = rgb
                else:
                    image[x_pos % self.img_rows, y_pos % self.img_cols] = rgb

        return imgs

    def bounds(self, img: np.ndarray, limit) -> Tuple[List[list], list]:
        """
        Define the bounds for the image `img` within the limits `limit`.
        """
        initial: List[np.ndarray] = []
        bounds: List[List[int]]
        if self.ev_strat == 0:
            img_dims = product(range(self.img_rows), range(self.img_cols))
            for count, (i, j) in enumerate(img_dims):
                initial += [i, j]
                for k in range(self.img_channels):
                    if not self.estimator.channels_first:
                        initial += [img[i, j, k]]
                    else:
                        initial += [img[k, i, j]]

                if count == (limit - 1):
                    break
                else:
                    continue

            min_bounds = [0, 0]
            for i in range(self.img_channels):
                min_bounds += [0]

            min_bounds = min_bounds * limit
            max_bounds = [self.img_rows, self.img_cols]
            for i in range(self.img_channels):
                max_bounds += [255]

            max_bounds = max_bounds * limit
            bounds = [min_bounds, max_bounds]
        else:
            bounds = [[0, self.img_rows], [0, self.img_cols]]
            for _ in range(self.img_channels):
                bounds += [[0, 255]]
            bounds = bounds * limit

        return bounds, initial

class ThresholdAttack(PixelThreshold):
    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        threshold: Optional[int] = None,
        ev_strat: int = 0,
        targeted: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        :param classifier: Trained classifier.
        :param threshold: Threshold value of attack
        :param ev_strat: Selects evolutionary strategy: CMAES (0) or DE (1)
        :param targeted: Toggles whether attack is targeted or untargeted
        :param verbose: Print messages of ES, show progress bars.
        """
        super().__init__(classifier, threshold, ev_strat, targeted, verbose)
        self.type_attack = 1

    def perturbate(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Perturbs the given image `img` with the given perturbation `x`.
        """
        if x.ndim < 2:
            x = x[None, ...]
        imgs = np.tile(img, [len(x)] + [1] * (x.ndim + 1))
        x = x.astype(int)

        for adv, image in zip(x, imgs):
            img_dims = product(range(image.shape[-3]), range(image.shape[-2]), range(image.shape[-1]),)
            for count, (i, j, k) in enumerate(img_dims):
                image[i, j, k] = adv[count]

        return imgs

class DifferentialEvolutionSolver:
    # Dispatch of mutation strategy method (binomial or exponential).
    binomial = {
        "best1bin": "_best1",
        "randtobest1bin": "_randtobest1",
        "currenttobest1bin": "_currenttobest1",
        "best2bin": "_best2",
        "rand2bin": "_rand2",
        "rand1bin": "_rand1",
    }
    exponential = {
        "best1exp": "_best1",
        "rand1exp": "_rand1",
        "randtobest1exp": "_randtobest1",
        "currenttobest1exp": "_currenttobest1",
        "best2exp": "_best2",
        "rand2exp": "_rand2",
    }
    init_err_msg = (
        "The population initialization method must be one of "
        "'latinhypercube' or 'random', or an array of shape "
        "(M, N) where N is the number of parameters and M>5"
    )

    def __init__(
        self,
        func,
        bounds,
        args=(),
        strategy="best1bin",
        totaliter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        maxfun=np.inf,
        callback=None,
        disp=False,
        polish=True,
        init="latinhypercube",
        atol=0,
    ):

        if strategy in self.binomial:
            self.mutation_func = getattr(self, self.binomial[strategy])
        elif strategy in self.exponential:
            self.mutation_func = getattr(self, self.exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")

        self.func = func
        self.args = args
        self.strategy = strategy
        self.callback = callback
        self.polish = polish
        self.tol = tol
        self.atol = atol
        self.disp = disp

        # Mutation constant should be in [0, 2), and dither if sequence
        self.scale = mutation
        if (
            not np.all(np.isfinite(mutation)) or 
            np.any(np.array(mutation) >= 2) or 
            (np.any(np.array(mutation) < 0))
        ):
            raise ValueError(
                "The mutation constant must be a float in "
                "U[0, 2), or specified as a tuple(min, max)"
                " where min < max and min, max are in U[0, 2)."
            )

        self.dither = None
        if hasattr(mutation, "__iter__") and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.limits = np.array(bounds, dtype="float").T
        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError(
                "bounds should be a sequence containing " "real valued (min, max) pairs for each value" " in x"
            )

        if totaliter is None:
            totaliter = 1000
        self.totaliter = totaliter

        if maxfun is None:
            maxfun = np.inf
        self.maxfun = maxfun

        self.scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.param_count = np.size(self.limits, 1)
        self.random_number_generator = check_random_state(seed)
        self.nb_pop_mem = max(5, popsize * self.param_count)
        self.pop_shape = (self.nb_pop_mem, self.param_count)
        self._nfev = 0

        if isinstance(init, string_types):
            if init == "random":
                self.init_pop_random()
            elif init == "latinhypercube":
                self.init_pop_lhs()
            else:
                raise ValueError(self.init_err_msg)
        else:
            self.init_pop_array(init)

    def init_pop_lhs(self):
        """
        Initialize population with Latin Hypercube Sampling for uniform sampling
        """
        rng = self.random_number_generator

        # Uniformly sample params
        segment_size = 1.0 / self.nb_pop_mem

        samples = (
            segment_size * rng.random_sample(self.pop_shape)
            + np.linspace(0.0, 1.0, self.nb_pop_mem, endpoint=False)[:, np.newaxis]
        )

        # Population of possible solutions
        self.population = np.zeros_like(samples)

        # Initialize population of possible solutions
        for j in range(self.param_count):
            order = rng.permutation(range(self.nb_pop_mem))
            self.population[:, j] = samples[order, j]

        # Reset population energies
        self.pop_energies = np.ones(self.nb_pop_mem) * np.inf

        # Reset number of function evaluations
        self._nfev = 0

    def init_pop_random(self):
        """
        Initialize population at random
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.pop_shape)

        # Reset population energies
        self.pop_energies = np.ones(self.nb_pop_mem) * np.inf

        # Reset number of function evaluations
        self._nfev = 0

    def init_pop_array(self, init):
        """
        Initialize population with user-specified population.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if np.size(popn, 0) < 5 or popn.shape[1] != self.param_count or len(popn.shape) != 2:
            raise ValueError("The population supplied needs to have shape" " (M, len(x)), where M > 4.")

        # Scale values and clip to bounds, assigning to population
        self.population = np.clip(self.unscale(popn), 0, 1)
        self.nb_pop_mem = np.size(self.population, 0)
        self.pop_shape = (self.nb_pop_mem, self.param_count)

        # Reset population energies
        self.pop_energies = np.ones(self.nb_pop_mem) * np.inf

        # Reset number of function evaluations counter
        self._nfev = 0

    def x(self):
        """
        The best solution from the solver
        """
        return self.scale(self.population[0])

    def solve(self):
        """
        Function executes the DifferentialEvolutionSolver
        """
        nit, warning_flag = 0, False
        status_message = _status_message["success"]

        if np.all(np.isinf(self.pop_energies)):
            self.calc_pop_energies()

        # Perform optimization
        for nit in range(1, self.totaliter + 1):
            # Evolve the population by one generation
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message["maxfev"]
                break

            if self.disp:
                print("differential_evolution step %d: f(x)= %g" % (nit, self.pop_energies[0]))

            # Does solver terminate
            convergence = self.convergence

            if (
                self.callback and
                self.callback(self.scale(self.population[0]), convergence=self.tol / convergence,)
                is True
            ):
                warning_flag = True
                status_message = "callback function requested stop early " "by returning True"
                break

            # Check intolerance
            intol = np.std(self.pop_energies) <= self.atol + self.tol * np.abs(np.mean(self.pop_energies))
            if warning_flag or intol:
                break

        else:
            status_message = _status_message["totaliter"]
            warning_flag = True

        de_result = OptimizeResult(
            x=self.x,
            fun=self.pop_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True),
        )

        if self.polish:
            result = minimize(self.func, np.copy(de_result.x), method="L-BFGS-B", bounds=self.limits.T, args=self.args,)

            self._nfev += result.nfev
            de_result.nfev = self._nfev

            if result.fun < de_result.fun:
                de_result.fun = result.fun
                de_result.x = result.x
                de_result.jac = result.jac
                self.pop_energies[0] = result.fun
                self.population[0] = self.unscale(result.x)

        return de_result

    def calc_pop_energies(self):
        """
        Calculate the energies of all the population members, places best member first
        """
        itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        candidates = self.population[:itersize]
        parameters = np.array([self.scale(c) for c in candidates]) 
        energies = self.func(parameters, *self.args)
        self.pop_energies = energies
        self._nfev += itersize

        minval = np.argmin(self.pop_energies)

        # put the lowest energy into the best solution position.
        lowest_energy = self.pop_energies[minval]
        self.pop_energies[minval] = self.pop_energies[0]
        self.pop_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]
    
    def convergence(self):
        """
        Population standard deviation of energies divided by mean
        """
        return np.std(self.pop_energies) / np.abs(np.mean(self.pop_energies) + _MACHEPS)

    def next(self):
        """
        Evolve the population by a single generation
        """
        # Ensure population is initialized
        if np.all(np.isinf(self.pop_energies)):
            self.calc_pop_energies()

        if self.dither is not None:
            dither_diff = self.dither[1] - self.dither[0]
            self.scale = self.random_number_generator.rand() * dither_diff + self.dither[0]

        itersize = max(0, min(self.nb_pop_mem, self.maxfun - self._nfev + 1))

        tests = np.array([self.mutate(c) for c in range(itersize)])
        for test in tests:
            self.enforce_constraint(test)

        parameters = np.array([self.scale(test) for test in tests])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize
        for candidate, (energy, test) in enumerate(zip(energies, tests)):
            # If energy of test candidate is lower than original population member, replace it
            if energy < self.pop_energies[candidate]:
                self.population[candidate] = test
                self.pop_energies[candidate] = energy

                # If test candidate also has a lower energy than the
                # best solution, replace that too
                if energy < self.pop_energies[0]:
                    self.pop_energies[0] = energy
                    self.population[0] = test

        return self.x, self.pop_energies[0]

    def mutate(self, candidate):
        """
        Test vector from mutation strategy
        """
        test = np.copy(self.population[candidate])

        rng = self.random_number_generator

        fill_point = rng.randint(0, self.param_count)

        if self.strategy in ["currenttobest1exp", "currenttobest1bin"]:
            bprime = self.mutation_func(candidate, self.choose_sample_set(candidate, 5))
        else:
            bprime = self.mutation_func(self.choose_sample_set(candidate, 5))

        if self.strategy in self.binomial:
            crossovers = rng.rand(self.param_count)
            crossovers = crossovers < self.cross_over_probability
            crossovers[fill_point] = True

            test = np.where(crossovers, bprime, test)
            return test

        elif self.strategy in self.exponential:
            i = 0
            while i < self.param_count and rng.rand() < self.cross_over_probability:
                test[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.param_count
                i += 1

            return test
    
    def scale(self, test):
        """
        Scale from a number between [0,1] to parameters.
        """
        return self.scale_arg1 + (test - 0.5) * self.scale_arg2

    def unscale(self, parameters):
        """
        Scale from parameters to a number between [0,1].
        """
        return (parameters - self.scale_arg1) / self.scale_arg2 + 0.5

    def enforce_constraint(self, test):
        """
        Enforce parameters to lie between limits
        """
        for index in np.where((test < 0) | (test > 1))[0]:
            test[index] = self.random_number_generator.rand()
    
    def __iter__(self):
        return self

    def _best1(self, samples):
        """
        Mutation strategies: best1bin, best1exp
        """
        r_0, r_1 = samples[:2]

        return self.population[0] + self.scale * (self.population[r_0] - self.population[r_1])

    def _rand1(self, samples):
        """
        Mutation strategies: rand1bin, rand1exp
        """
        r_0, r_1, r_2 = samples[:3]

        return self.population[r_0] + self.scale * (self.population[r_1] - self.population[r_2])

    def _randtobest1(self, samples):
        """
        Mutation strategies: randtobest1bin, randtobest1exp
        """
        r_0, r_1, r_2 = samples[:3]
        bprime = np.copy(self.population[r_0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r_1] - self.population[r_2])

        return bprime

    def _currenttobest1(self, candidate, samples):
        """
        Mutation strategies: currenttobest1bin, currenttobest1exp
        """
        r_0, r_1 = samples[:2]
        bprime = self.population[candidate] + self.scale * (
            self.population[0] - self.population[candidate] + self.population[r_0] - self.population[r_1]
        )

        return bprime

    def _best2(self, samples):
        """
        Mutation strategies: best2bin, best2exp
        """
        r_0, r_1, r_2, r_3 = samples[:4]
        bprime = self.population[0] + self.scale * (self.fill_samples(r_0, r_1, r_2, r_3))

        return bprime

    def _rand2(self, samples):
        """
        Mutation strategies: rand2bin, rand2exp
        """
        r_0, r_1, r_2, r_3, r_4 = samples
        bprime = self.population[r_0] + self.scale * (self.fill_samples(r_1, r_2, r_3, r_4))

        return bprime
    
    def fill_samples(self, a, b, c, d):
        return self.population[a] + self.population[b] - self.population[c] - self.population[d]

    def choose_sample_set(self, candidate, number_samples):
        """
        Chooses and removes random integers from population, except original candidate
        """
        indices = list(range(self.nb_pop_mem))
        indices.remove(candidate)
        self.random_number_generator.shuffle(indices)
        indices = indices[:number_samples]

        return indices

__all__ = ["differential_evolution"]

_MACHEPS = np.finfo(np.float64).eps

def differential_evolution(
    func,
    bounds,
    args=(),
    strategy="best1bin",
    totaliter=1000,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=None,
    disp=False,
    polish=True,
    init="latinhypercube",
    atol=0,
):
    """
    Function executes solve function on given params
    """
    solver = DifferentialEvolutionSolver(
        func,
        bounds,
        args=args,
        strategy=strategy,
        totaliter=totaliter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=polish,
        callback=callback,
        disp=disp,
        init=init,
        atol=atol,
    )

    return solver.solve()
