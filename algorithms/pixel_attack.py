from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import product
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from six import string_types
from scipy._lib._util import check_random_state
from scipy.optimize.optimize import _status_message
from scipy.optimize import OptimizeResult, minimize

from tqdm import tqdm

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format

from cma import CMAOptions, CMAEvolutionStrategy

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

class PixelThreshold(EvasionAttack):
    attack_params = EvasionAttack.attack_params + ["th", "es", "targeted", "verbose"]
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        th: Optional[int],
        es: int,
        targeted: bool,
        verbose: bool=True,
    ) -> None:

        self._project = True
        self.type_attack = -1
        self.th = th
        self.es = es
        self.targeted = targeted
        self.verbose = verbose
        PixelThreshold.validate_params(self)

        if self.estimator.channels_first:
            self.img_rows = self.estimator.input_shape[-2]
            self.img_cols = self.estimator.input_shape[-1]
            self.img_channels = self.estimator.input_shape[-3]
        else:
            self.img_rows = self.estimator.input_shape[-3]
            self.img_cols = self.estimator.input_shape[-2]
            self.img_channels = self.estimator.input_shape[-1]
    

    def validate_params(self) -> None:
        if self.th is not None:
            if self.th <= 0:
                raise ValueError("Perturbation size `eps` has to be positive.")

        if not isinstance(self.es, int):
            raise ValueError("Flag `es` has to be of type int.")

        if not isinstance(self.targeted, bool):
            raise ValueError("Flag `targeted` has to be of type bool.")

        if not isinstance(self.verbose, bool):
            raise ValueError("Flag `verbose` has to be of type bool.")

        if not isinstance(self.verbose, bool):
            raise ValueError("Argument `verbose` has to be of type bool.")
    

    def generate(self, inputs: np.ndarray, targets: Optional[np.ndarray]=None, total_iter: int=100, **kwargs) -> np.ndarray:
        """
        Function creates adversarial samples, returns them as numpy array
        """
        targets = check_and_transform_label_format(targets, self.estimator.nb_classes, return_one_hot=False)

        if targets is None:
            if self.targeted:
                raise ValueError("Targets need to be given for a targeted attack.")
            targets = np.argmax(self.estimator.predict(inputs), axis=1)
        else:
            if len(targets.shape) > 1:
                targets = np.argmax(targets, axis=1)
        
        if self.th is None:
            logger.info("Minimum perturbation executed for attack.")
        
        if np.max(inputs) <= 1:
            inputs = inputs * 255.0
        
        best_adv_inputs = []
        for image, class_target in tqdm(zip(inputs, targets), desc="Pixel threshold", disable=not self.verbose):
            if self.th is None:
                self.min_th = 127
                begin, end = 1, 127
                while True:
                    result_image = Union[List[np.ndarray], np.ndarray] = []
                    threshold = (begin + end) // 2
                    success, result_trial_image = self.attack(image, class_target, threshold, total_iter)
                    if result_image or success:
                        result_image = result_trial_image
                    if success:
                        self.min_th = threshold
                        end = threshold - 1
                    else:
                        begin = threshold + 1
                    
                    if end < begin:
                        if isinstance(result_image, list) and not result_image:
                            result_image = image
                        break
            else:
                success, result_image = self.attack(image, class_target, self.th, total_iter)
            best_adv_inputs += [result_image]

        best_adv_inputs = np.array(best_adv_inputs)

        if np.max(inputs) <= 1:
            inputs = inputs / 255.0
        
        if targets is not None:
            targets = to_categorical(targets, self.estimator.nb_classes)
        
        logger.info(
            "Success rate of Attack: %.2f%%", 100 * compute_success(self.estimator, inputs, targets, best_adv_inputs, self.targeted, 1),
        )

        return best_adv_inputs


    def bounds(self, image: np.ndarray, limit) -> Tuple[List[list], list]:
        """
        Function determines the bounds for the input image within the given limits

        :param image: Input image as an np.ndarray
        :param limit: Given boundaries
        :return: Boundaries of the image
        """
        min_bounds, max_bounds, bounds, initial = [], [], [], []
        
        for i, j, k in product(range(image.shape[-3]), range(image.shape[-2]), range(image.shape[-1])):
            temp = image[i,j,k]
            initial += [temp]
            bound = bound_limit(temp, limit)
            if self.es == 0:
                min_bounds += [bound[0]]
                max_bounds += [bound[1]]
            else:
                bounds += [bound]
        
        if self.es == 0:
            bounds = [min_bounds, max_bounds]
        
        return bounds, initial
            

    def perturbate(self, x: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Function perturbes the given image with the given x perturbation
        """
        return image


    def is_attack_success(self, adv_inputs, inputs, target_class):
        """
        Function determines whether the given adversarial inputs' perturbation for the
        given image was successful
        """
        pred_class = np.argmax(self.estimator.predict(self.perturbate(adv_inputs, inputs))[0])
        
        return bool(
            (self.targeted and pred_class == target_class) or
            (not self.targeted and pred_class != target_class)
        )


    def attack(self, image: np.ndarray, target_class: np.ndarray, limit: int, total_iter: int) -> Tuple[bool, np.ndarray]:
        """
        Function performs attack on the given image with a threshold limit for the target_class, which is the true label
        for an untargeted attack and the targeted label for a targeted attack
        """
        bounds, initial = self.bounds(image, limit)

        def predict_func(x):
            preds = self.estimator.predict(self.perturbate(x, image))[:, target_class]
            return preds if not self.targeted else 1 - preds
        
        def callback_func(x, convergence=None):
            if self.es == 0:
                if self.is_attack_success(x.result[0], image, target_class):
                    raise Exception("Attack completed.")
            else:
                return self.is_attack_success(x, image, target_class)
        
        if self.es == 0:
            options = CMAOptions()
            if not self.verbose:
                options.set("verbose", -9)
                options.set("verb_disp", 40000)
                options.set("verb_log", 40000)
                options.set("verb_time", False)

            options.set("bounds", bounds)

            if self.type_attack == 0:
                std = 63
            else:
                std = limit

            strategy = CMAEvolutionStrategy(initial, std / 4, options)

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

            adv_x = strategy.result[0]
        else:
            strategy = differential_evolution(
                predict_func,
                bounds,
                verbose=self.verbose,
                maxiter=total_iter,
                popsize=max(1, 400 // len(bounds)),
                recombination=1,
                atol=-1,
                callback=callback_func,
                polish=False,
            )
            adv_x = strategy.x

        if self._attack_success(adv_x, image, target_class):
            return True, self._perturb_image(adv_x, image)[0]
        else:
            return False, image


class PixelAttack(PixelThreshold):
    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        th: Optional[int] = None,
        es: int = 0,
        targeted: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        """
        super().__init__(classifier, th, es, targeted, verbose)
        self.type_attack = 0
    

    def perturbate(self, p: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Function executes perturbation p on the given image
        """
        if p.ndim < 2:
            p = np.array([p])
        images = np.tile(image, [len(p)] + [1] * (p.ndim + 1))
        p = p.astype(int)

        # Loop through
        for adverse, image in zip(p, images):
            for pixel in np.split(adverse, len(adverse) // (2 + self.img_channels)):
                x_pos, y_pos, *rgb = pixel
                if not self.estimator.channels_first:
                    image[x_pos % self.img_rows, y_pos % self.img_cols] = rgb
                else:
                    image[:, x_pos % self.img_rows, y_pos % self.img_cols] = rgb
        
        return images
    

    def bounds(self, image: np.ndarray, limit) -> Tuple[List[list], list]:
        """
        Function determines the bounds of the given image within the given limits
        """
        initial: List[np.ndarray] = []
        bounds: List[List[int]]

        # Using CMAES as Evolutionary Strategy
        if self.es == 0:
            for count, (i, j) in enumerate(product(range(self.img_rows), range(self.img_cols))):
                initial += [i, j]
                for k in range(self.img_channels):
                    if not self.estimator.channels_first:
                        initial += [image[i, j, k]]
                    else:
                        initial += [image[k, i, j]]

                if count == limit - 1:
                    break
                else:
                    continue
            min_bounds = [0, 0]
            for _ in range(self.img_channels):
                min_bounds += [0]
            min_bounds = min_bounds * limit
            max_bounds = [self.img_rows, self.img_cols]
            for _ in range(self.img_channels):
                max_bounds += [255]
            max_bounds = max_bounds * limit
            bounds = [min_bounds, max_bounds]
        # Using DE as Evolutionary Strategy
        else:
            bounds = [[0, self.img_rows], [0, self.img_cols]]
            for _ in range(self.img_channels):
                bounds += [[0, 255]]
            bounds = bounds * limit

        return bounds, initial


"""
The following code is a modification of scipy's differential evolution. Adapted by Dan Kondratyuk, 2018, from 
Andrew Nelson, 2014: "differential_evolution:The differential evolution global optimization algorithm"
"""
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

    init_error_msg = (
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
        total_iter=1000, 
        pop_size=15, 
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        max_fun=np.inf,
        callback_func=None,
        verbose=False,
        polish=True,
        initial="latinhypercube",
        atol=0,
    ):
    
        if strategy in self.binomial:
            self.mutation_func = getattr(self, self.binomial[strategy])
        elif strategy in self.exponential:
            self.mutation_func = getattr(self, self.exponetial[strategy])
        else:
            raise ValueError("Invalid mutation strategy selected.")
        self.strategy = strategy

        self.func = func
        self.args = args
        self.callback_func = callback_func
        self.cross_over_prob = recombination
        self.polish = polish
        self.verbose = verbose
        self.tol = tol
        self.atol = atol

        self.scale = mutation
        if not np.all(np.isfinite(mutation)) or np.any(np.array(mutation) >= 2) or np.any(np.array(mutation) < 0):
            raise ValueError(
            "Mutation constant must be float in U[0,2) or defined as tuple(min,max) where min < max and U[0,2) contains min,max."
        )

        self.dither = None
        if hasattr(mutation, "__iter__") and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()
        
        self.limits = np.array(bounds, dtype="float").T
        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError(
            "Bounds should be sequence with real-valued min,max pairs for each value in x."
        )

        if total_iter is None:
            total_iter = 1000
        if max_fun is None:
            max_fun = np.inf
        self.max_fun = max_fun
        
        self.scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.param_count = np.size(self.limits, 1)
        self.rand_nb_gen = check_random_state(seed)

        self.nb_pop_members = max(5, pop_size + self.param_count)
        self.pop_shape = (self.nb_pop_members, self.param_count)

        self.nfev = 0
        if isinstance(initial, string_types):
            if initial == "latinhypercube":
                self.init_pop_lhs()
            elif initial == "random":
                self.init_pop_rand()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_pop_array(initial)
    

    def init_pop_lhs(self):
        """
        Function intializes population with Latin Hypercube Sampling
        """
        seg_size = 1.0 / self.nb_pop_members
        samples = (
            seg_size * self.rand_nb_gen.random_sample(self.pop_shape) + 
            np.linespace(0.0, 1.0, self.nb_pop_members, endpoint=False)[:, np.newaxis]
        )

        self.pop = np.zeros_like(samples)

        for i in range(self.param_count):
            ord = self.rand_nb_gen.permutation(range(self.nb_pop_members))
            self.pop[:, i] = samples[ord, i]
        
        self.pop_energies = np.ones(self.nb_pop_members) * np.inf
        self.nfev = 0
    

    def init_pop_rand(self):
        """
        Function intializes population at random
        """
        self.pop = self.rand_nb_gen.random_sample(self.pop_shape)
        self.pop_energies = np.ones(self.nb_pop_members) * np.inf
        self.nfev = 0
    

    def init_pop_array(self, initial):
        """
        Function initializes population from specified array
        """
        popf = np.asfarray(initial)

        if np.size(popf, 0) < 5 or popf.shape[1] != self.param_count or len(popf.shape) != 2:
            raise ValueError(
            "Population must have shape (M, len(x)) where M > 4."
        )

        self.pop = np.clip(self._unscale_parameters(popf), 0, 1)
        self.nb_pop_members = np.size(self.pop, 0)
        self.pop_shape = (self.nb_pop_members, self.param_count)
        self.pop_energies = np.ones(self.nb_pop_members) * np.inf
        self.nfev = 0
    

    def best_soln(self):
        """
        Function returns the best solution from the solver
        """
        return self._scale_parameters(self.pop[0])
    

    def convergence(self):
        """
        Function calculates standard deviation of population energies / population mean
        """
        return np.std(self.pop_energies) / np.abs(np.mean(self.pop_energies) + _MACHEPS)
    
    
    def solve(self):
        """
        Function executes DifferentialEvolutionSolver
        """
        nit = 0
        warning_flag = False
        status_message = _status_message["success"]

        if np.all(np.isinf(self.pop_energies)):
            self.calc_pop_energies()
        
        for nit in range(1, self.total_iter + 1):
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message["maxfev"]
                break
                
            if self.verbose:
                print("differential_evolution step %d: f(x)= %g" % (nit, self.pop_energies[0]))
            
            convergence = self.convergence
            if(self.callback_func and self.callback_func(self._scale_parameters(self.pop[0]), convergence=self.tol / convergence,) is True):
                warning_flag = True
                status_message = "callback function requested stop early " "by returning True"
                break
            
            intol =  np.std(self.pop_energies) <= self.atol + self.tol * np.abs(np.mean(self.pop_energies))
            if warning_flag or intol:
                break

        else:
            status_message = _status_message["maxiter"]
            warning_flag = True
        
        diff_ev_result = OptimizeResult(
            x=self.best_soln,
            fun=self.pop_energies[0],
            nfev=self.nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True),
        )

        if self.polish:
            result = minimize(self.func, np.copy(diff_ev_result.x), method="L-BFGS-B", bounds=self.limits.T, args=self.args,)

            self.nfev += result.nfev
            diff_ev_result = self.nfev

            if result.fun < diff_ev_result.fun:
                diff_ev_result.fun = result.fun
                diff_ev_result.x = result.x
                diff_ev_result.jac = result.jac
                self.pop_energies[0] = result.fun
                self.pop[0] = self._unscale_parameters(result.x)
        
        return diff_ev_result


    def calc_pop_energies(self):
        """
        Function calculates the energies of population members, and inserts the best
        population member first
        """
        # Calculate pop energies
        iter_size = max(0, min(len(self.pop), self.max_fun - self.nfev + 1))
        candidates = self.pop[:iter_size]
        params = np.array([self._scale_parameters(p) for p in candidates])
        energies = self.func(params, *self.args)
        self.nfev += iter_size

        min_value = np.argmin(self.pop_energies)

        # Place best member first
        min_energy = self.pop_energies[min_value]
        self.pop_energies[min_value] = self.pop_energies[0]
        self.pop_energies[0] = min_energy

        self.pop[[0, min_value], :] = self.pop[[min_value, 0], :]
    

    def next(self):
        """
        Function wraps _next()
        """
        return self.__next__()
    

    def _next(self):
        """
        Triggers population evolution by one generation
        """
        # Make sure energies are calculated
        if np.all(np.isinf(self.pop_energies)):
            self.calc_pop_energies()
        
        if self.dither is not None:
            self.scale = self.rand_nb_gen() * (self.dither[1] - self.dither[0]) + self.dither
        
        # Format population data
        iter_size = max(0, min(self.nb_pop_members, self.max_fun - self.nfev + 1))
        tests = np.array([self._mutate(i) for i in range(iter_size)])
        for test in tests:
            self.enforce_constraints(test)
        params = np.array([self._scale_parameters(test) for test in tests])

        # Evolve population
        energies = self.func(params, *self.args)
        self.nfev += iter_size

        for candidate, (energy, test) in enumerate(zip(energies, tests)):
            if energy < self.pop_energies[candidate]:
                self.pop[candidate] = test
                self.pop_energies[candidate] = energy

                if energy < self.pop_energies[0]:
                    self.pop_energies[0] = energy
                    self.pop[0] = test
        
        return self.best_soln, self.pop_energies[0]


    def _mutate(self, candidate):
        """
        Function generates a test vector for a mutation strategy
        """
        test = np.copy(self.pop[candidate])
        rng = self.rand_nb_gen
        fill_pt = rng.randint(0, self.param_count)

        if self.strategy in ["currenttobest1exp", "currenttobest1bin"]:
            bprime = self.mutation_func(candidate, self.select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self.select_samples(candidate, 5))

        if self.strategy in self.binomial:
            cross_overs = rng.rand(self.param_count)
            cross_overs = cross_overs < self.cross_over_prob

            cross_overs[fill_pt] = True
            test = np.where(cross_overs, bprime, test)
            return test
        elif self.strategy in self.exponential:
            i = 0
            while i < self.param_count and rng.rand() < self.cross_over_prob:
                test[fill_pt] = bprime[fill_pt]
                fill_pt = (fill_pt + 1) % self.param_count
                i += 1

            return test


    def enforce_constraints(self, test):
        """
        Function enforces the test's parameters are within their limits
        """
        for idx in np.where((test < 0) | (test > 0))[0]:
            test[idx] = self.rand_nb_gen.rand()


    def __iter__(self):
        return self
    

    def _scale_parameters(self, test):
        """
        Scale from normal vector range to params
        """
        return self.scale_arg1 + (test - 0.5) * self.scale_arg2
    
    def _unscale_parameters(self, params):
        """
        Scale from params to normal vector range
        """
        return (params - self.scale_arg1) / self.scale_arg2 + 0.5

    
    def _best1(self, samples):
        r_0, r_1 = samples[:2]
        return self.pop[0] + self.scale * (self.pop[r_0] - self.pop[r_1])


    def _rand1(self, samples):
        r_0, r_1, r_2 = samples[:3]
        return self.pop[r_0] + self.scale * (self.pop[r_1] - self.pop[r_2])


    def _randtobest1(self, samples):
        r_0, r_1, r_2 = samples[:3]
        bprime = np.copy(self.population[r_0])
        bprime += self.scale * (self.pop[0] - bprime)
        bprime += self.scale * (self.pop[r_1] - self.pop[r_2])
        return bprime


    def _currenttobest1(self, candidate, samples):
        r_0, r_1 = samples[:2]
        bprime = self.pop[candidate] + self.scale * (
            self.pop[0] - self.pop[candidate] + self.pop[r_0] - self.pop[r_1]
        )
        return bprime


    def _best2(self, samples):
        r_0, r_1, r_2, r_3 = samples[:4]
        bprime = self.pop[0] + self.scale * (
            self.pop[r_0] + self.pop[r_1] - self.pop[r_2] - self.pop[r_3]
        )

        return bprime


    def _rand2(self, samples):
        r_0, r_1, r_2, r_3, r_4 = samples
        bprime = self.pop[r_0] + self.scale * (
            self.pop[r_1] + self.pop[r_2] - self.pop[r_3] - self.pop[r_4]
        )

        return bprime


    def select_samples(self, candidate, nb_samples):
        """
        Function gets random ints from population members, except the candidate
        """
        indices = list(range(self.nb_pop_members))
        indices.remove(candidate)
        self.rand_nb_gen.shuffle(indices)
        indices = indices[:nb_samples]

        return indices


__all__ = ["differential_evolution"]
_MACHEPS = np.finfo(np.float64).eps

def differential_evolution(
    func, 
    bounds, 
    args=(), 
    strategy="best1bin", 
    total_iter=1000, 
    pop_size=15, 
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback_func=None,
    verbose=False,
    polish=True,
    initial="latinhypercube",
    atol=0,
    ):
    """
    Function calculates the global minimum of multivariate function.

    :param func: Function to be minimized
    :param bounds: Bounds for the variables
    :param args: Additional fixed parameters for func
    :param strategy: Differential evolution strategy to use
    :param total_iter: Total number of iterations over which population is evolved
    :param pop_size: Multiplier for total population
    :param tol: Relative tolerance for convergence
    :param mutation: Differential weight constant. (Increasing widens search radius, slows convergence)
    :param recombination: Crossover probability constant. (Inc -> more mutations, less pop stability)
    :param seed: Seed for randomization
    :param verbose: Toggles display of status messages
    :param callback_func: Function that tracks minimization progress
    :param polish: Toggles scipy.optimize.minimize to polish best population member
    :param intitial: Select type of population initialization
    :param atol: Absolute tolerance for convergence
    :returns: Optimized result
    """
    solver = DifferentialEvolutionSolver(
        func,
        bounds,
        args=args,
        strategy=strategy,
        total_iter=total_iter,
        pop_size=pop_size,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=polish,
        callback_func=callback_func,
        verbose=verbose,
        initial=initial,
        atol=atol
    )

    return solver.solve()


def bound_limit(value, limit):
    return np.clip(value - limit, 0, 255), np.clip(value + limit, 0, 255)
