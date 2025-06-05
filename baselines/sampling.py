import numpy as np
import logging; logging.disable(logging.CRITICAL);
import optuna

from base import Baseline


class Optuna(Baseline):

    def __init__(self, init_obs, bounds, *args):
        super().__init__(init_obs, bounds)
        self.sampler = self._get_sampler()
        self.study = optuna.create_study(sampler=self.sampler)
        self.study.add_trials([
            optuna.create_trial(
                params=dict(zip([f'x{idx}' for idx in range(self.dim)], obs[:self.dim])), 
                distributions=dict(zip([f'x{idx}' for idx in range(self.dim)], [optuna.distributions.FloatDistribution(bounds[0][idx], bounds[1][idx]) for idx in range(self.dim)])),
                value=obs[self.dim]
            ) for obs in init_obs
        ])
        self.trial = None

    def _get_sampler(self):
        raise NotImplementedError
        
    def propose(self):
        self.trial = self.study.ask()
        X_new = []
        lower_bounds, upper_bounds = self.bounds
        for idx, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
            X_new.append(self.trial.suggest_float(f'x{idx}', lower, upper))
        return np.array(X_new), False

    def update(self, new_obs):
        super().update(new_obs)
        self.study.tell(self.trial, new_obs[self.dim])


class Random(Optuna):

    def _get_sampler(self):
        return optuna.samplers.RandomSampler()


class CMAES(Optuna):

    def _get_sampler(self):
        return optuna.samplers.CmaEsSampler()


class TPE(Optuna):

    def _get_sampler(self):
        return optuna.samplers.TPESampler()


class QMC(Optuna):

    def _get_sampler(self):
        return optuna.samplers.QMCSampler()
