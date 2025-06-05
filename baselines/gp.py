"""
This script contains our BO implementations.
"""

import numpy as np
import torch
import gpytorch.settings as gpts
from contextlib import ExitStack

from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize

from botorch.utils.sampling import draw_sobol_samples
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition.utils import get_optimal_samples
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.joint_entropy_search import qJointEntropySearch

from base import Baseline
from single_task_gp import SingleTaskGP

device = 'cpu'

class GP(Baseline):
    def __init__(self, init_obs, bounds, use_rbf_kernel=True):
        super().__init__(init_obs, bounds)
        self.y = -self.y  # maximize

        self.use_rbf_kernel = use_rbf_kernel
        self.bounds_np = np.array(bounds)  # shape [2, d]
        self.lower_bounds = torch.tensor(self.bounds_np[0], dtype=torch.float64, device=device)
        self.upper_bounds = torch.tensor(self.bounds_np[1], dtype=torch.float64, device=device)

        self.unit_bounds = torch.stack([
            torch.zeros_like(self.lower_bounds),
            torch.ones_like(self.upper_bounds)
        ])

        self.gp = None
        self.mll = None
        self._fit_model(self.X, self.y)

    def _to_torch(self, X):
        return torch.tensor(X, dtype=torch.float64, device=device)

    def _to_numpy(self, X):
        return X.detach().cpu().numpy()

    def _normalize(self, X):
        return (X - self.lower_bounds) / (self.upper_bounds - self.lower_bounds + 1e-9)

    def _unnormalize(self, X):
        return X * (self.upper_bounds - self.lower_bounds + 1e-9) + self.lower_bounds

    def _fit_model(self, X, y):
        X_t = self._normalize(self._to_torch(X))
        y_t = self._to_torch(y).unsqueeze(-1)
        self.gp = SingleTaskGP(
            X_t,
            y_t,
            covar_module=None,
            outcome_transform=Standardize(m=1),
            use_rbf_kernel=self.use_rbf_kernel
        ).to(device)
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(self.mll)

    def _get_acq(self):
        raise NotImplementedError

    def propose(self):
        acq = self._get_acq()
        new_x, _ = optimize_acqf(
            acq,
            bounds=self.unit_bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        return self._to_numpy(self._unnormalize(new_x.squeeze(0))), False

    def update(self, new_obs):
        super().update(new_obs)
        self.y[-1] = -self.y[-1]
        self._fit_model(self.X, self.y)


class GP_EI(GP):
    def _get_acq(self):
        return ExpectedImprovement(self.gp, best_f=self.y.max())


class GP_UCB(GP):
    def _get_acq(self):
        return UpperConfidenceBound(self.gp, beta=0.1)


class GP_LogEI(GP):
    def _get_acq(self):
        return LogExpectedImprovement(self.gp, best_f=self.y.max())


class GP_TS(GP):
    def propose(self):
        X_cand = draw_sobol_samples(
            bounds=self.unit_bounds,
            n=1000,
            q=1,
            seed=len(self.X),
        ).squeeze(-2).to(device)

        with ExitStack() as es:
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(True))
            es.enter_context(gpts.minres_tolerance(2e-3))
            es.enter_context(gpts.num_contour_quadrature(15))

            sampler = MaxPosteriorSampling(model=self.gp, replacement=False)
            X_next = sampler(X_cand, num_samples=1)

        return self._to_numpy(self._unnormalize(X_next.squeeze(0))), False


class GP_MES(GP):
    def _get_acq(self):
        candidate_set = torch.rand(1000, self.X.shape[-1], device=device)
        return qLowerBoundMaxValueEntropy(
            model=self.gp,
            candidate_set=candidate_set,
        )


class GP_JES(GP):
    def _get_acq(self):
        num_samples = 32
        optimal_inputs, optimal_outputs = get_optimal_samples(
            self.gp,
            bounds=self.unit_bounds,
            num_optima=num_samples,
            num_restarts=5,
        )
        return qJointEntropySearch(
            model=self.gp,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
        )
