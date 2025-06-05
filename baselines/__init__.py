from .gp import GP_EI, GP_UCB, GP_LogEI, GP_TS, GP_MES, GP_JES
from .sampling import Random, CMAES, TPE, QMC
from .extra_methods import DE, PSO


algos = {
    'gp-ei': GP_EI,
    'gp-ucb': GP_UCB,
    'gp-logei': GP_LogEI,
    'gp-ts': GP_TS,
    'gp-mes': GP_MES,
    'gp-jes': GP_JES,
    'random': Random,
    'cmaes': CMAES,
    'de': DE,
    'pso': PSO
}
