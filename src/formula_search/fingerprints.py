"""Fingerprints: the values of formulas on one fixed model.

A fingerprint is the value of a formula on one linear Gaussian model of the
complete DAG over its variables, drawn once. Formulas that are one function
of the observed law share their fingerprint, so formulas with different
fingerprints are different functions. Conversely, the complete DAG imposes
no constraint on the law, so different functions differ on almost every one
of its models: formulas with the same fingerprint are one function with high
probability, which a fresh check by HiProF must confirm.

Grouping formulas by fingerprint replaces a check against every class with a
lookup, and a single check for a duplicate; fingerprints never decide that
two formulas are equal. HiProF has no public interface for evaluating a
formula on a given model, so this module uses its internal
``GaussianEvaluator``, on a model drawn from a generator of its own, which
leaves the randomness of HiProF's checks untouched.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Set
from itertools import chain

from flint import fmpq, fmpq_mat
from hiprof.formula import parse_and_validate
from hiprof.formula.formula import Variable
from hiprof.verification.gaussian import GaussianEvaluator, LinearGaussianSCM

Fingerprint = tuple[fmpq, ...]


class FixedModel:
    """A linear Gaussian model of the complete DAG over some variables.

    The DAG orders the variables alphabetically, as every complete DAG has
    the same model. The coefficients and intercepts are nonzero, and the
    noise variances positive, integers of 64 bits drawn from
    ``random.Random(0)``, so the model over the same variables is the same
    in every run.
    """

    def __init__(self, variables: Iterable[str]) -> None:
        rng = random.Random(0)
        names = tuple(sorted(variables))
        n = len(names)

        def draw() -> int:
            return rng.choice((-1, 1)) * (rng.getrandbits(63) + 1)

        scm = LinearGaussianSCM(
            variables=names,
            coefficients=fmpq_mat(
                n,
                n,
                [draw() if j < i else 0 for i in range(n) for j in range(n)],
            ),
            intercepts=fmpq_mat(n, 1, [draw() for _ in names]),
            noise_covariance=fmpq_mat(
                n,
                n,
                [
                    rng.getrandbits(64) + 1 if i == j else 0
                    for i in range(n)
                    for j in range(n)
                ],
            ),
        )
        self._evaluator = GaussianEvaluator(scm)

    def fingerprint(
        self,
        formula: str,
        outputs: Set[str],
        inputs: Set[str],
    ) -> Fingerprint:
        """Return the value of a formula of type ``outputs | inputs``.

        The formula may lack some of the inputs, and its value then does not
        depend on them.

        :raises ValueError: If the formula does not parse.
        """
        kernel = self._evaluator.evaluate(parse_and_validate(formula)).align(
            outputs=tuple(map(Variable, sorted(outputs))),
            inputs=tuple(map(Variable, sorted(inputs))),
        )
        return tuple(
            chain(
                kernel.mean_intercept.entries(),
                kernel.mean_linear.entries(),
                kernel.covariance.entries(),
            )
        )
