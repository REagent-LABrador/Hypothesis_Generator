"""Thin Anthropic client wrapper.

One place that knows about models, effort, refusals, and the call budget, so
the reasoning stages stay about reasoning.

Structured output is not a convenience here -- it is what makes verdicts
machine-comparable across runs, which the validation harness needs. Every stage
returns a pydantic model or raises.
"""

from __future__ import annotations

import os
from typing import TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Opus 5 for both stages. Articulation and critique are the judgement steps of
# the whole system; a cheaper model here shows up directly as wrong verdicts on
# the retrospective set, which is the number being demoed.
MODEL = os.getenv("HYPGEN_MODEL", "claude-opus-5")

# Effort is the main cost/quality dial. `high` is the API default and a
# reasonable setting for both stages; sweep low/medium against a retrospective
# set before assuming you need more.
DEFAULT_EFFORT = os.getenv("HYPGEN_EFFORT", "high")


class RefusalError(RuntimeError):
    """The safety classifiers declined the request.

    Worth handling explicitly rather than letting it surface as an IndexError
    on ``content[0]``: a drug-discovery corpus sits adjacent to categories the
    classifiers watch, so this will fire occasionally on legitimate work.
    Surface it as a per-hypothesis warning with a visible reason rather than
    killing the batch.

    Server-side refusal ``fallbacks`` would be the nicer answer, but they are a
    beta parameter and ``messages.parse`` -- the structured-output helper this
    whole system depends on -- takes no ``betas`` argument (verified against
    anthropic 0.122.0). Reaching for the beta endpoint would mean giving up
    schema-validated output, which is a worse trade than losing one hypothesis
    to a refusal. Revisit when ``beta.messages.parse`` exists.
    """


class BudgetExceeded(RuntimeError):
    """The run hit ``params.budget.max_model_calls``.

    A ceiling, not a target. Hitting it means the record is incomplete, and the
    pipeline records that on the affected hypotheses rather than pretending the
    run finished clean.
    """


class Judge:
    """Every model call in the system goes through here."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        max_calls: int = 40,
        model: str = MODEL,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.max_calls = max_calls
        self.model = model
        self.calls = 0

    @property
    def remaining(self) -> int:
        return max(self.max_calls - self.calls, 0)

    def has_credentials(self) -> bool:
        """Whether this client can actually authenticate.

        The SDK resolves credentials lazily and only complains at request time,
        which turns a missing key into a stack trace forty frames deep, halfway
        through a run, after the deterministic work has already been done and
        thrown away. Checking up front costs nothing.

        ``auth_headers`` is the honest test rather than ``api_key``: it is
        populated for every credential source the client supports, including an
        OAuth profile from ``ant auth login``, which leaves ``api_key`` unset.
        """
        return bool(getattr(self.client, "auth_headers", None))

    def parse(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = 8000,
    ) -> T:
        """Run one structured-output call and return a validated model.

        Note on sampling: Opus 5 rejects ``temperature`` / ``top_p`` / ``top_k``
        outright (HTTP 400), so the "hot to write, cold to judge" split in
        ``RankingParams`` is expressed as *effort* and as prompt wording, not as
        a sampling temperature. Diversity between critics comes from giving each
        one a different lens, which is more effective than resampling the same
        prompt anyway.
        """
        if self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"model call budget exhausted ({self.max_calls} calls)"
            )
        self.calls += 1

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            output_format=schema,
            messages=[{"role": "user", "content": prompt}],
        )

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise RefusalError(f"request refused (category={category})")

        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(
                f"no parsed output (stop_reason={response.stop_reason})"
            )
        return parsed
