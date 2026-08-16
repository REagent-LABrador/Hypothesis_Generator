"""Focused HypGen run plus the existing cards and ROI adapters.

LIVE always instantiates the provider-backed Judge. REPLAY never does. The two
paths share the same output envelope and there is deliberately no fallback
between them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from adapters.common import Bundle
from adapters.valuation.program import ProgramFrame, emit as emit_valuation
from adapters.webui.payload import WebPayload, emit as emit_cards
from hyp_gen.graph import KnowledgeGraph
from hyp_gen.hypothesis import HypothesisDocument
from hyp_gen.params import PROFILES, Params
from hyp_gen.pipeline import Generator
from hyp_gen.reasoning.llm import Judge

ExecutionMode = Literal["LIVE", "REPLAY"]
OutputOrigin = Literal["LIVE_PROVIDER", "DETERMINISTIC_REPLAY"]


class RoiExecution(BaseModel):
    """The current rNPV module's deterministic execution block."""

    model_config = ConfigDict(extra="forbid")

    simulations: int = Field(ge=1, le=100_000)
    seed: int = Field(ge=0, le=4_294_967_295)
    simulation_assumptions: dict[str, Any] = Field(default_factory=dict)


class RoiSettings(BaseModel):
    """Caller-owned fields HypGen cannot derive from a literature graph."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    comparables: list[dict[str, Any]]
    execution: RoiExecution


class HeadlessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: KnowledgeGraph
    focus_thing_id: str = Field(min_length=1)
    profile: str
    valuation_frame: ProgramFrame
    roi: RoiSettings

    @field_validator("profile")
    @classmethod
    def known_profile(cls, value: str) -> str:
        if value not in PROFILES:
            raise ValueError(f"profile must be one of {', '.join(sorted(PROFILES))}")
        return value


class HeadlessError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    message: str


class HeadlessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["COMPLETE", "CANNOT_COMPLETE"]
    execution_mode: ExecutionMode
    output_origin: OutputOrigin
    hypothesis: HypothesisDocument | None = None
    cards: WebPayload | None = None
    roi_request: dict[str, Any] | None = None
    error: HeadlessError | None = None


def _origin(mode: ExecutionMode) -> OutputOrigin:
    return "LIVE_PROVIDER" if mode == "LIVE" else "DETERMINISTIC_REPLAY"


def _terminal(
    mode: ExecutionMode,
    reason_code: str,
    message: str,
    *,
    hypothesis: HypothesisDocument | None = None,
    cards: WebPayload | None = None,
) -> HeadlessResponse:
    return HeadlessResponse(
        status="CANNOT_COMPLETE",
        execution_mode=mode,
        output_origin=_origin(mode),
        hypothesis=hypothesis,
        cards=cards,
        error=HeadlessError(reason_code=reason_code, message=message),
    )


def _provider_error(error: Exception) -> tuple[str, str]:
    message = str(error)
    if re.search(r"api[_ ]?key|credential|authentication", message, re.IGNORECASE):
        return "CREDENTIAL_MISSING", message
    if re.search(r"timed?\s*out|timeout", message, re.IGNORECASE):
        return "PROVIDER_TIMEOUT", message
    return "PROVIDER_ERROR", message


def _judge(
    request: HeadlessRequest,
    mode: ExecutionMode,
    judge_factory: Callable[..., Judge],
) -> Judge | None:
    if mode == "REPLAY":
        return None
    judge = judge_factory(max_calls=Params.profile(request.profile).budget.max_model_calls)
    if not judge.has_credentials():
        raise RuntimeError(
            "no Anthropic API key, auth token, or profile credential resolved"
        )
    return judge


def run(
    payload: object,
    *,
    mode: ExecutionMode,
    judge_factory: Callable[..., Judge] = Judge,
) -> HeadlessResponse:
    """Execute one focused invocation and build its downstream handoffs."""

    try:
        request = HeadlessRequest.model_validate(payload)
    except ValidationError as error:
        return _terminal(mode, "INVALID_REQUEST", str(error))

    thing_ids = {thing.id for thing in request.graph.things}
    if request.focus_thing_id not in thing_ids:
        return _terminal(
            mode,
            "FOCUS_THING_NOT_FOUND",
            f"{request.focus_thing_id!r} is not present in graph {request.graph.graph_id}",
        )

    try:
        judge = _judge(request, mode, judge_factory)
        document = Generator(
            graph=request.graph,
            params=Params.profile(request.profile),
            judge=judge,
            focus_thing_id=request.focus_thing_id,
        ).run().top()
    except Exception as error:  # provider SDK errors vary by transport/version
        if mode == "LIVE":
            reason_code, message = _provider_error(error)
        else:
            reason_code, message = "EXECUTION_ERROR", str(error)
        return _terminal(mode, reason_code, message)

    if document is None:
        return _terminal(
            mode,
            "NO_FOCUSED_HYPOTHESIS",
            (
                f"no hypothesis containing {request.focus_thing_id} survived "
                f"profile {request.profile}"
            ),
        )

    bundle = Bundle.of([document])
    cards = emit_cards(bundle)
    try:
        valuation = emit_valuation(bundle, request.valuation_frame)
    except Exception as error:
        return _terminal(
            mode,
            "ROI_ADAPTER_ERROR",
            str(error),
            hypothesis=document,
            cards=cards,
        )
    if len(valuation.programs) != 1:
        detail = "; ".join(
            f"{item.hypothesis_id}: {item.reason}" for item in valuation.skipped
        ) or "the focused hypothesis is not an intervention-to-disease program"
        return _terminal(
            mode,
            "ROI_PROGRAM_NOT_EMITTED",
            detail,
            hypothesis=document,
            cards=cards,
        )

    roi_request = {
        "contract_version": "1.0.0",
        "module": "rnpv_roi_calculator",
        "request_id": request.roi.request_id,
        "program": valuation.programs[0],
        "comparables": request.roi.comparables,
        "execution": request.roi.execution.model_dump(mode="json"),
    }
    return HeadlessResponse(
        status="COMPLETE",
        execution_mode=mode,
        output_origin=_origin(mode),
        hypothesis=document,
        cards=cards,
        roi_request=roi_request,
    )
