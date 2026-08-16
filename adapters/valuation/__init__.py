"""hypothesis.json -> a program brief for the valuation model.

The ROI stage takes a program brief and returns rNPV, protected years, payer
access and a decision grade. ``emit(bundle, frame)`` writes its input.

The frame is mandatory and its four year fields start null: currency,
geography, route, launch year and above all the patent filing year are analyst
decisions, not graph findings, and a guess is indistinguishable from a sourced
value once it is in the JSON.

Expect NOT_DECISION_GRADE. A literature graph has no epidemiology, no payer
behaviour and no price, so the emitted program is honestly full of holes and
the ROI model's job is to name them. That gap list is the deliverable.

See INPUT_SCHEMA.md and OUTPUT_SCHEMA.md in this directory.
"""

from adapters.valuation.program import Emission, ProgramFrame, emit, program_input

__all__ = ["Emission", "ProgramFrame", "emit", "program_input"]
