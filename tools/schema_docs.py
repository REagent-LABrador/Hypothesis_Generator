"""Render a pydantic model as markdown field tables.

Schema docs are generated rather than written by hand because a hand-written
one is a promise nobody re-reads. The prose in each SCHEMA.md is authored; the
tables under it come from the models that actually read and write the files, so
the two cannot disagree for longer than one test run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def _type_name(schema: dict[str, Any]) -> str:
    """A short, readable type for the table's second column."""
    if "$ref" in schema:
        return f"[{schema['$ref'].rsplit('/', 1)[-1]}](#{schema['$ref'].rsplit('/', 1)[-1].lower()})"
    if "const" in schema:
        return f"`{schema['const']!r}`"
    if "enum" in schema:
        return " \\| ".join(f"`{v}`" for v in schema["enum"])
    if "anyOf" in schema:
        parts = [_type_name(s) for s in schema["anyOf"] if s.get("type") != "null"]
        optional = any(s.get("type") == "null" for s in schema["anyOf"])
        joined = " \\| ".join(dict.fromkeys(parts))
        return f"{joined} or null" if optional else joined
    kind = schema.get("type")
    if kind == "array":
        return f"list of {_type_name(schema.get('items', {}))}"
    if kind == "object":
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            return f"object of {_type_name(extra)}"
        return "object"
    return f"`{kind}`" if kind else "any"


def _rows(schema: dict[str, Any]) -> list[str]:
    required = set(schema.get("required", []))
    rows = []
    for name, field in schema.get("properties", {}).items():
        description = (field.get("description") or "").replace("\n", " ").strip()
        mark = "**yes**" if name in required else "no"
        rows.append(f"| `{name}` | {_type_name(field)} | {mark} | {description} |")
    return rows


def _section(title: str, schema: dict[str, Any], level: int = 2) -> str:
    heading = "#" * level
    lines = [f"{heading} {title}", ""]
    description = (schema.get("description") or "").strip()
    if description:
        lines += [description, ""]
    rows = _rows(schema)
    if rows:
        lines += ["| Field | Type | Required | Meaning |", "|---|---|---|---|", *rows, ""]
    elif not description:
        lines += ["_No fields._", ""]
    return "\n".join(lines)


def model_markdown(model: type[BaseModel], level: int = 2) -> str:
    """The model, then every type it references, each as a table.

    Nested types are emitted in the order they are first reached, so a reader
    following a link travels down the document rather than back up it.
    """
    schema = model.model_json_schema(ref_template="#/$defs/{model}")
    defs = schema.pop("$defs", {})

    out = [_section(model.__name__, schema, level)]
    for name in sorted(defs):
        out.append(_section(name, defs[name], level))
    return "\n".join(out).rstrip() + "\n"
