from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

_JSON_OBJECT = re.compile(r"\{.*}", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


def extract_json_object(text: str) -> dict[str, object] | None:
    """Return the first JSON object found in *text*, or None."""
    candidates = [text.strip()]
    if match := _JSON_OBJECT.search(text):
        candidates.append(match.group())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_structured_output(text: str, model: type[T], *, label: str) -> T:
    """Parse structured JSON from an LLM response into a Pydantic model."""
    candidates = [text.strip()]
    if match := _JSON_OBJECT.search(text):
        candidates.append(match.group())

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return model.model_validate_json(candidate)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            try:
                return model.model_validate(data)
            except ValidationError as nested:
                last_error = nested

    msg = f"Could not parse {label} from LLM response"
    raise ValueError(msg) from last_error
