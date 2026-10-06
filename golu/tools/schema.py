"""A small JSON Schema check for tool inputs.

Model output is untrusted: a streamed tool input can arrive truncated or with
the wrong types. We check the parts of JSON Schema our tools use (object,
required, property types, additionalProperties) before running anything.
"""

from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def validate(schema: dict[str, Any], args: Any) -> str | None:
    """Return a readable problem description, or None if `args` is valid."""
    if not isinstance(args, dict):
        return "Tool input must be a JSON object."
    props: dict[str, Any] = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return f"Missing required field(s): {', '.join(missing)}."
    if schema.get("additionalProperties") is False:
        extra = sorted(set(args) - set(props))
        if extra:
            return f"Unknown field(s): {', '.join(extra)}."
    for key, value in args.items():
        expected = props.get(key, {}).get("type")
        if expected is None:
            continue
        names = expected if isinstance(expected, list) else [expected]
        if not any(_matches(name, value) for name in names):
            return f"Field '{key}' must be {' or '.join(names)}, got {type(value).__name__}."
    return None


def _matches(type_name: str, value: Any) -> bool:
    # bool is a subclass of int in Python; JSON treats them as different types.
    if isinstance(value, bool) and type_name in {"integer", "number"}:
        return False
    return isinstance(value, _TYPES.get(type_name, (object,)))
