from golu.tools.schema import validate

SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "n": {"type": "integer"}},
    "required": ["path"],
    "additionalProperties": False,
}


def test_valid() -> None:
    assert validate(SCHEMA, {"path": "a", "n": 2}) is None


def test_missing_required() -> None:
    assert "path" in (validate(SCHEMA, {}) or "")


def test_wrong_type_and_bool_is_not_int() -> None:
    assert validate(SCHEMA, {"path": 1}) is not None
    assert validate(SCHEMA, {"path": "a", "n": True}) is not None


def test_unknown_field_and_non_object() -> None:
    assert validate(SCHEMA, {"path": "a", "x": 1}) is not None
    assert validate(SCHEMA, "nope") is not None
