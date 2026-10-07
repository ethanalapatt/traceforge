"""Built-in demo task, so ``reproduce`` works from any directory (same content as
``examples/normalize_name.json``)."""

NORMALIZE_NAME_TASK = {
    "schema_version": 1,
    "id": "normalize_name",
    "input_columns": ["full_name"],
    "examples": [
        {"inputs": ["  Ada Lovelace  "], "output": "ada.lovelace"},
        {"inputs": ["Grace  Hopper"], "output": "grace..hopper"},
        {"inputs": [" Alan Turing"], "output": "alan.turing"},
        {"inputs": ["MARGARET HAMILTON "], "output": "margaret.hamilton"},
    ],
}
