"""Read packaged experiment settings without importing a GPU framework."""

import json
import math
from importlib.resources import files

MODELS = ("llama31", "llama2", "mistral", "qwen25", "qwen3")
METHODS = ("dense", "teal", "larosa", "roshift")


def asset(*parts):
    return json.loads(files("roshift").joinpath("assets", *parts).read_text(encoding="utf-8"))


def model_config(model):
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    return asset("models", model + ".json")


def recipe(model, allocation, sparsity):
    model_config(model)
    if allocation not in ("uniform", "nonuniform") or sparsity not in (40, 50, 60):
        raise ValueError("Accuracy recipes cover uniform/nonuniform at 40, 50, 60 only")
    return asset("recipes", f"{model}_{allocation}_s{sparsity}.json")


def validate_recipe(data):
    layers = model_config(data["model"])["layers"]
    if len(data["alpha"]) != layers or len(data["recipe"]) != layers:
        raise ValueError("Recipe layer count does not match the model")
    for row in data["alpha"]:
        if len(row) != 6 or any(not math.isfinite(x) or x < 0 for x in row):
            raise ValueError("Invalid six-component Q mixture")
        if abs(sum(row) - 1) > 1e-6:
            raise ValueError("Q mixture must sum to one")
    for row in data["recipe"]:
        if len(row) != 7 or any(not math.isfinite(x) or not 0 <= x <= 100 for x in row):
            raise ValueError("Invalid seven-projection sparsity recipe")
        if data["allocation"] == "uniform" and any(abs(x-data["target"]) > 1e-6 for x in row):
            raise ValueError("Uniform recipe contains a nonuniform projection")
    replay = data.get("replay_layers", list(range(layers)))
    if len(set(replay)) != len(replay) or any(i not in range(layers) for i in replay):
        raise ValueError("Invalid Q replay layer sequence")
    return True


def validate_catalog():
    count = 0
    for model in MODELS:
        for allocation in ("uniform", "nonuniform"):
            for sparsity in (40, 50, 60):
                validate_recipe(recipe(model, allocation, sparsity))
                count += 1
    return {"models": len(MODELS), "recipes": count, "valid": True}
