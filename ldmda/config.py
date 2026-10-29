"""Config loading: base.yaml + method yaml + --set overrides."""
import copy
import os

import yaml


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    v = yaml.safe_load(value)             # "81" -> 81, "[1,2]" -> list, "true" -> True
    if isinstance(v, str):
        try:
            v = float(v)                  # YAML leaves "1e-3" as a string
        except ValueError:
            pass
    d[keys[-1]] = v


def load_config(method_cfg, base_cfg=None, overrides=()):
    """Merge base config, method config and 'a.b.c=value' overrides."""
    if base_cfg is None:
        base_cfg = os.path.join(os.path.dirname(method_cfg), "base.yaml")
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    with open(method_cfg) as f:
        cfg = _merge(cfg, yaml.safe_load(f))
    for o in overrides:
        key, value = o.split("=", 1)
        _set(cfg, key, value)
    return cfg


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
