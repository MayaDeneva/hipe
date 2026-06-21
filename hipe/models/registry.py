# hipe/models/registry.py
_REGISTRY: dict = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_model(name, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
