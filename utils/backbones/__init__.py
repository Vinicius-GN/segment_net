import pkgutil
import importlib
import inspect

from .base import BaseBackbone

__all__ = []

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    
    for attr_name, attr_value in module.__dict__.items():
        if inspect.isclass(attr_value) and issubclass(attr_value, BaseBackbone) and attr_value is not BaseBackbone:
            globals()[attr_name] = attr_value
            __all__.append(attr_name)
            
