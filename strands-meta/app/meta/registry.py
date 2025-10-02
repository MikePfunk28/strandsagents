from typing import Dict, Callable
REGISTRY: Dict[str, Callable] = {}
def register(name:str, fn:Callable): REGISTRY[name]=fn
def get(name:str): return REGISTRY.get(name)
