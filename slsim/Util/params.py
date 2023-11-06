"""Utilities for managing parameter defaults and validation in the slsim package.

Desgined to be unobtrusive to use.
"""
from functools import wraps
from inspect import getsourcefile, getargspec
from importlib import import_module
from typing import Callable, Any
from enum import Enum
import inspect
import pydantic

"""
Set of routines for validating inputs to functions and classes. The elements of this
module should never be imported directly. Instead, @check_params can be imported 
directly from the Util module.
"""

class SlSimParameterException(Exception):
    pass


_defaults = {}

class _FnType(Enum):
    """
    
    Enum for the different types of functions we can have. This is used to determine
    how to parse the arguments to the function.
    
    There are three possible cases:
        1. The function is a standard function, defined outside a class
        2. The function is a standard object method, 
           taking "self" as the first parameter
        3. The funtion is a class method (or staticmethod), not taking 
           "self" as the first parameter

    """

    STANDARD = 0
    METHOD = 1
    CLASSMETHOD = 2

def determine_fn_type(fn: Callable) -> _FnType:
    """
    Determine which of the three possible cases a function falls into. Cases
    0 and 2 are actually functionally identical. Things only get spicy when we
    have a "self" argument.

    However the tricky thing is that decorators operate on functions and methods when
    they are imported, not when they are used. This means "inspect.ismethod" will 
    always return False, even if the function is a method.

    We can get around this by checking if the parent of the function is a class. Then,
    we check if the first argument of the function is "self". If both of these are true,
    then the function is a method.
    """
    if not inspect.isfunction(fn):
        raise TypeError("decorator @check_params can only be used on functions!")
    qualified_obj_name = fn.__qualname__
    qualified_obj_path = qualified_obj_name.split(".")
    if len(qualified_obj_path) == 1:
        # If the qualified name isn't split, this is a standard function not
        # attached to a class
        return _FnType.STANDARD
    
    spec = inspect.getfullargspec(fn)
    if spec.args[0] == "self":
        return _FnType.METHOD
    else:
        return _FnType.CLASSMETHOD


def check_params(fn: Callable) -> Callable:
    """A decorator for enforcing checking of params in __init__ methods. This decorator
    will automatically load the default parameters for the class and check that the
    passed parameters are valid. It expeects a "params.py" file in the same folder as
    the class definition. Uses pydantic models to enforce types, sanity checks, and
    defaults.

    From and end user perspective, there is no difference between this and a normal
    __init__ fn. Developers only need to add @check_params above their __init__ method
    definition to enable this feature, then add their default parameters to the
    "params.py" file.

    """
    fn_type = determine_fn_type(fn)



    @wraps(init_fn)
    def new_init_fn(obj: Any, *args, **kwargs) -> Any:
        # Get function argument names
        pargs = {}
        if args:
            largs = getargspec(init_fn).args
            for i in range(len(args)):
                pargs[largs[i + 1]] = args[i]
        # Doing it this way ensures we still catch duplicate arguments
        parsed_args = get_defaults(init_fn)(**pargs, **kwargs)
        return init_fn(obj, **dict(parsed_args))

    return new_init_fn


def get_defaults(init_fn: Callable) -> pydantic.BaseModel:
    path = getsourcefile(init_fn)
    obj_name = init_fn.__qualname__.split(".")[0]
    start = path.rfind("slsim")
    modpath = path[start:].split("/")
    modpath = modpath[1:-1] + ["params"]
    modpath = ".".join(["slsim"] + modpath)
    # Unfortunately, there doesn't seem to be a better way of doing this.

    if modpath not in _defaults:
        # Little optimization. We cache defaults so we don't have to reload them
        # every time we construct a new object.
        _defaults[modpath] = load_parameters(modpath, obj_name)
    return _defaults[modpath]


def load_parameters(modpath: str, obj_name: str) -> pydantic.BaseModel:
    """Loads parameters from the "params.py" file which should be in the same folder as
    the class definition."""
    try:
        defaults = import_module(modpath)
    except ModuleNotFoundError:
        raise SlSimParameterException(
            "No default parameters found in module " f'"{modpath[-2]}"'
        )
    try:
        obj_defaults = getattr(defaults, obj_name)
    except AttributeError:
        raise SlSimParameterException(
            f"No default parameters found for class " f'"{obj_name}"'
        )
    if not issubclass(obj_defaults, pydantic.BaseModel):
        raise SlSimParameterException(
            f'Defaults for "{obj_name}" are not in a ' "pydantic model!"
        )
    return obj_defaults
