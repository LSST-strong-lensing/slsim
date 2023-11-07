"""Utilities for managing parameter defaults and validation in the slsim package.

Desgined to be unobtrusive to use.
"""
from functools import wraps
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
    """Enum for the different types of functions we can have. This is used to determine
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
    """Determine which of the three possible cases a function falls into. Cases 0 and 2
    are actually functionally identical. Things only get spicy when we have a "self"
    argument.

    However the tricky thing is that decorators operate on functions and methods when
    they are imported, not when they are used. This means "inspect.ismethod" will always
    return False, even if the function is a method.

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
    if fn_type == _FnType.STANDARD:
        new_fn = standard_fn_wrapper(fn)
    elif fn_type == _FnType.METHOD:
        new_fn = method_fn_wrapper(fn)
    elif fn_type == _FnType.CLASSMETHOD:
        new_fn = standard_fn_wrapper(fn)

    return new_fn


def standard_fn_wrapper(fn: Callable) -> Callable:
    """A wrapper for standard functions.

    This is used to parse the arguments to the function and check that they are valid.
    """

    @wraps(fn)
    def new_fn(*args, **kwargs) -> Any:
        # Get function argument names
        pargs = {}
        if args:
            largs = list(inspect.signature(fn).parameters.keys())
            for i in range(len(args)):
                arg_value = args[i]
                if arg_value is not None:
                    pargs[largs[i]] = args[i]
        # Doing it this way ensures we still catch duplicate arguments
        defaults = get_defaults(fn)
        parsed_args = defaults(**pargs, **kwargs)
        return fn(**dict(parsed_args))

    return new_fn


def method_fn_wrapper(fn: Callable) -> Callable:
    @wraps(fn)
    def new_fn(obj: Any, *args, **kwargs) -> Any:
        # Get function argument names
        parsed_args = {}
        if args:
            largs = list(inspect.signature(fn).parameters.keys())

            for i in range(len(args)):
                arg_value = args[i]
                if arg_value is not None:
                    parsed_args[largs[i + 1]] = arg_value
        # Doing it this way ensures we still catch duplicate arguments
        parsed_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        defaults = get_defaults(fn)
        parsed_args = defaults(**parsed_args, **parsed_kwargs)
        return fn(obj, **dict(parsed_args))

    return new_fn


def get_defaults(fn: Callable) -> pydantic.BaseModel:
    module_trace = inspect.getmodule(fn).__name__.split(".")
    file_name = module_trace[-1]
    parent_trace = module_trace[:-1]
    parent_path = ".".join(parent_trace)
    param_path = ".".join([parent_path, "_params"])
    fn_qualname = fn.__qualname__
    cache_name = parent_path + "." + fn_qualname
    if cache_name in _defaults:
        return _defaults[cache_name]

    try:
        _ = import_module(param_path)
    except ModuleNotFoundError:
        raise SlSimParameterException(
            f'No default parameters found in module {".".join(parent_trace)},'
            " but something in that module is trying to use the @check_params decorator"
        )
    try:
        param_model_file = import_module(f"{param_path}.{file_name}")
    except AttributeError:
        raise SlSimParameterException(
            f'No default parameters found for file "{file_name}" in module '
            f'{".".join(parent_trace)}, but something in that module is trying to use '
            "the @check_params decorator"
        )

    if fn.__name__ == "__init__":
        expected_model_name = "_".join(fn_qualname.split(".")[:-1])
    else:
        expected_model_name = "_".join(fn_qualname.split("."))

    try:
        parameter_model = getattr(param_model_file, expected_model_name)
    except AttributeError:
        raise SlSimParameterException(
            "No default parameters found for function " f'"{fn_qualname}"'
        )
    if not issubclass(parameter_model, pydantic.BaseModel):
        raise SlSimParameterException(
            f'Defaults for "{fn_qualname}" are not in a pydantic model!'
        )
    _defaults[cache_name] = parameter_model
    return _defaults[cache_name]


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
