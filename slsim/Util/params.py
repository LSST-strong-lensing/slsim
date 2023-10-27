"""
Utilities for managing parameter defaults and validation in the slsim package.
Desgined to be unobtrusive to use.
"""
from functools import wraps
from inspect import getsourcefile, getargspec
from pathlib import Path
from importlib import import_module
from typing import Callable

class SlSimParameterException(Exception):
    pass

_defaults = {}

def check_params(init_fn: Callable) -> Callable:
    """
    A decorator for enforcing checking of params in __init__ methods. This
    decorator will automatically load the default parameters for the class
    and check that the passed parameters are valid. It expeects a "params.py"
    file in the same folder as the class definition. Uses pydantic models
    to enforce types, sanity checks, and defaults.

    From and end user perspective, there is no difference between this and a normal
    __init__ fn. Developers only need to add @check_params above their __init__
    method definition to enable this feature, then add their default parameters
    to the "params.py" file.
    """


    if not init_fn.__name__.startswith('__init__'):
        raise SlSimParameterException('pcheck decorator can currently only be used'\
                                    ' with__init__ methods')

    @wraps(init_fn)
    def new_init_fn(obj, *args, **kwargs):
        # Get function argument names
        all_args = {}
        if args:
            largs = getargspec(init_fn).args
            for i in range(len(args)):
                all_args[largs[i+1]] = args[i]
        all_args.update(kwargs)
        parsed_args = get_defaults(init_fn)(**all_args)
        return init_fn(obj, **dict(parsed_args))
    return new_init_fn


def get_defaults(init_fn):
    path = getsourcefile(init_fn)
    obj_name = init_fn.__qualname__.split('.')[0]
    start = path.rfind("slsim")
    modpath = path[start:].split('/')
    modpath = modpath[1:-1] + ["params"]
    modpath = ".".join(["slsim"] + modpath)
    # Unfortunately, there doesn't seem to be a better way of doing this.

    if modpath not in _defaults:
        #Little optimization. We cache defaults so we don't have to reload them
        # every time we construct a new object.
        _defaults[modpath] = load_parameters(modpath, obj_name)
    return _defaults[modpath]

def load_parameters(modpath, obj_name):
    """
    Loads parameters from the "params.py" file which should be in the same folder
    as the class definition.
    """
    try:
        defaults = import_module(modpath)
    except ModuleNotFoundError:
        raise SlSimParameterException('No default parameters found in module '\
                                    f'\"{modpath[-2]}\"')
    try:
        obj_defaults = getattr(defaults, obj_name)
    except AttributeError:
        raise SlSimParameterException(f'No default parameters found for class '\
                                    f'\"{obj_name}\"')
    return obj_defaults
