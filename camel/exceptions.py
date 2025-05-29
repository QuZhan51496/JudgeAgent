#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 14:46
@Author  : alexanderwu
@File    : exceptions.py
"""


import asyncio
import functools
import traceback
from typing import Any, Callable, Tuple, Type, TypeVar, Union

ReturnType = TypeVar("ReturnType")


def handle_exception(
    _func: Callable[..., ReturnType] = None,
    *,
    exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    exception_msg: str = "",
    default_return: Any = None,
) -> Callable[..., ReturnType]:
    """handle exception, return default value"""

    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
