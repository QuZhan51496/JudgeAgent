from __future__ import annotations

import asyncio
import json
import os.path
import uuid
from abc import ABC
from asyncio import Queue, QueueEmpty, wait_for
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from camel.exceptions import handle_exception

# 定义一个泛型类型变量
T = TypeVar("T", bound="BaseModel")

class BaseContext(BaseModel, ABC):
    @classmethod
    @handle_exception
    def loads(cls: Type[T], val: str) -> Optional[T]:
        i = json.loads(val)
        return cls(**i)

class Dependency(BaseContext):
    dependency_type: str = ""  # api_call, inheritance, data_flow, etc.
    function: str = ""
    contract: Dict[str, Any] = Field(default_factory=dict)  # 存储接口契约信息
    
class DependencyContract(BaseContext):
    caller: str = ""
    callee: str = ""
    dependencies: List[Dependency] = Field(default_factory=list)

class DependencyNetwork(BaseContext):
    graph: List[DependencyContract] = Field(default_factory=list)
