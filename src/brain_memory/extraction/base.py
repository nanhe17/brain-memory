"""解析器契约——整个系统的提取边界。

下游的一切（检索因子、Phase 3 巩固分组、模式分离）都消费这里产出的
结构化字段。修改这个 schema 就是修改系统契约，务必谨慎。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from brain_memory.models import ExtractedExperience


@runtime_checkable
class ExperienceParser(Protocol):
    """把原始经验文本转换为结构化 :class:`ExtractedExperience`。"""

    def parse(
        self,
        text: str,
        *,
        source: str = "conversation",
        context: str | None = None,
        timestamp: datetime | None = None,
    ) -> ExtractedExperience:
        """解析 *text*；实现必须无副作用。

        ``timestamp=None`` 表示"现在"（UTC）。实现不应在异常输入上抛错
        ——优雅降级：一条元数据贫弱的记忆也好过丢失的经验。
        """
        ...
