"""遗忘：记忆强度、衰减扫描与软生命周期迁移。

文档的生命周期（§14）在这里收束：强记忆存续，弱记忆漂向归档，归档后
长期无人问津的最终落入 FORGOTTEN 终态——那仍是软状态（restore() 有效；
本阶段绝不物理删除）。

强度从已存字段即时计算（无写放大，公式调整永远自洽），复用检索层的
归一化辅助函数，使"什么让记忆强"在排序与衰减中含义一致。
"""

from brain_memory.forgetting.decay import (
    DecaySweeper,
    episode_strength,
    last_touched,
    semantic_strength,
)

__all__ = ["DecaySweeper", "episode_strength", "semantic_strength", "last_touched"]
