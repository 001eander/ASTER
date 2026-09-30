"""因子入库 CLI：把过门控的内循环产物登记进 ``factor_library/``。

内循环产物停在 ``runs/<ts>/eb/solutions/`` 与 ``best/``，本脚本把其中过门控的
``factor.py`` 复制进因子库并登记 registry，是人工确认后执行的一条命令，不挂在
daily pipeline 上。规则与退出码见 :mod:`quant.factor_lib.promote`。

用法::

    uv run python scripts/promote_factor.py \\
      --factor runs/20260930T101500/eb/solutions/0007/solution/factor.py \\
      --score runs/20260930T101500/eb/solutions/0007/score.json \\
      --factor-id vol_ratio_5_20_neg \\
      --hypothesis "5/20 日均量比取负，缩量做多" \\
      --signal-source volume --time-scale short --mechanism volume_ratio \\
      --op mutation --parent vol_ratio_5_20 \\
      --run-id 20260930T101500 --generation 2

``score.json`` 的 ``details.gate_passed`` 不为 ``true`` 时直接拒绝入库；``factor_id``
重复或目标文件已存在同样拒绝。成功后人工检查摘要，再 ``git add factor_library/`` 提交。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.factor_lib.promote import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
