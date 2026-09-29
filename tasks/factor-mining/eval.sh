#!/usr/bin/env bash
# hyra-pi 沙盒评分入口。
#
# 沙盒契约（src/sandbox.ts）：
#   宿主把方案目录复制为 /work/solution、任务目录复制为 /work/task，
#   本脚本复制到 /work/eval.sh，然后执行
#     docker run --rm --network none -v <workDir>:/work -w /work <image> \
#       bash -lc "bash /work/eval.sh /work/solution"
#   因此：
#     - $1 是方案目录（容器内固定为 /work/solution），其中应含 factor.py。
#     - 数据只读挂载在 /data，可用 FACTOR_DATA_DIR 覆盖。
#     - 退出码 0 时须在 /work/score.json 写出
#         {"score": float, "higher_is_better": true, "notes": "..."}
#       并由 harness 读取；退出码非 0 记 ok=false。
#
# 本脚本只是 quant.eval.factor 管线的薄包装，评分逻辑全在 Python 侧。
set -euo pipefail

solution_dir="${1:-}"

if [[ -z "${solution_dir}" || ! -f "${solution_dir}/factor.py" ]]; then
  echo "用法：bash eval.sh <方案目录>；目录下需含 factor.py（收到：${solution_dir:-<空>}）" >&2
  exit 1
fi

data_dir="${FACTOR_DATA_DIR:-/data}"
score_out="${SCORE_OUT:-/work/score.json}"

python -m quant.eval.factor "${solution_dir}/factor.py" \
  --data-dir "${data_dir}" \
  --out "${score_out}"
