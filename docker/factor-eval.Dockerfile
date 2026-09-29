# syntax=docker/dockerfile:1
# 因子挖掘沙盒镜像（issue #21）。
#
# 只装评估链依赖（polars / pyarrow / numpy / radon），版本与仓库 uv.lock 对齐；
# 不含 akshare / autogluon / torch（沙盒禁网，且因子评估用不到），镜像保持小、启动快。
# quant/ 包打入 /app 并设 PYTHONPATH，eval.sh 以 `python -m quant.eval.factor` 调用。
# 数据不打包进镜像，运行时只读挂载到 /data（见 docker/README.md）。
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "polars==1.44.2" \
    "pyarrow==24.0.0" \
    "numpy==2.5.3" \
    "radon==6.0.1"

WORKDIR /app
COPY quant/ /app/quant/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 无 ENTRYPOINT：hyra-pi 沙盒以 `bash -lc "bash /work/eval.sh /work/solution"` 驱动。
