# docker/：因子挖掘沙盒镜像（issue #21）

`factor-eval.Dockerfile` 构建 `aster-factor-eval` 镜像，供 hyra-pi 内循环沙盒使用：
Python 3.12 + polars / pyarrow / numpy / radon（版本与 uv.lock 对齐），`quant/` 包打在
`/app` 并设好 `PYTHONPATH`。镜像不含数据、不含 akshare / autogluon / torch。

## 构建

```bash
docker build -f docker/factor-eval.Dockerfile -t aster-factor-eval .
```

## 数据只读挂载

行情数据不入镜像，运行时把宿主 `data/` 只读挂到 `/data`：

```bash
docker run --rm --network none \
  -v /path/to/ASTER/data:/data:ro \
  -v <workDir>:/work -w /work \
  aster-factor-eval bash -lc "bash /work/eval.sh /work/solution"
```

内循环里由编排层自动完成：`HYRA_PI_EXTRA_MOUNTS` 环境变量声明额外挂载，单条格式同
`docker -v`（`<宿主路径>:<容器路径>:ro`），多条用分号分隔：

```powershell
$env:HYRA_PI_IMAGE = "aster-factor-eval"
$env:HYRA_PI_EXTRA_MOUNTS = "C:\Users\陶唐\Workspace\ASTER\data:/data:ro"
node dist/cli.js run --task tasks/factor-mining
```

## 手动冒烟（不经内循环）

```bash
mkdir work && cp -r tasks/factor-mining work/task && cp factor_library/mom_5.py work/solution/factor.py  # 示意
docker run --rm --network none -v ${PWD}/work:/work -v ${PWD}/data:/data:ro -w /work \
  aster-factor-eval bash -lc "bash /work/task/eval.sh /work/solution"
cat work/score.json
```
