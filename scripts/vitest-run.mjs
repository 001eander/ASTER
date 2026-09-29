// vitest 启动包裹：rolldown 依 `process.config.variables.shlib_suffix === "dll.a"`
// 识别 MSVC / MinGW。msys64 的 MinGW Node 会被分到 win32-x64-gnu 分支，而 rolldown
// 并未发布该平台的原生绑定，加载链最终抛「Cannot find native binding」。
// napi-rs 约定 NAPI_RS_NATIVE_LIBRARY_PATH 作为最高优先级的绑定覆盖项，这里在
// 检测到 MinGW Node 时把它指向已安装的 win32-x64-msvc 绑定（napi 为纯 C ABI，
// MinGW Node 可正常加载 MSVC 编译的 .node）。官方 MSVC 版 Node 与 POSIX 平台
// 不满足条件，行为保持不变。
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const isMingwNode = process.config?.variables?.shlib_suffix === "dll.a";
const binding = fileURLToPath(
  new URL(
    "../node_modules/@rolldown/binding-win32-x64-msvc/rolldown-binding.win32-x64-msvc.node",
    import.meta.url,
  ),
);

if (isMingwNode && !process.env.NAPI_RS_NATIVE_LIBRARY_PATH && existsSync(binding)) {
  process.env.NAPI_RS_NATIVE_LIBRARY_PATH = binding;
}

const vitest = fileURLToPath(new URL("../node_modules/vitest/vitest.mjs", import.meta.url));
const child = spawn(process.execPath, [vitest, ...process.argv.slice(2)], {
  stdio: "inherit",
  env: process.env,
});
child.on("close", (code) => process.exit(code ?? 1));
