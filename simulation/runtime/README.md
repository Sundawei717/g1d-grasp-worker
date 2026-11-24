# 相机与仿真桥接源码

[代码导航](../../docs/code-map.md) · [仿真环境与数据交接](../operations.md)

这里收录当前数采与闭环评测直接依赖的五个 `projects.g1d_rsi` 模块，来源是实验服务器上的项目代码。模块路径保持不变，便于保留原来的导入语句。

| 文件 | 用途 |
|---|---|
| [cameras.py](projects/g1d_rsi/cameras.py) | 读取相机配置、挂载相机、核对内外参和初始化设置 |
| [camera_capture.py](projects/g1d_rsi/camera_capture.py) | 读取当前控制步的 RGB，核对帧 ID、原生参考时间和仿真时间戳 |
| [native_render.py](projects/g1d_rsi/native_render.py) | 在明确的仿真时钟边界渲染相机画面 |
| [simulation_clock.py](projects/g1d_rsi/simulation_clock.py) | 读取、验证仿真时钟与时间步关系 |
| [sim60_legacy_bridge.py](projects/g1d_rsi/sim60_legacy_bridge.py) | 兼容当前 Isaac Sim / Isaac Lab 的原生对象接口 |

已配置 Isaac Sim 环境的新机器可以将这个目录加入 Python 模块搜索路径：

```bash
# 在克隆的 GitHub 仓库根目录执行；已有 Isaac 环境初始化脚本先运行。
export G1D_REPO_ROOT="$PWD"
export PYTHONPATH="$G1D_REPO_ROOT/simulation/runtime${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import projects.g1d_rsi.camera_capture as m; print(m.__file__)'
```

输出应指向这里的源码。若启动包装脚本再次设置 `PYTHONPATH`，需检查最终进程实际导入的文件位置。Isaac Sim/Isaac Lab、本机 Vulkan/CUDA 支持、资产和标定仍需单独安装/交接；该目录不是仿真器发行包。
