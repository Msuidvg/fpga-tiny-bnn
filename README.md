# FPGA Tiny-BNN

可复现的 FPGA 二值手势识别与软硬件一致性验证工程。包含 NumPy 训练、4567 张已清洗二值样本、冻结 V1 权重、RTL 推理核、最大连通域与 ROI 归一化，以及 Python/RTL 对拍。

从双板视觉与仿生手比赛工程提取，开源范围从二值手部掩膜开始。摄像头、ISP、DDR、HDMI、以太网与舵机板级工程不属于此仓库。

```text
256×144 binary mask
  → largest 8-connected component (31 recyclable labels)
  → 10% margin / clipping / square padding / nearest 32×32
  → 1024 → 96 → 48 → 5 binary neural network
  → integer scores / gesture ID / score margin
```

## 快速验证

需要 Python 3.11 或更新版本。命令均在仓库根目录执行：

```sh
python -m pip install -r requirements.txt
python scripts/run_checks.py
```

完整 RTL 对拍另需 PATH 中有 `iverilog` 和 `vvp`：

```sh
python scripts/run_checks.py --rtl
```

日志和新结果写入 `build/`。GitHub Actions 在 Ubuntu 上运行相同命令，覆盖 48 组核心向量与 204 组 CCL、归一化及完整管线向量。无需开发板和厂商工具即可运行仿真。

## 实现要点

- 推理核每拍执行一个二值乘法，使用 XNOR 匹配计数、整数阈值及有符号输出偏置。
- 权重以 32-bit 分块同步预取，面向块 RAM 推断；V1 逻辑权重共 103152 bit（12894 byte），物理 RAM 占用需由目标器件综合确认。
- CCL 只保存一行标签，回收已结束连通域的标签，并利用相邻采样间隔分阶段合并。
- ROI 归一化与 Python 统一采用 ties-to-even 舍入、边界裁剪、方形补零及最近邻缩放。
- 隐藏层宽度可配置；输入 1024 bit、输出五类及内部计数器宽度仍有固定限制，详见接口文档。

## 指标与边界

| 项目 | V1 结果及含义 |
|---|---|
| 随附数据 | 六个独立采集批次，共 4567 张 32×32 二值掩膜；从历史 4579 张中排除 12 张误采样本 |
| 六折 LOBO | 汇总准确率 78.69%，宏平均召回率 78.35%；历史跨批评估 |
| 全量部署模型拟合 | 96.09%，使用参与训练的数据复算 |
| Python 整数实现 | 历史 4579 张一致；本仓库重新在清洗后的 4567 张上逐层验证 |
| RTL 核历史对拍 | 48/48，包含 20 个模型误分类难例 |
| 完整管线历史对拍 | 204/204，比较归一化图像、分数、类别与置信差 |
| V1 核延迟 | 103153 个时钟；75 MHz 下理论约 1.38 ms，仅指分类核 |

软件与 RTL 一致性不代表识别准确率为 100%。78.69% 和 96.09% 均对应历史 4579 张数据，不作为随附清洗集的新评估结果。V1 保留原冻结权重，清洗集用于重训与当前算术复核；其重新训练结果应另行报告。204 组 parity 数据属于碎裂轮廓诊断域，历史真实标签正确率为 47.55%，未用于训练。困难背景仍是模型短板。本仓库不提供独立核布局布线结果，不将整片系统资源或时序作为该核指标。

冻结模型 SHA-256：`12dd5fe1e6ae5402d0b08fe530f0a2937a59587bb6daebd8943f709c1d3c6717`。

## 训练与导出

```sh
python -m tools.bnn.train_bnn --summary-only
python -m tools.bnn.train_bnn --epochs 240 --seed 20260930 --model build/model.npz --report build/training.json
python -m tools.bnn.export_bnn_fpga --model models/v1/model.npz --output-dir build/fpga
python -m tools.bnn.export_bnn_eram32 --help
```

训练命令重新执行六折 LOBO 和全量训练，耗时随 CPU/BLAS 而变化。正式 V1 的原始种子和配置以 `reports/training_v1.json` 为准；重新训练结果不承诺与历史文件逐字节相同。新模型默认写入 `build/`，保留冻结模型。

## 目录

| 路径 | 内容 |
|---|---|
| `rtl/` | 四个主要模块及推断 RAM 子模块 |
| `sim/` | 四组自动检查 testbench |
| `tools/bnn/` | 训练、整数参考、导出与单元测试 |
| `models/v1/` | 冻结模型及 FPGA 内存文件 |
| `data/` | 清洗后的二值数据与批次统计 |
| `testvectors/` | 核心及完整管线对拍输入和期望值 |
| `docs/` | [接口](docs/INTERFACES.md)、[验证](docs/VERIFICATION.md)、[来源](docs/PROVENANCE.md) |

代码、随附模型与二值数据按 [MIT](LICENSE) 许可提供。NumPy、OpenCV、Icarus Verilog 为独立依赖，遵循各自许可证；仓库未包含厂商专有 IP。
