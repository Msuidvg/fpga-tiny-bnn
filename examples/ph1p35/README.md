# PH1P35 ROM 回放与 UART 输出

面向原双板系统的接收副板，器件 PH1P35MDG324，速度档 -3，系统时钟 50 MHz。`core` 模式回放 32×32 输入，`pipeline` 模式回放 256×144 二值掩膜并执行 CCL、归一化和 BNN 分类。两种模式使用同一份 V1 权重和三组对拍样本。

## 布线后结果

使用 Anlogic TD 6.2.168116 构建。表内资源是 TD 层级报告的字段，顶层包含回放 ROM、控制逻辑和 UART。

| 模式与统计范围 | logic LUT | seq | ERAM20K | DSP | setup WNS | hold WNS |
|---|---:|---:|---:|---:|---:|---:|
| `core` 顶层 | 1104 | 1784 | 9 | 1 | +8.392 ns | +0.059 ns |
| `core` 的分类核 | 769 | 370 | 8 | 1 | — | — |
| `pipeline` 顶层 | 5298 | 3955 | 18 | 3 | +1.798 ns | +0.022 ns |
| `pipeline` 的识别管线 | 5027 | 3551 | 11 | 3 | — | — |

两种顶层的 setup 和 hold TNS 均为 0，TD 报告的时序覆盖率均为 100%。完整管线内的 CCL 层级占 2994 logic LUT。 [分类核资源](../../reports/ph1p35/core.area) · [完整管线资源](../../reports/ph1p35/pipeline.area) · [分类核时序](../../reports/ph1p35/core.timing) · [完整管线时序](../../reports/ph1p35/pipeline.timing)

| 模式 | UART 记录的周期数 | 按 50 MHz 换算 |
|---|---:|---:|
| `core` | 103154 | 2.06308 ms |
| `pipeline` | 257386 | 5.14772 ms |

周期数由 RTL 仿真从回放开始握手计到控制器采样 `result_valid` 的边沿。`core` 从加载完 32×32 输入后起算，包含控制器采样结果的一拍。`pipeline` 从帧起始起算，包含 ROM 逐像素回放和行间空拍。UART 发送发生在结果采样之后。 [UART 仿真记录](../../reports/ph1p35/uart_simulation.txt)

## 从源码构建

在仓库根目录执行以下命令，按本机安装位置填写 `--td`。输出路径应是一个新目录。

```powershell
python examples/ph1p35/build.py --mode core --td F:/software/TD/bin/td_commands_prompt.exe --output build/ph1p35/core
python examples/ph1p35/build.py --mode pipeline --td F:/software/TD/bin/td_commands_prompt.exe --output build/ph1p35/pipeline
```

输出包括可打开的 `replay.al`、约束、ROM 初始化数据、`routed.area`、`final.timing` 和 `replay.bit`。构建脚本检查时序余量后再生成位流。 [下载已构建位流](https://github.com/Msuidvg/fpga-tiny-bnn/releases/tag/v0.2.0)

## 接线与板端验收

| 信号 | FPGA 引脚 | 接法或用途 |
|---|---|---|
| `I_sys_clk` | C4 | 副板板载 50 MHz |
| `I_reset_n` | B9 | K3，按下时低电平 |
| `O_uart_tx` | V4 | 副板 TX 接 3.3 V USB-TTL 转接器的 RX，共地 |

核对板型和引脚后，用 BitWriter 下载所选模式的位流。按 K3 复位并释放。UART 为 115200 baud、8N1。将 `COM5` 换成设备管理器显示的端口，`--mode` 与下载的位流对应。

```powershell
python -m pip install pyserial
python examples/ph1p35/replay.py --mode pipeline --output build/uart-check --port COM5
```

验收命令逐个核对样本编号、分类、五类分数、分数差和周期数。三个样本应收到如下结果。

| 回放编号 | 对拍向量 | 真实类别 | V1 输出类别 | 整数分数 |
|---|---:|---:|---:|---|
| 0 | 0 | 1 | 1 | `[23, -16, -1, 2, 3]` |
| 1 | 79 | 3 | 3 | `[-17, 0, 27, 18, 11]` |
| 2 | 170 | 5 | 4 | `[-13, -8, 11, 42, 7]` |

每条消息固定 30 字节。依次为 `BN` 两字节、回放编号、状态、预测类别、分数差、5 个小端有符号 32 位分数、1 个小端无符号 32 位周期数。状态 `0` 表示成功，`1` 表示标签表溢出，`2` 表示等待结果超时。向量 170 是已知误判，用于验证硬件输出是否复现模型行为。

当前验证证据包括两套布线后报告、位流和 RTL UART 字节对拍。实板连接后以串口验收命令的输出记录为准。
