# 历史运行结果的位置

三个服务器上散落于 `C01Python` 的相关运行材料已按所属项目整理，并完成
源文件和目标文件的逐项 SHA-256 校验。原目录保留，现有项目结果不覆盖。

| 服务器 | C01Python 的绝对路径 |
| --- | --- |
| server197（192.168.191.197） | `/research_data/users/zhangxifeng/C01Python/` |
| server193（192.168.191.193:9088） | `/home/zhangxf/C01Python/` |
| server216（175.102.135.216） | `/data1/zjinzxf2025/C01Python/` |

在各服务器的上述目录下，新增结果分别位于：

- `FM4PDE/outputs/main/organized_20260910/`：主实验、Burgers 实验及采样时间。
- `FM4PDE/outputs/ablations/organized_20260910/`：消融、NS 参数比较、样本平均及相关试运行。
- `FM4PDEbaseline/outputs/organized_20260910/`：基线结果与计时；本次在 197、193 上有对应材料。
- `DiffusionPDE/outputs/organized_20260910/`：DiffusionPDE 的预测、比较、计时及输入生成核验。

193 调试目录中保存的四份模型文件另存于
`FM4PDE/outputs/pretrained/organized_20260910/`。
每个有新增材料的项目都提供 `outputs/ORGANIZED_20260910.md`，列出该服务器的
原目录和现目录。全量目录对应关系见 [source_directory_map.tsv](source_directory_map.tsv)。

本次核对 94,526 个源文件，新增归档 74,185 个文件、30,909,265,470 字节，
另复用 197 上已存在并重新比对的 20,684 个文件、10,449,241,923 字节。
共用配置、日志、环境记录和比较表可能在多个相关项目中各保留一份，
这些副本不代表增加了实验次数。

混合实验按文件中的方法名称分别保存；正式运行、试运行和中断记录保留
原来的目录名称及原始内容。整理过程不重新计算或修改论文数值。
源代码工作目录、原输入和原依赖保持原位，具体范围见各服务器的
`spec_*.json` 中 `selected` 和 `retained` 两项。

197 上的旧 NS 主实验位于
`FM4PDE/outputs/main/revision_20260909/ns_main_revision_0909/local/complete_local_ns_study/main_results/`。
本次通过逐文件核对复用该归档。最初路径检查的失败记录保留在
`verification_runs/collected_server197/` 的证据包中，修正路径后完整核验通过。

各新归档目录的 `.organization_receipt.json` 记录每个文件的原路径、
现路径、大小和 SHA-256。整体结果见 [verification_summary.json](verification_summary.json)；
该文件还登记三个证据包及独立核验记录的位置和校验值。
源文件在整理前后均经过检查，目录结构、内容和文件权限保持一致。

整理程序为 [organize_results.py](organize_results.py)。运行前先生成计划，
执行时拒绝发生变化的源文件、未经确认的符号链接和不同内容的既有结果。
程序只创建新结果目录或核对已有副本，不删除原目录。
