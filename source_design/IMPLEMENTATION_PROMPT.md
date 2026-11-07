# 可直接交给代码助手的实验实现 Prompt

你是医疗 MLLM 研究工程师。请在当前仓库实现 CLEAR-Med-v1，并优先阅读 METHOD.md、EXPERIMENTS.md、prompts.json、config.example.json、clear_med_core.py、train_gate.py 和 calibrate_policy.py。**这是待验证研究方案，不能声称已经复现论文或获得性能提升。**

## 目标

冻结 Qwen/Qwen2.5-VL-7B-Instruct 的全部生成和视觉权重。对同一初始医学 VQA 输出执行：中性证据规划 -> 答案盲化视觉观察 -> 冻结证据绑定 -> 依赖后继隔离 -> 新会话的 selective/reset 双候选 -> 匿名证据资格审查 -> 四分类结果门控。只训练小型 MLP gate，不训练 generator，不默认引入 RL、外部大模型、隐藏状态优化或专用定位器。

## 必须实现的模块

1. `data/manifest.py`：从用户指定的本地数据目录读取。将推理输入与标签存储分开；推理 DTO 不得包含 answer/reference/gold/transition 等标签字段。真实答案只给离线 label/eval 模块。按患者或来源图像/文章划分，记录所有别名、语言译文和近重复。跨 split 相交立即报错。
2. `models/backend.py`：定义 `generate_json(role, payload, images, seed, token_budget)` 与 `embed_score(payload, images)`。提供默认 Qwen2.5-VL 适配器；核对所安装 transformers 与官方模型卡的 API，不猜接口。每个角色调用使用独立新 messages；不共享含旧回答的 KV cache 或会话对象。视觉预处理、图像像素预算一致并可追踪。
3. `evidence/planner.py`：加载 prompts.json 的 plan 模板。claim_ids 是私有映射，只把 public_tests 的白名单字段传观察器。最大两个问题。禁止 Planner 编造观察值或临床阈值。
4. `evidence/observer.py`：Observer 只接收 public_tests、实际图像和 provenance 清单。不能接收旧回答、旧 claim 文本、选项提示、候选、GT、模型之前的判断。质检不足返回 limited/unavailable。
5. `tools/crop.py`：仅执行裁剪/缩放等明确白名单操作，不执行任意模型 Python。校验 bbox、坐标尺度、输入文件；保留全图方向，登记图像 hash、裁剪变换与实际输出。一个病例最多一次额外观察；工具返回的原始像素不是诊断真值。不把像素换算成未知的毫米。
6. `evidence/binder.py`：对原 claims 进行 SUPPORTED/CONTRADICTED/UNRESOLVED 绑定。只有可用当前病例证据才能支持或反驳；无证据不等于 CONTRADICTED。程序验证每个引用真实存在。
7. `state/rebuild.py`：复用 clear_med_core.py。隔离 CONTRADICTED/UNRESOLVED 及其依赖后继；只保留独立已核验的 visual/context 事实。保留顺序不等于原时间前缀，允许保留错误之后的独立事实。旧最终答案、被隔离文字、错误诊断提示、raw trace 不进入 repair payload。
8. `repair/generate.py`：selective 与 reset 共用 repair 模板、相同新证据与解码。两者唯一区别是 retained_facts 是否为空；各一个候选。K 为空时去掉确定性重复分支。保存所有候选，不因结果不佳而删除。未知或无法解析记录失败，最多一次格式重试，计入成本。
9. `repair/audit.py`：匿名随机排列候选，检查新病例事实是否由冻结证据支持、必要正确事实是否保留。不暴露原候选身份/参考。程序核对 evidence ID。不支持的候选 eligible=false，不能靠 gate 高分覆盖资格失败。
10. `gate/features.py`：评分器可见 I/Q/E/y0/ya/scope/audit，但不见 GT。使用固定模板后附评分标记，读取最后标记位置的最终语言隐藏向量；保存 X、case_id、group_id 与特征版本。评分器独立于修复会话。不要把语言模型口头给的 0.9 当作校准概率。
11. `gate/labels.py`：离线计算 c0、ca，生成 WW=0、WC=1、CW=2、CC=3。同 case_id 的原始正确性必须一致。保留负收益/无收益和原本正确的样本。训练只使用 train，温度在 dev，lambda/delta 在独立 calibration，测试一律冻结。
12. `gate/train.py`：可复用 train_gate.py。CE+同病例候选排序；KEEP=0。门控值 P(WC)-lambda*P(CW)，lambda 校准网格 1/2/4，delta 网格 0/.05/.1/.2。2% Harm|C 只是示例研究运行点；不能宣称临床安全或有限样本保证。
13. `eval/run.py`：所有条件复用同一初始输出与冻结 evaluation manifest。实现自然输出、固定候选池、replay 前提对、预算匹配和必要消融。按 image/patient/article 成组配对 bootstrap。解析失败/拒答不得静默删除。
14. `logging/trace.py`：记录输入字段白名单、hash、prompt 版本、模型 revision、seed、所有调用/token/visual-token/forward/工具/时延、候选审查和 gate 结果。不得将测试标签写入推理日志可见字段。

## 先运行的最小实验

先用约 200 个自然病例比较 Raw+E、Selective+E、Reset+E、Random-drop+E。所有分支共用 E，机制对照也去掉旧最终答案；另在人工可审计子集加入 oracle E 上界。第一步不需要训练 gate。

机制信号成立后才构造约 5,000 条训练问题的候选集。样本量为预算目标，不是数据集事实。训练数据保留原正确/原错误自然比例或保存抽样权重。官方测试集绝不能拿来建记忆、调阈值或选择 prompt。

## 基线和必须输出的结果

one-pass、vanilla reflection、same-evidence raw refinement、same-evidence reset、always-selective、same-pool evidence judge、signed-gain gate、完整四类 gate、同计算预算 best-of-N。原论文的 adapted 版本明确标 adapted，不能把简化提示词版本称为复现。

输出 Acc0、Accfinal、ΔAcc、W→C/initial-W、C→W/initial-C、总体破坏率、修改覆盖率、harm/edit、证据可用覆盖率与 paired cluster CI。校验 ΔAcc=(N_WC-N_CW)/N。报告 input/output/visual tokens、所有 forward、p50/p95 latency、峰值显存与硬件。候选池 oracle 只能标上界。

## 必须先通过的测试

- Observer payload 没有旧答案、旧 claim、参考答案和 private claim_ids。
- Repair payload 不含被隔离的哨兵字符串、raw trace、旧答案或旧缓存。
- 与错节点无依赖、且有证据支持的后续事实仍被保留。
- 无证据/来源错配变 UNRESOLVED，不能自动变 CONTRADICTED。
- 不合格候选无论概率多高都不可提交。
- 所有净值不大于阈值时保持原输出；KEEP 不标成“正确”。
- 同图、同患者、同文章及双语近重复不跨分割。
- 初始正确样本可以产生 CW 标签，不能只训练成功修复。
- 训练/预测概率顺序严格为 WW/WC/CW/CC。
- 数据、模型或 GPU 不可用时明确报告，不生成虚构结果表。

## 交付顺序

先交可运行 mock 单元测试和字段权限测试；再交真实模型一个病例的完整可审计轨迹；再交 200 例机制结果；最后进行 gate 训练与完整测试。请在 README 精确区分已经执行的测试、尚未接入的组件和尚未完成的实验。不要臆测论文中没有明确给出的实现。
