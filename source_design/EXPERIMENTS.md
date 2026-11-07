# 实验计划与判据

## 0. 先冻结协议

固定模型 commit、processor、图像像素预算、解码、prompt hash、数据 manifest、病例组、失败处理和评分器。baseline 与方法复用完全相同的初始输出。不确定或无法解析的最终输出按预注册规则计分；在主 forced-answer 准确率中，拒答与解析失败计错误，不得静默删掉。

主方法为一轮、至多两个核验问题、至多一次额外观察、至多两个修订候选。默认 7 次生成调用（初始、计划、观察、绑定、两分支修复、候选审查），加两次非生成评分 forward；有放大重观察时至多 8 次生成。格式修复重试单独计入，最多一次。提前 KEEP 时使用实际调用数。第一版不承诺低时延。

## E0：200 个自然病例的机制试验，不先训练 gate

在同一原答案上固定自动获得的观察 E，比较：

1. Raw+E：保留原 claims、追加同一 E；机制对照也移除旧最终答案。
2. Selective+E：本方法的证据白名单状态。
3. Reset+E：只给 Q、I、E。
4. Random-drop+E：控制被删 token 数/陈述数，与 selective 尽量匹配，不能利用真实错误位置。

抽样须包含原本答对的病例，不能只挑错误样本。另在预先选出的可审计子集用专家/明确标注产生 oracle 观察，重复 1–3。自动/oracle 为两层证据条件，不把 oracle 当作部署方法。原则上给可观察事实而非最终诊断/答案字母；若问题问的就是该事实，标明该 oracle 直接揭示答案信息。

若 Selective+oracle-E 对 Raw+oracle-E 无收益，说明当前状态重建假设未得到支持；若只有 oracle 有收益，问题在获取可靠观察。

## E1：自然错误主结果

训练来源：PMC-VQA 官方训练来源的视觉问题，严格与所有评价来源按文章/图像/近重复去重。优先约 5,000 train，独立开发/校准各约 500，样本量为起始预算而非数据集事实。固定 source/article/image 分组，不能按 QA 行随机划分。

同域：官方 test_clean。外部：SLAKE、VQA-RAD 可用官方测试划分；视觉题与全量同时报告。基准清理与版本可能不同，必须发布实际清单与 hash。所有模型都在同一批样本上配对评价。

基本表：one-pass；vanilla reflection；same-evidence raw refinement；same-evidence reset；SELECTIVE always-accept；完整 CLEAR-Med；同预算 self-consistency/best-of-N。没有真正实现原论文全部算法时，方法名称必须带 adapted/inspired，不冒称原论文复现。

主指标定义（N 为全体病例）：

- Acc0、Accfinal、ΔAcc。
- Repair = N(W→C)/N(initial W)。
- Harm|C = N(C→W)/N(initial C)。这是答案破坏率，不是患者临床伤害率。
- Population harm = N(C→W)/N。
- Edit coverage = N(edited)/N；harm/edit = N(C→W)/N(edited)。
- 必须满足 ΔAcc=[N(W→C)-N(C→W)]/N。
- 外加错误接受率/正确保留率、证据可判断覆盖率、错误类型分层。

按 patient 优先、否则 image/article 做成组配对 bootstrap（建议 2,000 次），跨种子报告。不能把同图重复问题、同病例多次采样当成独立患者扩大显著性。稀有有害修订给计数和区间，不只给百分比。

## E2：固定候选池，隔离 gate 效果

冻结 y0、E、selective 候选、reset 候选以及 candidate audit。所有选择器从相同候选池选：

- KEEP。
- 永远采用 selective；永远采用 reset。
- 候选自身置信度最高。
- 普通匿名 evidence judge。
- 不看 E 的四类结果 gate。
- 看 E 的 signed-gain 估计头（强简单基线）。
- 四类结果 gate，只有 CE。
- 完整四类 gate，CE+同病例排序。
- hindsight oracle（仅上界，不计部署结果）。

比较 risk–edit coverage 曲线，在同修改率/同候选池报告，而不只给各方法默认阈值。oracle pool accuracy 区分“没有正确候选”和“没选对候选”。检查模型只认题目难度/分支标签、但不能在同病例区分候选的捷径。

## E3：false/correct premise 成对重放

从自然失败摘要提取最小视觉前提；提取器只看 Q 和摘要，不看图和参考。审核者看图，保留可明确反驳且与问题相关的前提；配对的正确前提由明确图像依据支持，并人工复核。不能简单批量翻转左右/否定充当主基准。

false、correct、neutral 仅改变先前 assistant turn 的单条前提；图像、问题、最终 user prompt、解码完全相同。评测模型不知道类别，也看不到另一条前提。不同模型共用审核后的前提对，保存来源模型用于分层。

报告 A_false、A_correct、两者等权平均（注明不是疾病分类 balanced accuracy），初始正确保留、错误前提语义复现率及 paired image-cluster CI。只测 false 会偏爱无差别丢弃历史的方法。

## E4：消融与鲁棒性

1. 观察者可见原答案 vs 盲化：隔离答案诱导。
2. raw claims vs evidence-only canonical facts：隔离真正的状态替换。
3. no graph / flat filter vs 依赖后继闭包：判断图是否值得保留。
4. 只 selective vs 只 reset vs 双候选：同预算/同候选数控制。
5. 去 gate、去 KEEP、去 same-case rank：隔离选择机制。
6. 无裁剪 vs 受限裁剪：同时报告图像 token 成本。
7. 自动证据 vs oracle 证据：只作诊断。
8. 元数据错误来源与语义错误观察分别测试；前者可能被 hash 规则轻易识别，不能充当视觉推理鲁棒性的证明。
9. 证据缺失应允许 UNRESOLVED；不能把“去图后答案不变”一概算错误，文本可能已经携带充分观察。
10. 图像置换时原参考可能已失效；没有新图金标准时只报告依赖变化，不报告错误的准确率。反事实正确性实验使用具有分别审核真值的图像/事实对。

## E5：模型、评估器与计算控制

默认 G/V 为同一冻结 Qwen2.5-VL-7B-Instruct，不靠更大模型获得不公平增益。可增加 HuatuoGPT-Vision-7B 作为医疗骨干验证；其视觉接口需按官方实现接入。

跨生成器测试可以固定评分骨干与 gate，只替换原答/修订生成器。此时是“跨生成器迁移”，不是把不同隐藏维度的 gate 直接移植到另一架构。若观察器也替换，必须单独报告此变化。

报告 input/text tokens、visual tokens、generated tokens、全部 forward、工具次数、p50/p95 wall-clock、峰值显存。API 与本地服务需明确硬件/并发/缓存。调用数相同不等于算力相同。主张效率需成本匹配或 Pareto 曲线；较强 judge/grounder 同样提供给适用基线。

## E6：解释/观察质量审计

最终答对不代表依据正确。预先分层抽取约 200 个病例，覆盖保持、修复成功、破坏及原本答对的输出。两位有相应影像背景的审核者盲化查看图片、问题、证据与回答，不看方法名/自动分数。评估视觉事实、来源对应、依赖使用、正确事实丢失；不把语言合理性等同真实推理因果。

模型 judge 可做预筛但不能宣称等价专家。标注分歧给一致性、仲裁和 uncertain 比例。若按罕见类型过采样，整体质量估计须按抽样概率加权。

## 预先规定的成功条件

完整方法应同时：在自然测试上获得可信净改进；在相同证据下优于 raw refinement；在相同候选池/修改率下改善损益权衡。若只优于 one-pass，不足以支持核心机制。若只在人工错误/特定模型上有效，结论必须相应缩小。
