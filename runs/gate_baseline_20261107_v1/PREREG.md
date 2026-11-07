# G1：冻结候选池上的可训练 gate baseline

用户最新指令优先：先跑通 baseline 并按现有计划训练；新颖性不再是本次启动门槛。
本轮选择原始方案的冻结 Qwen2.5-VL-7B + 小型四分类 MLP，不训练生成器、不引入 RL。
E0/D1/D2 不重跑、不修改；本轮不接入尚无独立临床验证的新支持规则。

## 数据与权限

只读复用已结束 E0 v2 的全部 200 个 PMC-VQA 训练来源病例、原始输出、同 E 的
selective/reset 两候选。保留原正确、原错误、无改善、有害、格式失败。每例两行、同权。
按原 article group 排序后 seed2026110707 打乱，120/20/20/40 组分别 train/dev/calibration/
diagnostic_eval；同例/同文章候选不得跨分割。若非200独立组或文件缺失就报错，不能补样。
不读取官方测试集。E0 的全体标签曾分析过，40例只能称回顾性开发诊断，不能称未触碰测试。

提特征进程只读图片与白名单评分 DTO，不读取监督文件。DTO 含 Q/choices、冻结 E、
y0/ya、候选所用的事实及格式/解释缺失标记；不含 reference/correctness/outcome/class/
split/group/分支身份。病例/候选 ID 仅用于结果回写，证据局部ID统一重映射。
真实标签只由离线准备/MLP训练/评估进程读取；WW/WC/CW/CC 固定为0/1/2/3。

## 特征与训练

只读共享 Qwen2.5-VL-7B-Instruct，沿用已有 revision
cc594898137f460bfe9f0759e9844b3ce807cfb5，PRO5000 已有环境。
processor min_pixels=256*28*28、max_pixels=768*28*28，BF16、SDPA、eval/no_grad，
全参数冻结。固定评分文本后附 `SCORE_STATE`，取最终语言位置最后一层 hidden 向量。
同内容可复用向量但保留每个候选的标签/分母；不生成新候选或参考。不截断文本以避免
静默丢字段；总序列>8192则记录昂贵失败并停止，不自动调prompt重跑。

仅在 train 上拟合 LayerNorm -> Linear(d,512) -> GELU -> Dropout(.1) -> Linear(512,4)。
AdamW lr=1e-3、weight_decay=.01，病例batch32，最多100epoch，dev CE早停patience15。
训练种子2026110711/12/13。两项固定配置：CE baseline；CE + .1 同病例排序。
排序比较同例两候选及零值KEEP，target gain=ca-c0，U_train=P(WC)-P(CW)，hinge margin=.1，
仅比较真值收益不同的对。没有可比较对时rank=0，不制造样本。
标准化（均值/方差）仅拟合train；所有特征仍保留，零方差尺度置1。
模型按dev CE选checkpoint；温度在dev固定网格 .5/.75/1/1.5/2/3 中最小NLL。

独立calibration选lambda=1/2/4、delta=0/.05/.1/.2：最大净正确数，约束经验Harm|C<=.02；
平局优先少编辑、更高lambda、更高delta。KEEP始终是候选，不满足约束或U<=delta就KEEP。
2%只是小样本研究运行点，无临床安全/统计保证。候选仅做答案字母有效性检查；没有独立
证据资格审查模块，本轮不是完整方法，更不能声称候选临床合格。缺解释不编造。

## 评价与输出

diagnostic_eval 在checkpoint/温度/lambda/delta冻结后一次性读出：KEEP、always-selective、
always-reset、两个训练配置各seed的gate、候选池hindsight上界（只上界）。报告全部分母、
Acc0/final、WC/CW、修复率、Harm|C、编辑率、无效/解释缺失、类别计数、同例候选差异。
校验deltaAcc=(WC-CW)/N；配对图组bootstrap2000次，不因区间宽删结果。
训练损失下降仅代表优化运行，不代表机制成立或泛化提升。

## 资源、失败与交付

pro5000物理GPU6；启动前显存已检查，实际加载前再次检查>=24GiB可用，预留峰值余量。
所有主子进程用中性入口；不改他人进程；不访问my-gpu。只读复用已有16.6GB权重，
新增本轮数据/向量/MLP/checkpoint预计<1GB，守护6GB停写、独立硬预算3h（包含准备和重试），
总新增严格<50GB；没有自动昂贵重跑。先记录原始开始时间和deadline，父守护和外层timeout
独立于SSH存活，60秒低成本存储检查。异常保留，不能改名重置时钟。
代码、协议、汇总报告通过代理公开；图像、labels、逐例内容、向量和模型参数不公开。
复用automation-2唯一心跳；完成后汇总训练/评估并自主选择下一项主线实验。
