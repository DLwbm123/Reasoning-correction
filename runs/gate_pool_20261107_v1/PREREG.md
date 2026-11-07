# G2：新来源文章自然候选池与四类 gate 训练

动机：用户要求按原始计划继续训练，G1 已跑通，但只有120训练/20校准例、校准WC=0，
且过拟合。扩充自然来源文章是判断原始四类gate是否能学习的必要下一步；不改G1结果。
本轮不以方法新颖性作为启动门槛，也不声称旧Selective机制已获支持。

## 冻结数据与权限

PMC-VQA官方train.csv，沿用E0视觉关键词/排除规则；按article去除官方test/test_clean/test_2
全部来源文章、旧E0和D1文章；排除旧图片与本池内编码字节完全相同的图像（科学分组，
非传输校验），每篇一例。取样不看模型成败，仅检查来源答案字母有效。seed2026110708。
冻结1600新文章，独立分组1000train/200dev/200calibration/200diagnostic_eval；病例执行
顺序seed2026110709打乱，不按split顺序跑。官方测试答案不读；所有来源均为官方训练源，
保留200例为新开发诊断，不替代最终外部验证。未完成患者/图像近重复排查，按文章组披露。
来源标签在独立私有文件，只给离线监督。推理和特征进程不读取监督与split。

## 候选与训练

冻结Qwen2.5-VL-7B-Instruct，复用revision cc594898137f460bfe9f0759e9844b3ce807cfb5。
沿用已完成D1 answer-first prompts和逐test观察执行绑定，固定中性Observer问题；最多2项，
来源/test由实际执行赋予，原始观察/assessability保留，不编造视觉观察或临床真值。
每例initial/plan/逐项observe/bind/同E的Selective与Reset；不计算本轮不训练的Raw/Random/
Summary分支，不改变两候选repair模板。K空时复用同输入确定输出。所有原正确/错误、
无改进、有害及格式失败均保留。保留原始limited/unavailable状态，只有adequate可支持K。
没有独立候选证据审查，格式可用不等于临床合格。旧图状态算法不变，局限继续报告。

全1600例候选持久化完成后才离线制WW/WC/CW/CC监督，再由无标签DTO提评分hidden特征。
不对未完成池训练，截止中断时原始分母1600保留，不只报告成功子集。非CUDA单例异常保存
失败，按初始/候选缺失为错误；CUDA/OOM或结构异常昂贵失败停机，不自动重跑。
评分接口和MLP沿用G1：最终SCORE_STATE末token、3584维、train-only标准化，LN-512-GELU-
Dropout(.1)-4；AdamW1e-3/WD.01、batch32病例、100epoch、dev CE早停15。
CE、CE+.1同例排序，各seed2026110711/12/13；KEEP零值，rank margin .1。
dev温度网格.5/.75/1/1.5/2/3；cal lambda1/2/4、delta0/.05/.1/.2，最大净正确数且经验
Harm|C<=.02；平局少编辑、高lambda、高delta；无可行点则KEEP。2%非安全保证。
不依据未来eval或校准事件结果重抽分组/扩样/调整prompt。

## 评价

checkpoint/温度/策略冻结后一次性评价200例：KEEP、alwaysSelective、alwaysReset、
CE和CE+rank三种子、pool hindsight上界仅诊断。报告Acc、WC/CW、破坏/修复/编辑/接受率，
原始分母、四类事件数、候选同答、K空/图作用、观察覆盖、格式与解释缺失、完整成本，
按文章成组配对bootstrap2000次，校验净增恒等式。若全KEEP如实保留，不以低loss或零编辑
宣称收益。模型观察不是金标准，答案对不认证解释忠实。

## 资源、时钟与交付

pro5000 GPU6当前空闲不足，选jiangsuiyang物理GPU6（3090，当前24124MiB空闲）；不访问
my-gpu、不改他人进程。存储在规定NAS目录，新中性入口/tmp/m6q2；只读复用旧模型缓存。
按E0实际约20秒/例估计1600例约9小时，加特征与训练留余量；整体原始硬预算20小时，
自首次数据准备计（含设置和失败），collection在原始deadline前2小时停止，不完整不训练。
父守护与外层timeout独立于SSH；不重置时钟。预计新增图片<0.5GB、日志/特征/checkpoint
<1GB，独立守护6GB保守停写，实际新增必须<50GB。编码提特征前>=21.5GiB free，.90
进程显存上限（G1实测峰值allocated17.39GB/reserved20.44GB）；OOM保留，不自动换配置。
启动后核对主子进程及GPU显示中性，使用既有环境，不复制权重，不删除历史。
完成后通过代理公开源代码/冻结规则/汇总报告；病例、标签、向量、模型参数、完整日志和
患者内容不公开。复用automation-2唯一心跳，健康无变化静默，故障/完成/路线变化通知。
