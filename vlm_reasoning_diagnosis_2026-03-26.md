# VLM 结构化输出问题诊断

日期：2026-03-26

## 背景

当前任务：

- 使用教师 VLM 为 VLN 数据生成带推理过程的监督信号。
- 当前实现位于 [`vln_data_synthesizer.py`](/home/gs/my_project/human_ann_cot/vln_data_synthesizer.py)，流程是先让模型输出结构化 JSON。
- 然后在本地对 JSON 做校验，再渲染成最终的 Markdown 训练模板。

本轮诊断目标：

- 理解为什么有些帧会出现 `empty response`。
- 判断问题到底来自 stop signal、`response_format=json_object`、prompt 设计、服务端 thinking 模式，还是输出被截断。

## 主要现象

典型 warning 如下：

```text
[WARNING] VLNDataSynthesizer - VLM 输出格式不合格 (attempt 1/3): empty response | raw前200字符:
[WARNING] VLNDataSynthesizer - response_format=json_object 在当前多模态请求上返回空内容，后续改用纯提示词 JSON 模式。
```

这几条日志的含义是：

- HTTP 请求本身是成功的。
- 但返回里的 `message.content` 是空的。
- 因此本地解析器把这次结果判成了 `empty response`。

## 已验证的事实

### 1. 所谓“空回复”并不是真的没有生成内容

在一次实际抓到的失败样本中，场景是 `episode 1 / frame 8`，原始返回表现为：

- `message.content = null`
- `message.reasoning` 中有很长一段内部思考内容
- `finish_reason = "length"`

这说明：

- 模型实际上生成了 token。
- 这些 token 被消耗在了 reasoning 通道，而不是最终答案通道。
- 在真正把 JSON 写进 `content` 之前，就已经撞上了 token 上限。

### 2. 问题不主要出在 `response_format=json_object`

我对 `frame 7` 和 `frame 8` 做过诊断矩阵，对比了两种情况：

- `response_format=json_object = True`
- `response_format=json_object = False`

两种情况下的结果都高度一致：

- `finish_reason = "length"`
- `content_empty = true`
- `reasoning_len` 很大
- 本地解析错误都是 `empty response`

结论：

- 单独打开或关闭 `response_format=json_object` 并不能解决问题。
- 真正的问题发生得更早：模型根本没有成功进入最终答案输出阶段。

### 3. 带校验反馈的重试并不能稳定打破循环

我对比过以下两种请求：

- 普通单轮请求
- 带“上次 JSON 校验失败，请修正”的重试型请求

观察结果：

- 普通请求：稳定出现 `content_empty = true`
- 带反馈重试：仍然会出现 `content_empty = true`
- 某些情况下，`reasoning` 反而更长

结论：

- 单纯把校验错误回灌给模型，并不能稳定把它从内部思考循环里拉出来。
- 有时还会给模型更多文字材料，让它继续在 prompt 上打转。

### 4. 服务端 thinking 模式是目前最强的主因线索

我单独测试了两种模式：

- 默认模式
- `chat_template_kwargs.enable_thinking = False`

观察结果如下。

默认模式：

- `content_empty = true`
- `reasoning_len = 3570`
- `finish_reason = "length"`

关闭 thinking 后：

- `content_empty = false`
- `reasoning_len = 0`
- `finish_reason = "length"`
- 输出开始真正写进 `content`

结论：

- 这个服务端非常可能默认启用了 reasoning / thinking 模式。
- 在这种模式下，token 预算优先消耗在内部思考，而不是最终 JSON 内容上。
- 这是目前最强、最明确的根因证据。

### 5. 关闭 thinking 后，第二层问题才暴露出来

当 `enable_thinking = False` 时，模型确实开始往 `content` 里写 JSON，但结果仍然可能失败，因为：

- `finish_reason = "length"`
- JSON 被截断
- 本地报错会变成类似：
  - `Invalid JSON: EOF while parsing a string ...`

这说明：

- 关闭 thinking 是必要条件，因为它能把输出拉回正确通道。
- 但这还不够，模型仍然可能输出过长，导致 JSON 还没写完就被截断。

## 当前失败模式

目前观察到的失败模式基本可以拆成两阶段：

1. 默认行为下：
   - 模型把 token 消耗在 `message.reasoning`
   - `message.content` 保持为空
   - 本地解析器报 `empty response`

2. 关闭 thinking 后：
   - 模型开始把 JSON 写进 `message.content`
   - 但 JSON 仍然可能过长
   - 生成过程撞到 `max_tokens`
   - 本地解析器报截断 / 非法 JSON

## 已基本排除的解释

以下解释目前没有得到证据支持：

- “第 8 帧比前几帧大很多，所以才失败”
  - `frame 7` 和 `frame 8` 的 prompt 长度非常接近
  - 图像 payload 大小也接近

- “只有 `response_format=json_object` 才会引发问题”
  - 去掉它之后问题仍然存在

- “主因是缺少 stop token”
  - 当前更强的证据指向 reasoning 模式消耗了 token 预算
  - 模型不是简单地“最终答案写太多停不下来”，而是很多时候根本还没进入最终答案输出

## 当前解释

按目前证据排序，最可能的根因是：

1. 这个模型 / 后端路径默认启用了 thinking / reasoning 模式。
2. 当前任务 prompt 约束较多，模型容易把大量 token 花在“理解规则、复述规则、重新阅读 prompt”上。
3. 即使关闭了 thinking，模型仍然倾向于输出过长 JSON，最终被 `max_tokens` 截断。

## 相关代码位置

结构化解析逻辑：

- [`vln_data_synthesizer.py#L404`](/home/gs/my_project/human_ann_cot/vln_data_synthesizer.py#L404)

当前请求构造和重试逻辑：

- [`vln_data_synthesizer.py#L675`](/home/gs/my_project/human_ann_cot/vln_data_synthesizer.py#L675)

当前初始化里默认先尝试 JSON response mode 的位置：

- [`vln_data_synthesizer.py#L251`](/home/gs/my_project/human_ann_cot/vln_data_synthesizer.py#L251)

当前结构化 JSON prompt 模板：

- [`vln_data_synthesizer.py#L46`](/home/gs/my_project/human_ann_cot/vln_data_synthesizer.py#L46)

## 下一步诊断计划

下一轮测试应优先从参数层面做最小改动，再决定要不要大改 prompt：

1. 在正式生成流程中强制 `enable_thinking = False`
2. 在不改 schema 的前提下，进一步降低 prompt 冗余度
3. 调整 `max_tokens`，观察合法 JSON 完成率是否提升
4. 在一个固定的小切片上比较成功率，例如：
   - `episode 1`
   - `frames 1-10`

## 当前阶段性结论

截至目前，最合理的解释是：

- 模型的主要失败原因不是“没识别到 stop signal”。
- 更主要的原因是：后端把大量 token 消耗在内部 reasoning，而不是最终答案内容。
- 关闭 thinking 后，剩余的主要问题变成了输出过长，导致 JSON 被截断。
