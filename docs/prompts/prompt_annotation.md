# Role
你是一个资深的具身智能（Embodied AI）Python 开发专家，精通视觉语言导航（VLN）的数据处理和多模态大模型 API 调用。

# Task
请帮我编写一个 Python 脚本 `vln_data_synthesizer.py`，用于合成大模型（Teacher Model, 如 Qwen-VL）的 Chain-of-Thought (CoT) 标注数据，以便后续微调 9B 的小模型。

# Core Requirements
请实现一个名为 `VLNDataSynthesizer` 的类，包含以下核心流程：

1. `normalize_bbox(bbox, img_width, img_height)`: 
   - 输入原始像素格式 `[xmin, ymin, xmax, ymax]`。
   - 输出归一化到 0-1000 区间的整数列表。

2. `build_prompt(step_data, script_memory)`: 
   - 基于我后续提供的英文 Prompt 模板，将当前步的真值数据（如指令、已完成的子任务、目标物体名称、归一化后的 BBox、下一步动作等）格式化填充为字符串。
   - `script_memory` 是一个存放历史真实动作文本的 list。将其 join 成一段连贯的字符串填入 prompt。

3. `update_script_memory(script_memory, step_data)`:
   - 为了防止模型幻觉累积，我们使用脚本强行维护真实历史。
   - 在每一步结束后，提取真值数据中的执行动作，生成如 "Step {t}: Executed {action}." 的文本，追加到 `script_memory` 中。

4. `process_trajectory(trajectory_data)`:
   - 遍历一条轨迹的所有 step。
   - 每一跳读取图像（使用 PIL 或者 base64，根据主流 VLM API 规范处理）。
   - 调用大模型 API（请写一个 mock 的 `call_vlm_api(prompt, image)` 函数占位，保留完整的 retry 和 error handling 逻辑）。
   - 将合成结果解析并保存为 OpenAI / HuggingFace 标准的对话微调格式（JSONL），包含 `user` (指令+图像) 和 `assistant` (生成的 CoT + Action)。

# Coding Style
- 使用面向对象编程，添加详细的类型提示 (Type Hints) 和 Docstring。
- 使用 `logging` 模块打印处理进度，不要用 `print`。
- 将 Prompt 模板抽离为单独的全局变量或独立文件，方便我随时修改。