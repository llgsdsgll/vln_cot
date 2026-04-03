# Code Layout

当前目录已经按用途做了分类整理。

## 目录分类

### `qwen_annotation/`

当前正在使用的“大模型生成标注”相关代码都放在这里：

- `qwen_annotation/annotate_vln_episode.py`
- `qwen_annotation/qwen_api.py`
- `qwen_annotation/run_annotate_vln_episode.sh`
- `qwen_annotation/README.md`

这一组文件负责：

- 调用 Qwen 多模态模型
- 对 VLN instruction 做子任务划分
- 对每帧做 object / region 标注
- 保存逐帧结果、缓存和调试输出

详细使用说明见：

- [qwen_annotation/README.md](/home/gs/my_project/data_generation_pipline/qwen_annotation/README.md)

### `tools/`

和当前 Qwen 标注主流程分开的辅助脚本放在这里：

- `tools/generate_vln_video.py`
- `tools/compare_scene_overlap.py`
- `tools/check_api.py`

这些脚本主要用于：

- 生成 VLN 视频或帧
- 检查场景数据
- 做 Habitat API 或数据集相关辅助分析

### `frames/`

输入视频和逐帧图像：

- `frames/1.mp4`
- `frames/ep00001_frameXXXX.jpg`

### `output/`

所有运行输出、缓存和标注结果。

### `habitat-lab/` 和 `VLN-CE/`

外部代码库和依赖工程，保留原样，不和当前自写脚本混在一起。

## 兼容入口

为了不影响你原来的使用方式，根目录保留了兼容启动脚本：

- `./run_annotate_vln_episode.sh`

它会直接转发到：

- `./qwen_annotation/run_annotate_vln_episode.sh`

所以你现在仍然可以在项目根目录直接运行：

```bash
./run_annotate_vln_episode.sh
```

## 推荐使用方式

如果你关心的是当前这套 Qwen 标注流程，主要只需要看这几个位置：

- `qwen_annotation/`
- `frames/`
- `output/`

如果你想继续整理，我下一步还可以帮你把 `tools/` 下面再细分成：

- `tools/video/`
- `tools/dataset/`
- `tools/debug/`
