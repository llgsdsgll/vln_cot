# NavGen

### Introduction

The NavGen pipeline consists of four steps: generating the LH-VLN task config, recording trajectories, decomposing trajectories, and generating step-by-step tasks. These correspond to `gen_task`, `gen_traj`, `split_traj`, and `gen_step_task` in `main.py`, respectively.

We use a chat model in the NavGen pipeline. The repository is now configured to use DashScope's OpenAI-compatible endpoint with `qwen3.6-plus` by default, so please prepare your `DASHSCOPE_API_KEY` in advance. You can also override the model and endpoint with `--llm_model`, `--vlm_model`, and `--llm_base_url`.

Although the entire NavGen pipeline is run at once in `main.py`, due to considerations regarding network connectivity and the stability of large model outputs, we recommend running the code step by step in sequence.

### Start

Before you start generating, I would like to introduce some important parameters.

- API_KEY: Your API key. If omitted, the code will read `DASHSCOPE_API_KEY` from the environment. For the complete generation process of a task, it costs about 2000 tokens.
- loop: The number of cycles of the first step(generating LH-VLN task). This depends on how many tasks you want to generate.
- sample_region: Whether to sample rooms based on the Euler distance between rooms. Since the Euler distance cannot well represent the actual distance between rooms, it is set to **False** by default.
- sample_obj: Whether to randomly sample objects in the room. Since this can reduce the influence of large model preference selection, it is set to **True** by default.
- max_step: The maximum number of action steps in the second step(record the trajectory). Used to limit the trajectory that is too long. The default setting is **500**.
- ram_logs: The log save path of step 3 and step 4. This can help you confirm the actual progress of the NavGen.

These parameters can be viewed in `main.py`.

Now you can generate tasks with the following command:

```bash
cd nav_gen
python main.py --loop 100
```

Note: Due to factors such as navigation path obstruction, not all tasks generated in the first step can successfully record trajectories in the second step. For these tasks, after the second step trajectory generation is completed, the task directory will not contain a 'success' folder, but a 'tmp' folder instead. You can either delete them directly or resample the navigation starting points. The default navigation starting point is [randomly sampled from the navigable area](https://github.com/HCPLab-SYSU/LH-VLN/blob/4bf526a0518308d2b3030c9fb411e377267f5a2a/nav_gen/habitat_base/simulation.py#L43).
