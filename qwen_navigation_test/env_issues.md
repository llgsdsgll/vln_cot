# 环境问题排查记录

运行 Qwen3.5 VLN-CE 导航测试时遇到的所有环境问题及解决方案。

---

## 1. `ModuleNotFoundError: No module named 'torch'`

**原因**：系统 Python 未安装 PyTorch。

**解决**：根据 Python 3.9 + CUDA 12.4 安装对应版本：
```bash
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

---

## 2. `ImportError: libOpenGL.so.0: cannot open shared object file`

**原因**：系统缺少 `libOpenGL` 共享库，habitat_sim 渲染依赖该库。

**解决**：
```bash
apt-get install -y libopengl0
```

---

## 3. `ImportError: libEGL.so.1: cannot open shared object file`

**原因**：系统缺少 `libEGL` 共享库，habitat_sim 使用 EGL 进行无头渲染。

**解决**：
```bash
apt-get install -y libegl1
```

---

## 4. `ModuleNotFoundError: No module named 'vlnce_baselines'`

**原因**：直接运行脚本时，Python 找不到项目根目录下的模块。

**解决**：运行前设置 `PYTHONPATH`：
```bash
export PYTHONPATH=.
# 或
python3 -c "import sys; sys.path.insert(0, '.')"
```

---

## 5. 多个缺失依赖包

运行时依次发现以下依赖缺失，逐一安装：

| 缺失包 | 安装命令 |
|--------|----------|
| `lmdb` | `pip install lmdb` |
| `msgpack_numpy` | `pip install msgpack-numpy` |
| `tensorboard` | `pip install tensorboard` |
| `webdataset`（需旧版） | `pip install "webdataset==0.1.103"` |
| `jsonlines` | `pip install jsonlines` |
| `dtw` | `pip install dtw-python` |
| `fastdtw` | `pip install fastdtw` |
| `tensorflow` | `pip install tensorflow-cpu` |
| `openai` | `pip install openai` |

**注意**：`webdataset` 需安装 `0.1.x` 版本，`0.2.x` 版本删除了 `wds.Dataset` 类，与 habitat-lab 代码不兼容。

---

## 6. `numpy` 版本冲突

**原因**：`dtw-python` 安装时自动将 numpy 升级到 2.0，导致 `quaternion` 包报错：
```
ImportError: numpy.core.multiarray failed to import
```

**解决**：强制回退到 numpy 1.x：
```bash
pip install "numpy<2.0"
```

---

## 7. `NameError: mass is not found on habitat_sim`

**原因**：`habitat-lab` 配置中包含 `MASS` 等物理属性字段，但 `habitat_sim 0.3.3` 的 `AgentConfiguration` 不支持这些字段，`overwrite_config` 函数抛出异常。

**解决**：在 `habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py` 的 `AgentConfiguration` 的 `ignore_keys` 中加入物理属性：
```python
ignore_keys={
    "is_set_start_state",
    "sensors",
    "start_position",
    "start_rotation",
    # 新增：physics keys not supported in habitat_sim 0.3.x
    "mass",
    "linear_acceleration",
    "angular_acceleration",
    "linear_friction",
    "angular_friction",
    "coefficient_of_restitution",
},
```

---

## 8. `TypeError: SensorSpec.position` 类型不兼容

**原因**：`overwrite_config` 将 `POSITION: [0, 1.25, 0]`（Python list）直接赋值给 `SensorSpec.position`，但 habitat_sim 0.3.x 要求该字段为 `magnum.Vector3` 类型。

**解决**：将 `position` 和 `orientation` 加入 `ignore_keys`，然后手动用 `mn.Vector3` 赋值：
```python
import magnum as mn

ignore_keys={..., "position", "orientation"}

if hasattr(sensor.config, "POSITION"):
    sim_sensor_cfg.position = mn.Vector3(sensor.config.POSITION)
if hasattr(sensor.config, "ORIENTATION"):
    sim_sensor_cfg.orientation = mn.Vector3(sensor.config.ORIENTATION)
```

---

## 9. `AttributeError: SensorSpec has no attribute 'parameters'`

**原因**：habitat_sim 0.3.x 的 `SensorSpec` 已移除 `parameters` 字典，旧代码通过 `sim_sensor_cfg.parameters["hfov"]` 设置视野角。

**解决**：改用 `habitat_sim.CameraSensorSpec`，直接设置 `hfov` 和 `resolution` 属性：
```python
sim_sensor_cfg = habitat_sim.CameraSensorSpec()
sim_sensor_cfg.resolution = [h, w]
sim_sensor_cfg.hfov = mn.Deg(sensor.config.HFOV)
```

---

## 10. `unable to find CUDA device 0 among 1 EGL devices`

**原因**：EGL 无头渲染时无法将 CUDA device ID 映射到 EGL device，常见于多 GPU 或无显示器服务器环境。

**解决**：运行时设置 `EGL_DEVICE_ID` 和 `CUDA_VISIBLE_DEVICES` 环境变量：
```bash
export EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
```

已写入 `run.sh`，无需手动设置。

---

## 修改文件汇总

| 文件 | 修改内容 |
|------|----------|
| `habitat_extensions/config/vlnce_task.yaml` | 更新数据集路径和场景路径为绝对路径 |
| `habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py` | 兼容 habitat_sim 0.3.x：新增 ignore_keys、改用 CameraSensorSpec、用 mn.Vector3 设置 position |
