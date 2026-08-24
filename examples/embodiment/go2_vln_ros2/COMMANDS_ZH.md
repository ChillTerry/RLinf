# Go2 + Uni-NaVid 真机 VLN 命令速查

本文收录日常联调最常用的命令。完整设计和安全说明见同目录 `README.md`。

> 执行 `forward/left/right` 或模型控制前，必须有人在 Go2 旁持有物理急停，机器人位于平整、空旷区域。物理急停不能被网络看门狗替代。

## 0. 基本信息与环境初始化

下文使用：

```text
推理机 RLinf：/home/qyx/realworldvln/RLinf
Go2 工作区： /home/unitree/qianyx/go2_vln_ws
Go2 SSH 别名：go2
Go2 局域网 IP：192.168.123.18
TCP/UDP 端口：8765/8766
```

首次生成通信密钥：

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

密钥必须在两端完全一致，不要写入 Git 或 YAML。

推理机每个新终端：

```bash
cd /home/qyx/realworldvln/RLinf
export GO2_VLN_TOKEN=zLtQ5MN6jiv2j1UOFABSFrwKxoh63xbWEUjwwXxp5hk
source scripts/socket_setup.sh
```

Go2 每个新终端：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/qianyx/unitree_vendor_ws/install/setup.bash
source /home/unitree/qianyx/go2_vln_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/qianyx/unitree_vendor_ws/cyclonedds.xml
export GO2_VLN_TOKEN='替换为相同密钥'
```

## 1. 推理机可用性检测

### Python、RLinf、PyTorch、CUDA

```bash
which python
python --version
python -c "import rlinf; print('RLinf:', rlinf.__file__)"
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "import ray, hydra, gymnasium, transformers, timm, sentencepiece; print('Python dependencies OK')"
nvidia-smi
```

如果 `nvidia-smi` 显示 `Driver/library version mismatch`，先重启机器，不要只降低用户态 NVML 库。

### 模型与 checkpoint

```bash
test -d /home/qyx/realworldvln/models/uninavid_weights/uninavid-7b-full-224-video-fps-1-grid-2 && echo 'Uni-NaVid base model OK'
test -f /home/qyx/realworldvln/models/uninavid_weights/eva_vit_g.pth && echo 'Vision tower OK'
test -f '/home/qyx/realworldvln/checkpoints/20260626-03:43:15-habitat_rxr_grpo_uninavid-87/habitat_rxr_grpo_uninavid/checkpoints/best_model/actor/model_state_dict/full_weights.pt' && echo 'RL checkpoint OK'
```

实际加载路径在：

```text
examples/embodiment/config/realworld_go2_vln_eval_uninavid.yaml
```

### Ray

设置密钥后重启 Ray，使 worker 继承环境变量：

```bash
ray stop --force
ray start --head
ray status
pgrep -af 'ray|eval_embodied_agent|interactive_model_test'
```

## 2. Go2 执行端可用性检测

先执行第 0 节的 Go2 环境初始化。

### ROS2 overlay 与包

```bash
which ros2
python3 --version
echo "$RMW_IMPLEMENTATION"
echo "$CYCLONEDDS_URI"

ros2 pkg prefix rmw_cyclonedds_cpp
ros2 pkg prefix unitree_api
ros2 pkg prefix unitree_go
ros2 pkg prefix go2_vln_interfaces
ros2 pkg prefix go2_vln_executor
ros2 pkg prefix go2_vln_socket_gateway
```

预期 `ros2` 来自 `/opt/ros/foxy/bin/ros2`，Go2 VLN 包来自 `/home/unitree/qianyx/go2_vln_ws/install/`。

### Unitree 状态与动作接口

```bash
ros2 topic type /sportmodestate
ros2 topic hz /sportmodestate
ros2 topic hz /utlidar/cloud_base
ros2 topic info /api/sport/request -v
```

`/sportmodestate` 必须持续更新，否则动作服务器会返回 `STALE_STATE`。

### RealSense

```bash
python3 -c 'import rclpy, pyrealsense2; print("ROS2 and RealSense Python OK")'
rs-enumerate-devices | head -n 30
```

网关启动后在另一个 Go2 终端检查：

```bash
ros2 node list | grep -E 'go2_vln|realsense'
ros2 topic type /camera/camera/color/image_raw
ros2 topic hz /camera/camera/color/image_raw
ros2 topic echo /camera/camera/color/camera_info --once
```

图像类型必须是 `sensor_msgs/msg/Image`。不要用 `/frontvideostream` 验证 RealSense，该固件话题不是本系统使用的标准 RGB 输入。

## 3. 同步与编译

### 本地同步到 Go2

先预览，再正式同步：

```bash
cd /home/qyx/realworldvln/RLinf
bash scripts/sync_go2_to_robot.sh --host go2 --dry-run
bash scripts/sync_go2_to_robot.sh --host go2
```

### Go2 重新编译

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/qianyx/unitree_vendor_ws/install/setup.bash

cd /home/unitree/qianyx/go2_vln_ws
colcon build --symlink-install --cmake-clean-cache --packages-select \
  go2_vln_interfaces \
  go2_vln_executor \
  go2_vln_socket_gateway
source install/setup.bash
```

确认：

```bash
ros2 pkg prefix go2_vln_executor
ros2 pkg prefix go2_vln_socket_gateway
```

### Go2 修改同步回本地

仅在确实修改过狗端源码时使用：

```bash
cd /home/qyx/realworldvln/RLinf
bash scripts/sync_go2_from_robot.sh --host go2 --dry-run
bash scripts/sync_go2_from_robot.sh --host go2
git status --short
git diff -- examples/embodiment/go2_vln_ros2/src
```

## 4. 常规启动

### Go2：执行器、RealSense、socket 网关

执行第 0 节 Go2 环境初始化后：

```bash
ros2 launch go2_vln_socket_gateway go2_vln_socket.launch.py
```

正常日志应包含：

```text
Go2 VLN executor listening on /go2_vln/execute_nav_primitive
RealSense connected successfully
RGB source is ROS topic '/camera/camera/color/image_raw'
Go2 socket gateway listening on TCP 0.0.0.0:8765 and UDP 0.0.0.0:8766
```

另一个 Go2 终端检查：

```bash
ros2 node list | grep go2_vln
ros2 action list -t | grep go2_vln
ss -lntup | grep -E ':8765|:8766'
```

如果已有外部 RealSense 驱动占用相机：

```bash
ros2 launch go2_vln_socket_gateway go2_vln_socket.launch.py \
  start_realsense_publisher:=false \
  image_topic:=/your/external/color/image_raw
```

停止所有狗端组件：在 launch 终端按 `Ctrl+C`。

### 推理机：环境与 Ray

```bash
cd /home/qyx/realworldvln/RLinf
export GO2_VLN_TOKEN='替换为相同密钥'
source scripts/socket_setup.sh
ray stop --force
ray start --head
ray status
```

## 5. 通信与图像测试（不运动）

```bash
ping -c 10 192.168.123.18
nc -vz 192.168.123.18 8765

python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.123.18 \
  --timeout 15
```

预期：

```text
udp_heartbeat_fresh=True
image_available=True
action_active=False
RGB OK: ... shape=(480, 640, 3)
```

保存并检查一帧：

```bash
python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.123.18 \
  --timeout 15 \
  --save-image /tmp/go2_rgb.ppm

xdg-open /tmp/go2_rgb.ppm
python -c "from PIL import Image; im=Image.open('/tmp/go2_rgb.ppm'); print(im.mode, im.size)"
```

## 6. 单个运动原语测试

测试前确认 `/sportmodestate` 正常、场地清空、物理急停可用。

STOP：

```bash
python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.123.18 --timeout 15 --action stop
```

左转、右转、前进：

```bash
python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.123.18 --timeout 15 --action left --confirm-motion

python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.123.18 --timeout 15 --action right --confirm-motion

python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.123.18 --timeout 15 --action forward --confirm-motion
```

默认目标是前进 0.25 m、旋转 30°；成功下界约为 `0.225 m` 和 `0.471 rad`。出现 `TIMEOUT` 时同时查看狗端 `Primitive progress` 日志。

## 7. 通信延迟测试

```bash
ping -c 100 -i 0.05 192.168.123.18

python examples/embodiment/go2_vln_ros2/socket_latency_test.py \
  --host 192.168.123.18 \
  --control-samples 100 \
  --image-samples 20
```

安全支撑后，可增加 STOP RTT：

```bash
python examples/embodiment/go2_vln_ros2/socket_latency_test.py \
  --host 192.168.123.18 \
  --control-samples 100 \
  --image-samples 20 \
  --stop-samples 20
```

初始参考：TCP 控制 p95 小于 50 ms，原始 RGB 请求 p95 小于 200 ms；以实际局域网基线为准。

## 8. 模型全流程运行

### 只推理，不让 Go2 运动

```bash
python examples/embodiment/go2_vln_ros2/interactive_model_test.py \
  --host 192.168.123.18 \
  --max-steps 30 \
  --max-episodes 100 \
  --video-fps 2
```

该模式获取实时 RGB，但不下发预测动作；动作日志显示 `execute=false`。

### 正常真机 VLN

先用较小步数：

```bash
python examples/embodiment/go2_vln_ros2/interactive_model_test.py \
  --host 192.168.123.18 \
  --max-steps 10 \
  --max-episodes 100 \
  --video-fps 2 \
  --execute-actions
```

程序首先要求输入 `MOVE`，然后在每个 `Instruction>` 后输入一条英文 VLN 指令。模型只加载一次；模型预测 STOP 或达到最大步数后返回下一个 `Instruction>`。输入 `q` 结束常驻会话。

运行中按 Enter 或 Ctrl+C 会请求受控 STOP，但需要等当前运动原语返回；需要立即停止时使用物理急停。短路径稳定后再增大 `--max-steps`。

## 9. 日志与输出

```text
真机 episode： logs/go2_model_motion/<episode时间戳>/
真机会话日志：logs/go2_model_motion/_session_<会话时间戳>/run.log
无运动 episode： logs/go2_model_dry_run/<episode时间戳>/
无运动会话日志：logs/go2_model_dry_run/_session_<会话时间戳>/run.log
```

查看最新真机会话：

```bash
ls -td logs/go2_model_motion/_session_* | head -n 1
tail -f "$(ls -td logs/go2_model_motion/_session_* | head -n 1)/run.log"
```

筛选关键事件：

```bash
rg -n 'UNINAVID RESPONSE|GO2 MODEL|GO2 RESULT|STOP|TIMEOUT|Traceback|ERROR' \
  "$(ls -td logs/go2_model_motion/_session_* | head -n 1)/run.log"
```

每个 episode 目录主要包含：

```text
episode.json              episode_end.json
actions.jsonl             action_results.jsonl
frames/                   <episode时间戳>.mp4
driver_result.json
```

## 10. 常见故障快速定位

### TCP/UDP 不通

推理机：

```bash
ping -c 3 192.168.123.18
nc -vz 192.168.123.18 8765
```

Go2：

```bash
ss -lntup | grep -E ':8765|:8766'
pgrep -af 'go2_vln|ros2 launch'
```

`udp_heartbeat_fresh=False` 时确认两端密钥一致、UDP 8766 可达且没有旧网关占用端口。

### 没有图像

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic type /camera/camera/color/image_raw
ros2 node list | grep go2_vln_realsense_publisher
```

### 动作 TIMEOUT

```bash
ros2 topic hz /sportmodestate
ros2 topic echo /sportmodestate --once
```

同时检查：

```text
Primitive progress: action=... distance=... yaw=...
```

当前执行器会按 `command_rate_hz` 持续发送 Move；若距离/角度仍不增长，检查 Go2 自带避障、Sport Mode、地面条件和里程计。

### Ray/OOM

```bash
free -h
nvidia-smi
ray status
ray logs raylet.out --tail 200
```

结束本地评估后可重启 Ray：

```bash
ray stop --force
ray start --head
```

## 11. 最常修改的配置

推理端：

```text
examples/embodiment/config/realworld_go2_vln_eval_uninavid.yaml
examples/embodiment/config/env/realworld_go2_vln.yaml
```

Go2 执行端：

```text
examples/embodiment/go2_vln_ros2/src/go2_vln_executor/config/go2_vln_executor.yaml
examples/embodiment/go2_vln_ros2/src/go2_vln_socket_gateway/config/go2_vln_socket_gateway.yaml
examples/embodiment/go2_vln_ros2/src/go2_vln_socket_gateway/config/go2_vln_realsense.yaml
```

修改狗端配置或代码后，必须重新同步、`colcon build` 并重启 launch。
