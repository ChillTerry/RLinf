# Go2 VLN socket deployment

日常使用的中文命令速查见 [`COMMANDS_ZH.md`](COMMANDS_ZH.md)，包括组件检测、同步编译、单动作测试和完整模型运行命令。

The inference and robot machines do not use ROS2 DDS across the LAN. ROS2 is
kept only inside the Go2 execution machine.

```text
RLinf -- TCP 8765 --> authenticated action requests/results and RGB frames
RLinf -- UDP 8766 --> authenticated 5 Hz watchdog heartbeat
                              |
                    go2_vln_socket_gateway
                         /           \
       RealSense ROS2 RGB              robot-local ROS2
               |                            |
 sensor_msgs/Image subscriber   ExecuteNavPrimitive + Sport + safety watchdogs
```

TCP is used for ordered, reliable messages. A 640x480 RGB8 observation is
921600 bytes and is requested only after reset or a completed primitive. UDP is
used only for the watchdog, so a failed inference process or broken LAN link
does not wait for a blocked TCP request to time out.

## Safety properties

- One authenticated TCP client is allowed at a time.
- Every action retains `episode_id` and `sequence_id`; replayed sequences are
  rejected by the local executor.
- Actions are never retried after an ambiguous TCP failure.
- Moving actions require a fresh authenticated UDP heartbeat.
- TCP disconnect or UDP heartbeat expiry cancels the active local action and
  causes the executor to publish `StopMove`.
- Sport state and Unitree requests stay on the robot machine. The optional
  executor-side point-cloud crop is disabled by default, relying on the Go2
  built-in collision-avoidance mode.
- The default camera is a RealSense color stream published locally as a
  standard `sensor_msgs/Image`. Every TCP image request waits for a frame
  published after the request, rather than returning a frame cached during the
  preceding motion. The SDK2 front camera remains an explicit fallback.

The pre-shared token protects against accidental or unauthenticated commands
on a trusted LAN. The stream is not encrypted. Use a dedicated robot LAN, VPN,
or TLS before operating on an untrusted network.

## File placement

Copy only these source packages to the Go2 execution machine:

```text
src/go2_vln_interfaces/
src/go2_vln_executor/
src/go2_vln_socket_gateway/
```

Keep these on the RLinf inference machine:

```text
rlinf/envs/realworld/go2/
examples/embodiment/config/env/realworld_go2_vln.yaml
examples/embodiment/config/realworld_go2_vln_eval_uninavid.yaml
examples/embodiment/go2_vln_ros2/socket_smoke_test.py
examples/embodiment/go2_vln_ros2/socket_latency_test.py
scripts/socket_setup.sh
model checkpoints and the .venv-go2 environment
```

The ROS2 `build/`, `install/`, and `log/` directories must not be copied. Build
them on the Go2 machine against its installed `unitree_ros2` environment. The
bundled publisher opens the camera directly through `pyrealsense2`; it does not
require the external `realsense2_camera` ROS package:

```bash
python3 -c 'import pyrealsense2 as rs; print("pyrealsense2:", rs.__file__)'
```

### Synchronize the robot-side source

Two scripts synchronize only the three packages listed above. They derive the
local source path from the RLinf checkout, so they can be run from any current
directory. They require passwordless SSH and `rsync` on both machines.

Preview and then upload local changes to the robot:

```bash
cd /home/qyx/realworldvln/RLinf
scripts/sync_go2_to_robot.sh --host unitree@192.168.3.15 --dry-run
scripts/sync_go2_to_robot.sh --host unitree@192.168.3.15
```

If `go2` is defined in `~/.ssh/config`, `--host go2` is also accepted. Run the
scripts directly or with `bash`; an accidental `sh script.sh` invocation is
automatically restarted under Bash.

Preview and then bring robot-side changes back into the local checkout:

```bash
scripts/sync_go2_from_robot.sh --host unitree@192.168.3.15 --dry-run
scripts/sync_go2_from_robot.sh --host unitree@192.168.3.15
git status --short
git diff -- examples/embodiment/go2_vln_ros2/src
```

The default robot workspace is `/home/unitree/qianyx/go2_vln_ws`. Override it
with `--remote-workspace /absolute/path` or `GO2_REMOTE_WORKSPACE`. The scripts
do not remove extra files by default. Add `--delete` only when an exact mirror
is intended; its deletion scope is restricted to the three package folders.
Each package prints an itemized relative-path list. In `--dry-run` mode this is
the list that would be synchronized; otherwise it is the list actually
synchronized. A package with no listed paths was already up to date. In the
itemized prefix, `>f` is a transferred file, `cd` is a created directory, and
`*deleting` is a deletion requested through `--delete`.

## 1. Build on the Go2 execution machine

Use the ROS distribution installed on the robot. The current onboard machine
uses Foxy with Python 3.8 and the isolated CycloneDDS overlay:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/qianyx/unitree_vendor_ws/install/setup.bash

# Use the system Python that supplies rclpy. Do not activate Conda.
which python3
python3 --version
python3 -c 'import rclpy, pyrealsense2; print("ROS2 and RealSense Python OK")'

cd /home/unitree/qianyx/go2_vln_ws
colcon build --symlink-install --cmake-clean-cache --packages-select \
  go2_vln_interfaces \
  go2_vln_executor \
  go2_vln_socket_gateway
source install/setup.bash
```

Generate a long random token:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Set the same token on both machines. Do not store it in YAML:

```bash
export GO2_VLN_TOKEN='replace-with-the-generated-token'
```

## 2. Start the robot-local gateway

Use the Unitree ROS2 environment that can already see the local topics. No
cross-machine CycloneDDS configuration is needed:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/qianyx/unitree_vendor_ws/install/setup.bash
source /home/unitree/qianyx/go2_vln_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/qianyx/unitree_vendor_ws/cyclonedds.xml
export GO2_VLN_TOKEN='replace-with-the-generated-token'

ros2 topic hz /sportmodestate
ros2 topic hz /utlidar/cloud_base

ros2 launch go2_vln_socket_gateway go2_vln_socket.launch.py
```

The launch starts the bundled `pyrealsense2` publisher, primitive executor, and
TCP/UDP gateway. Expected logs include:

```text
RealSense connected successfully; publishing color on '/camera/camera/color/image_raw'.
RGB source is ROS topic '/camera/camera/color/image_raw' (sensor_msgs_image).
```

In a second robot terminal, verify the published standard ROS2 messages:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/qianyx/unitree_vendor_ws/install/setup.bash
source /home/unitree/qianyx/go2_vln_ws/install/setup.bash

ros2 node list | grep go2_vln_realsense_publisher
ros2 topic type /camera/camera/color/image_raw
ros2 topic hz /camera/camera/color/image_raw
ros2 topic echo /camera/camera/color/camera_info --once
```

The image type must be `sensor_msgs/msg/Image`. The default camera parameters
are in `config/go2_vln_realsense.yaml`: 640x480, 30fps, BGR8 color enabled, and
depth disabled. To use an already-running external RealSense ROS2 driver, do
not let both processes open the same USB device; disable the bundled publisher
and pass the external topic:

```bash
ros2 launch go2_vln_socket_gateway go2_vln_socket.launch.py \
  start_realsense_publisher:=false \
  image_topic:=/your/external/color/image_raw
```

The SDK2 sidecar is not started in either RealSense mode.

Check the LAN listeners:

```bash
ss -lntup | grep -E ':8765|:8766'
```

Expected:

- TCP `0.0.0.0:8765`
- UDP `0.0.0.0:8766`

Gateway parameters are in
`src/go2_vln_socket_gateway/config/go2_vln_socket_gateway.yaml`. The default
is `image_source: ros_topic`, `image_message_type: sensor_msgs_image`, and
`image_topic: /camera/camera/color/image_raw`. Do not point this adapter at the
firmware's fragmented `/frontvideostream` payload.

To temporarily use the original Unitree camera, first verify
`unitree_sdk2py`, then select the fallback explicitly:

```bash
python3 -c 'from unitree_sdk2py.go2.video.video_client import VideoClient; print("VideoClient OK")'
export GO2_NETWORK_INTERFACE='replace-with-the-tested-interface'
ros2 launch go2_vln_socket_gateway go2_vln_socket.launch.py \
  image_source:=unitree_video_client \
  image_message_type:=go2_front_video \
  start_realsense_publisher:=false
```

## 3. Test from the inference machine without ROS2

```bash
cd /home/qyx/realworldvln/RLinf
export GO2_VLN_TOKEN='replace-with-the-generated-token'
source scripts/socket_setup.sh

nc -vz 192.168.3.15 8765

python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.3.15
```

The smoke test verifies authentication, TCP, UDP heartbeat, and one RGB frame.
It must report `udp_heartbeat_fresh=True` and does not move the robot.

Save a received frame for visual inspection:

```bash
python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.3.15 \
  --save-image /tmp/go2_rgb.ppm
```

With the robot supported and external estop active, test STOP:

```bash
python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.3.15 \
  --action stop
```

## 4. Run a one-step RLinf evaluation

Restart Ray after setting the token so workers inherit it:

```bash
cd /home/qyx/realworldvln/RLinf
export GO2_VLN_TOKEN='replace-with-the-generated-token'
source scripts/socket_setup.sh
ray stop --force
ray start --head
```

Keep the external estop active for the first end-to-end request:

```bash
bash examples/embodiment/run_realworld_eval.sh \
  realworld_go2_vln_eval_uninavid \
  env.eval.override_cfg.socket_host=192.168.3.15 \
  env.eval.max_steps_per_rollout_epoch=1 \
  env.eval.max_episode_steps=1 \
  env.eval.override_cfg.max_num_steps=1 \
  'env.eval.override_cfg.task_description=Walk forward and stop.'
```

Increase from 1 to 3 and then 10 steps only after the required failure tests
pass.

## 4A. Interactive model-pipeline test without robot motion

Use this mode before the one-step motion test. It receives live Go2 images and
runs Uni-NaVid, but every predicted action is intercepted inside the RLinf
environment and is not sent to the robot gateway.

Activate the RLinf environment and restart Ray after exporting the shared
token so Ray workers inherit it:

```bash
cd /home/qyx/realworldvln/RLinf
source .venv-go2/bin/activate
export GO2_VLN_TOKEN='replace-with-the-generated-token'
source scripts/socket_setup.sh
ray stop --force
ray start --head
```

Start the interactive driver:

```bash
python examples/embodiment/go2_vln_ros2/interactive_model_test.py \
  --host 192.168.3.12 \
  --max-steps 30 \
  --video-fps 2
```

Enter one instruction at each `Instruction>` prompt. During an episode, press
Enter or Ctrl+C to stop it manually. A model-predicted `STOP` also ends the
episode automatically. Enter `q` at the next instruction prompt to exit.

Each prediction is printed in this form:

```text
[GO2 MODEL] episode=1 step=1 predicted_action=FORWARD(1) execute=false
```

`execute=false` is the safety check: no action request was sent to the robot.
The robot-side gateway should continue serving images but must not print
`Socket command received` during this test.

Every instruction creates a timestamped directory under
`logs/go2_model_dry_run/`. It contains:

- `<timestamp>.mp4`: video assembled from the images used by the episode;
- `actions.jsonl`: ordered predictions and action IDs;
- `action_results.jsonl`: execution status, actual displacement, and yaw;
- `episode.json`: instruction and episode metadata;
- `episode_end.json`: terminal reason and final step;
- `driver_result.json`: completion or manual-stop result;
- `frames/`: original lossless PPM frames used to create the video.

Uni-NaVid and its checkpoint are loaded once per interactive session. Each
instruction is appended to the session's `instruction_queue.jsonl`; the same
RLinf workers reset the environment and consume the next instruction. The new
`episode_id` clears Uni-NaVid's visual cache without reloading model weights.
The complete shared console log is stored in the sibling
`_session_<timestamp>/run.log` directory.

## 4B. Supervised single-primitive motion test

Do not load the model for this test. Keep one operator at the robot with the
physical estop, use a clear level area, and verify `/sportmodestate` is fresh
before enabling motion:

```bash
ros2 topic hz /sportmodestate
ros2 action list -t | grep go2_vln
```

First send STOP. It does not require the motion-confirmation option:

```bash
python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.3.12 \
  --timeout 15 \
  --action stop
```

Then test one primitive per invocation. Moving primitives require the explicit
`--confirm-motion` safety acknowledgement:

```bash
python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.3.12 --timeout 15 --action left --confirm-motion

python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.3.12 --timeout 15 --action right --confirm-motion

python examples/embodiment/go2_vln_ros2/socket_smoke_test.py \
  --host 192.168.3.12 --timeout 15 --action forward --confirm-motion
```

The default expected motions are forward 0.25 m and turns of 30 degrees.
Because completion uses tolerance thresholds, a successful result may report
about 0.225--0.25 m or about 0.47--0.52 rad. The executor-side point-cloud
check is disabled by default with `enable_pointcloud_obstacle_check: false`;
the Go2 built-in collision-avoidance mode remains responsible for obstacles.

## 4C. Supervised end-to-end model motion test

Only after STOP, left, right, forward, built-in collision avoidance, and
heartbeat-loss stop have passed, run a three-step model-controlled episode:

```bash
python examples/embodiment/go2_vln_ros2/interactive_model_test.py \
  --host 192.168.3.12 \
  --max-steps 3 \
  --video-fps 2 \
  --execute-actions
```

The program requires typing the exact confirmation word `MOVE` before it
accepts an instruction. In this mode prediction lines show `execute=true`, and
outputs are written below `logs/go2_model_motion/`. Press Enter or Ctrl+C to
stop the episode; socket disconnect, goal cancellation, and the robot-local
heartbeat watchdog provide additional stopping paths, but do not replace the
physical estop.

Observations are strictly action-synchronous. RLinf waits for the robot-local
action result, waits the configured `post_action_settle_sec` (currently 0.0
seconds for Go2), and only then requests a fresh camera frame. The next model
prediction cannot begin before that frame arrives. The log line
`[GO2 OBSERVE] ... requesting_post_action_rgb=true` marks this boundary.

For Go2 evaluation, each raw generated response is printed as
`[UNINAVID RESPONSE]` and also written to
`rlinf/model_responses_rank_0.jsonl`. This distinguishes a model that never
generates `stop` from one that generates it later in a multi-action response.
When the selected action is STOP, the driver waits for StopMove, the final
post-action image, and result logging. The env and rollout workers exchange a
terminal sentinel, reset the episode, and wait for the next instruction while
keeping the loaded model alive.

## 5. Required failure tests

During a supported or low-speed primitive, verify each condition publishes
`StopMove`:

1. Terminate RLinf so TCP closes and UDP heartbeat stops.
2. Drop UDP 8766 while leaving TCP connected.
3. Disconnect Wi-Fi.
4. Stop `/sportmodestate`.
5. Trigger `/go2_vln/estop`.
6. Verify the Go2 built-in collision-avoidance mode with a supervised test.

The robot-side action timeout and state timeout remain authoritative even if
the socket gateway fails.

## 6. Communication latency

Measure the network baseline first:

```bash
ping -c 100 -i 0.05 192.168.3.15
```

Then measure authenticated TCP control RTT and fresh RGB transfer time without
moving the robot:

```bash
python examples/embodiment/go2_vln_ros2/socket_latency_test.py \
  --host 192.168.3.15 \
  --control-samples 100 \
  --image-samples 20
```

With the robot supported, `/sportmodestate` fresh, and estop behavior already
validated, optionally include STOP action RTT:

```bash
python examples/embodiment/go2_vln_ros2/socket_latency_test.py \
  --host 192.168.3.15 \
  --control-samples 100 \
  --image-samples 20 \
  --stop-samples 20
```

As an initial LAN acceptance target, use p95 below 50 ms for TCP control and
below 200 ms for a fresh raw RGB request. Record the actual baseline rather
than treating these values as hard real-time guarantees.
