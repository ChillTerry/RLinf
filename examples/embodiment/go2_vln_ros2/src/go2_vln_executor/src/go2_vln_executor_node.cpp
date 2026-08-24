// Copyright 2026 The RLinf Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "go2_vln_interfaces/action/execute_nav_primitive.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "unitree_api/msg/request.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"

namespace {

constexpr int64_t kStopMoveApiId = 1003;
constexpr int64_t kMoveApiId = 1008;

double wrap_angle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

}  // namespace

class Go2VlnExecutor : public rclcpp::Node {
 public:
  using ExecuteNavPrimitive = go2_vln_interfaces::action::ExecuteNavPrimitive;
  using GoalHandle = rclcpp_action::ServerGoalHandle<ExecuteNavPrimitive>;

  Go2VlnExecutor() : Node("go2_vln_executor") {
    action_name_ = declare_parameter<std::string>(
        "action_name", "/go2_vln/execute_nav_primitive");
    sport_request_topic_ = declare_parameter<std::string>(
        "sport_request_topic", "/api/sport/request");
    sport_state_topic_ =
        declare_parameter<std::string>("sport_state_topic", "/sportmodestate");
    pointcloud_topic_ = declare_parameter<std::string>(
        "pointcloud_topic", "/camera/depth/color/points");
    heartbeat_topic_ = declare_parameter<std::string>(
        "heartbeat_topic", "/go2_vln/heartbeat");
    estop_topic_ =
        declare_parameter<std::string>("estop_topic", "/go2_vln/estop");

    forward_distance_m_ = declare_parameter<double>("forward_distance_m", 0.25);
    turn_angle_rad_ = declare_parameter<double>("turn_angle_rad", M_PI / 6.0);
    distance_tolerance_m_ =
        declare_parameter<double>("distance_tolerance_m", 0.025);
    yaw_tolerance_rad_ =
        declare_parameter<double>("yaw_tolerance_rad", M_PI / 60.0);
    forward_speed_mps_ =
        declare_parameter<double>("forward_speed_mps", 0.3);
    yaw_speed_radps_ =
        declare_parameter<double>("yaw_speed_radps", 1.2);
    command_rate_hz_ = declare_parameter<double>("command_rate_hz", 20.0);
    action_timeout_sec_ =
        declare_parameter<double>("action_timeout_sec", 8.0);
    state_timeout_sec_ =
        declare_parameter<double>("state_timeout_sec", 0.5);
    heartbeat_timeout_sec_ =
        declare_parameter<double>("heartbeat_timeout_sec", 1.0);

    enable_pointcloud_obstacle_check_ =
        declare_parameter<bool>("enable_pointcloud_obstacle_check", false);
    obstacle_min_x_m_ =
        declare_parameter<double>("obstacle_min_x_m", 0.05);
    obstacle_max_x_m_ =
        declare_parameter<double>("obstacle_max_x_m", 0.60);
    obstacle_half_width_m_ =
        declare_parameter<double>("obstacle_half_width_m", 0.35);
    obstacle_min_z_m_ =
        declare_parameter<double>("obstacle_min_z_m", -0.30);
    obstacle_max_z_m_ =
        declare_parameter<double>("obstacle_max_z_m", 0.50);
    obstacle_min_points_ =
        declare_parameter<int>("obstacle_min_points", 3);

    sport_request_pub_ =
        create_publisher<unitree_api::msg::Request>(sport_request_topic_, 10);
    sport_state_sub_ = create_subscription<unitree_go::msg::SportModeState>(
        sport_state_topic_, rclcpp::SensorDataQoS(),
        [this](unitree_go::msg::SportModeState::ConstSharedPtr message) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          x_ = message->position[0];
          y_ = message->position[1];
          yaw_ = message->imu_state.rpy[2];
          state_received_ = true;
          last_state_time_ = now();
        });
    if (enable_pointcloud_obstacle_check_) {
      pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          pointcloud_topic_, rclcpp::SensorDataQoS(),
          std::bind(&Go2VlnExecutor::on_pointcloud, this,
                    std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "Point-cloud obstacle check enabled on %s",
                  pointcloud_topic_.c_str());
    } else {
      RCLCPP_WARN(
          get_logger(),
          "Point-cloud obstacle check disabled; relying on Go2 built-in "
          "collision avoidance");
    }
    heartbeat_sub_ = create_subscription<std_msgs::msg::UInt64>(
        heartbeat_topic_, 10,
        [this](std_msgs::msg::UInt64::ConstSharedPtr) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          heartbeat_received_ = true;
          last_heartbeat_time_ = now();
        });
    estop_sub_ = create_subscription<std_msgs::msg::Bool>(
        estop_topic_, 10,
        [this](std_msgs::msg::Bool::ConstSharedPtr message) {
          estop_active_.store(message->data);
          if (message->data) {
            publish_stop();
          }
        });

    action_server_ = rclcpp_action::create_server<ExecuteNavPrimitive>(
        this, action_name_,
        std::bind(&Go2VlnExecutor::handle_goal, this, std::placeholders::_1,
                  std::placeholders::_2),
        std::bind(&Go2VlnExecutor::handle_cancel, this,
                  std::placeholders::_1),
        std::bind(&Go2VlnExecutor::handle_accepted, this,
                  std::placeholders::_1));

    watchdog_timer_ = create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&Go2VlnExecutor::watchdog, this));
    RCLCPP_INFO(get_logger(), "Go2 VLN executor listening on %s",
                action_name_.c_str());
  }

  ~Go2VlnExecutor() override { publish_stop(); }

 private:
  rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID &,
      std::shared_ptr<const ExecuteNavPrimitive::Goal> goal) {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    if (goal_reserved_ || goal->action > ExecuteNavPrimitive::Goal::NO_OP) {
      return rclcpp_action::GoalResponse::REJECT;
    }
    auto previous = last_sequence_by_episode_.find(goal->episode_id);
    if (previous != last_sequence_by_episode_.end() &&
        goal->sequence_id <= previous->second) {
      RCLCPP_WARN(get_logger(),
                  "Rejected replayed goal episode=%lu sequence=%lu",
                  goal->episode_id, goal->sequence_id);
      return rclcpp_action::GoalResponse::REJECT;
    }
    last_sequence_by_episode_[goal->episode_id] = goal->sequence_id;
    goal_reserved_ = true;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
      const std::shared_ptr<GoalHandle>) {
    publish_stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle) {
    std::thread(
        [this, goal_handle]() {
          execute(goal_handle);
          std::lock_guard<std::mutex> lock(goal_mutex_);
          goal_reserved_ = false;
        })
        .detach();
  }

  void execute(const std::shared_ptr<GoalHandle> goal_handle) {
    const auto goal = goal_handle->get_goal();
    const auto start_time = now();
    double start_x;
    double start_y;
    double start_yaw;
    if (!read_fresh_state(start_x, start_y, start_yaw)) {
      finish(goal_handle, ExecuteNavPrimitive::Result::STALE_STATE,
             "No fresh /sportmodestate sample.", 0.0, 0.0);
      return;
    }
    if (estop_active_.load()) {
      finish(goal_handle, ExecuteNavPrimitive::Result::ESTOP,
             "Emergency stop is active.", 0.0, 0.0);
      return;
    }

    if (goal->action == ExecuteNavPrimitive::Goal::STOP ||
        goal->action == ExecuteNavPrimitive::Goal::NO_OP) {
      publish_stop();
      finish(goal_handle, ExecuteNavPrimitive::Result::SUCCEEDED,
             goal->action == ExecuteNavPrimitive::Goal::STOP
                 ? "Robot stopped."
                 : "No-op completed.",
             0.0, 0.0);
      return;
    }
    if (enable_pointcloud_obstacle_check_ &&
        goal->action == ExecuteNavPrimitive::Goal::FORWARD &&
        obstacle_active_.load()) {
      finish(goal_handle, ExecuteNavPrimitive::Result::OBSTACLE,
             "Obstacle detected before forward motion.", 0.0, 0.0);
      return;
    }

    motion_active_.store(true);
    const bool forward = goal->action == ExecuteNavPrimitive::Goal::FORWARD;
    const double yaw_command =
        forward ? 0.0
                : (goal->action == ExecuteNavPrimitive::Goal::LEFT ? 1.0 : -1.0) *
                      yaw_speed_radps_;
    rclcpp::Rate rate(command_rate_hz_);
    while (rclcpp::ok()) {
      double x;
      double y;
      double yaw;
      if (!read_fresh_state(x, y, yaw)) {
        finish(goal_handle, ExecuteNavPrimitive::Result::STALE_STATE,
               "Sport state became stale.", 0.0, 0.0);
        return;
      }
      const double distance = std::hypot(x - start_x, y - start_y);
      const double yaw_delta = wrap_angle(yaw - start_yaw);

      if (goal_handle->is_canceling()) {
        publish_stop();
        motion_active_.store(false);
        auto result = make_result(ExecuteNavPrimitive::Result::CONTROL_ERROR,
                                  "Goal canceled.", distance, yaw_delta);
        goal_handle->canceled(result);
        return;
      }
      if (estop_active_.load()) {
        finish(goal_handle, ExecuteNavPrimitive::Result::ESTOP,
               "Emergency stop activated.", distance, yaw_delta);
        return;
      }
      if (!heartbeat_is_fresh()) {
        finish(goal_handle, ExecuteNavPrimitive::Result::CONTROL_ERROR,
               "RLinf heartbeat timed out.", distance, yaw_delta);
        return;
      }
      if ((now() - start_time).seconds() > action_timeout_sec_) {
        finish(goal_handle, ExecuteNavPrimitive::Result::TIMEOUT,
               "Navigation primitive timed out.", distance, yaw_delta);
        return;
      }

      bool reached = false;
      if (goal->action == ExecuteNavPrimitive::Goal::FORWARD) {
        if (enable_pointcloud_obstacle_check_ && obstacle_active_.load()) {
          finish(goal_handle, ExecuteNavPrimitive::Result::OBSTACLE,
                 "Obstacle detected during forward motion.", distance,
                 yaw_delta);
          return;
        }
        reached = distance >= forward_distance_m_ - distance_tolerance_m_;
      } else {
        reached = std::abs(yaw_delta) >=
                  turn_angle_rad_ - yaw_tolerance_rad_;
      }

      auto feedback = std::make_shared<ExecuteNavPrimitive::Feedback>();
      feedback->distance_m = static_cast<float>(distance);
      feedback->yaw_rad = static_cast<float>(yaw_delta);
      feedback->nearest_obstacle_m =
          static_cast<float>(nearest_obstacle_m_.load());
      feedback->state = reached ? "stopping" : "moving";
      goal_handle->publish_feedback(feedback);

      if (reached) {
        finish(goal_handle, ExecuteNavPrimitive::Result::SUCCEEDED,
               "Navigation primitive completed.", distance, yaw_delta);
        return;
      }

      // Sport Move is a velocity request, not a latched position goal. Renew
      // it at command_rate_hz_ until odometry reaches the primitive threshold.
      // This makes the outer primitive controller robust to command expiry.
      publish_move(forward ? forward_speed_mps_ : 0.0, 0.0, yaw_command);
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Primitive progress: action=%u distance=%.3f/%.3f yaw=%.3f/%.3f",
          static_cast<unsigned>(goal->action), distance, forward_distance_m_,
          yaw_delta, turn_angle_rad_);
      rate.sleep();
    }

    finish(goal_handle, ExecuteNavPrimitive::Result::CONTROL_ERROR,
           "ROS2 shutdown interrupted motion.", 0.0, 0.0);
  }

  void finish(const std::shared_ptr<GoalHandle> &goal_handle, uint8_t status,
              const std::string &message, double distance, double yaw) {
    publish_stop();
    motion_active_.store(false);
    auto result = make_result(status, message, distance, yaw);
    if (goal_handle->is_active()) {
      if (status == ExecuteNavPrimitive::Result::SUCCEEDED ||
          status == ExecuteNavPrimitive::Result::OBSTACLE) {
        goal_handle->succeed(result);
      } else {
        goal_handle->abort(result);
      }
    }
  }

  std::shared_ptr<ExecuteNavPrimitive::Result> make_result(
      uint8_t status, const std::string &message, double distance,
      double yaw) const {
    auto result = std::make_shared<ExecuteNavPrimitive::Result>();
    result->status = status;
    result->message = message;
    result->distance_m = static_cast<float>(distance);
    result->yaw_rad = static_cast<float>(yaw);
    return result;
  }

  bool read_fresh_state(double &x, double &y, double &yaw) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!state_received_ ||
        (now() - last_state_time_).seconds() > state_timeout_sec_) {
      return false;
    }
    x = x_;
    y = y_;
    yaw = yaw_;
    return true;
  }

  bool heartbeat_is_fresh() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return heartbeat_received_ &&
           (now() - last_heartbeat_time_).seconds() <= heartbeat_timeout_sec_;
  }

  void watchdog() {
    if (!motion_active_.load()) {
      return;
    }
    double x;
    double y;
    double yaw;
    if (estop_active_.load() || !heartbeat_is_fresh() ||
        !read_fresh_state(x, y, yaw)) {
      publish_stop();
    }
  }

  void publish_move(double vx, double vy, double vyaw) {
    unitree_api::msg::Request request;
    request.header.identity.id = request_id_.fetch_add(1);
    request.header.identity.api_id = kMoveApiId;
    std::ostringstream parameter;
    parameter << "{\"x\":" << vx << ",\"y\":" << vy << ",\"z\":"
              << vyaw << "}";
    request.parameter = parameter.str();
    sport_request_pub_->publish(request);
  }

  void publish_stop() {
    unitree_api::msg::Request request;
    request.header.identity.id = request_id_.fetch_add(1);
    request.header.identity.api_id = kStopMoveApiId;
    sport_request_pub_->publish(request);
  }

  void on_pointcloud(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) {
    std::size_t obstacle_points = 0;
    double nearest = std::numeric_limits<double>::infinity();
    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*cloud, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y(*cloud, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z(*cloud, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        if (!std::isfinite(*x) || !std::isfinite(*y) ||
            !std::isfinite(*z)) {
          continue;
        }
        if (*x >= obstacle_min_x_m_ && *x <= obstacle_max_x_m_ &&
            std::abs(*y) <= obstacle_half_width_m_ &&
            *z >= obstacle_min_z_m_ && *z <= obstacle_max_z_m_) {
          ++obstacle_points;
          nearest = std::min(nearest, static_cast<double>(*x));
        }
      }
    } catch (const std::runtime_error &error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                            "Invalid PointCloud2: %s", error.what());
      return;
    }
    obstacle_active_.store(
        obstacle_points >= static_cast<std::size_t>(obstacle_min_points_));
    nearest_obstacle_m_.store(nearest);
  }

  std::string action_name_;
  std::string sport_request_topic_;
  std::string sport_state_topic_;
  std::string pointcloud_topic_;
  std::string heartbeat_topic_;
  std::string estop_topic_;

  double forward_distance_m_;
  double turn_angle_rad_;
  double distance_tolerance_m_;
  double yaw_tolerance_rad_;
  double forward_speed_mps_;
  double yaw_speed_radps_;
  double command_rate_hz_;
  double action_timeout_sec_;
  double state_timeout_sec_;
  double heartbeat_timeout_sec_;
  bool enable_pointcloud_obstacle_check_;
  double obstacle_min_x_m_;
  double obstacle_max_x_m_;
  double obstacle_half_width_m_;
  double obstacle_min_z_m_;
  double obstacle_max_z_m_;
  int obstacle_min_points_;

  rclcpp_action::Server<ExecuteNavPrimitive>::SharedPtr action_server_;
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr sport_request_pub_;
  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr
      sport_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt64>::SharedPtr heartbeat_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr estop_sub_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;

  std::mutex state_mutex_;
  bool state_received_ = false;
  bool heartbeat_received_ = false;
  double x_ = 0.0;
  double y_ = 0.0;
  double yaw_ = 0.0;
  rclcpp::Time last_state_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_heartbeat_time_{0, 0, RCL_ROS_TIME};

  std::mutex goal_mutex_;
  bool goal_reserved_ = false;
  std::unordered_map<uint64_t, uint64_t> last_sequence_by_episode_;
  std::atomic<bool> motion_active_{false};
  std::atomic<bool> obstacle_active_{false};
  std::atomic<bool> estop_active_{false};
  std::atomic<double> nearest_obstacle_m_{
      std::numeric_limits<double>::infinity()};
  std::atomic<int64_t> request_id_{1};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Go2VlnExecutor>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  node.reset();
  rclcpp::shutdown();
  return 0;
}
