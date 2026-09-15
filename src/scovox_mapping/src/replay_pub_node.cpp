// scovox_replay_pub -- publishes a staged SceneNN or KITTI unit onto the topics
// scovox_node subscribes to, so the NODE can be profiled on the same data the
// offline replay drivers measure.
//
// It deliberately does NOT publish semantics. scovox_node reads the same
// `%06u.topk` blobs the drivers read, straight off disk, keyed by the frame
// index this publisher writes into `header.stamp.nanosec`. Keeping the labels
// on that path is what makes the measurement a MAPPING profile rather than a
// segmentation-network benchmark, and it is what makes a voxel-for-voxel
// comparison against a driver dump meaningful at all.
//
// Everything here exists to reproduce a driver's per-frame inputs exactly.
// Where a convention had to be matched rather than chosen, the matching rule is
// stated at the point it is applied:
//
//   * FRAME INDEX. SceneNN files are 1-based (`depth%05d.png`, `%06d.topk` of
//     f+1); KITTI files are 0-based (`velodyne/%06d.bin`, `%06d.topk` of f).
//     The node derives its index as `stamp.nanosec & 0xFFFF`, so the stamp
//     carries the dataset's own numbering and the two never drift by one.
//   * SUBSAMPLING. The drivers iterate a stride-subsampled image, read depth at
//     `(u*stride, v*stride)` of the full PNG, and index the .topk blob at
//     `(u, v)` -- so the staged blobs are at the SUBSAMPLED size. The node
//     instead indexes .topk with its own loop variables. The two agree only if
//     this publisher subsamples the depth image itself and the node then runs
//     `stride: 1` against subsampled intrinsics. That is what it does.
//   * OPTICAL ROTATION. The node post-multiplies an RGB-D TF by the fixed
//     optical rotation kR; the trajectory files are already optical-to-world.
//     So the TF published here carries `R * kR^T`, which the node turns back
//     into `R`. LiDAR has no such rotation and is published as-is.
//   * TF UP FRONT. Every pose is broadcast before the first frame, not
//     alongside it. The node rejects an RGB-D frame outright on an exact-stamp
//     TF miss, and a race between /tf delivery and image delivery would show up
//     as silently dropped frames rather than as an error.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Geometry>
#include <png.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Intrinsics { float fx{0}, fy{0}, cx{0}, cy{0}; int width{0}, height{0}; };

// SceneNN's asus.ini: whitespace-separated `key value` lines.
Intrinsics loadAsusIni(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open intrinsics: " + path);
  Intrinsics k; std::string key; double v;
  while (f >> key >> v) {
    if      (key == "fx")     k.fx = (float)v;
    else if (key == "fy")     k.fy = (float)v;
    else if (key == "cx")     k.cx = (float)v;
    else if (key == "cy")     k.cy = (float)v;
    // The key names are the replay's, not a guess: asus.ini spells the image
    // size `depth_width`/`depth_height`.
    else if (key == "depth_width")  k.width  = (int)v;
    else if (key == "depth_height") k.height = (int)v;
  }
  if (k.fx <= 0.f || k.fy <= 0.f || k.width <= 0 || k.height <= 0)
    throw std::runtime_error("intrinsics file missing fx/fy/depth_width/depth_height: " + path);
  return k;
}

Intrinsics subsampled(const Intrinsics& k, int s) {
  if (s <= 1) return k;
  Intrinsics o;
  o.fx = k.fx / (float)s; o.fy = k.fy / (float)s;
  o.cx = k.cx / (float)s; o.cy = k.cy / (float)s;
  o.width = k.width / s;  o.height = k.height / s;
  return o;
}

// A `trajectory.log`: repeating [index line][4 rows of 4 floats]. Both datasets
// are staged in this one format -- SceneNN camera-to-world, KITTI
// velodyne-to-world -- so one reader serves both.
std::vector<Eigen::Matrix4f> loadTrajectoryLog(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open trajectory: " + path);
  std::vector<Eigen::Matrix4f> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    Eigen::Matrix4f M;
    bool ok = true;
    for (int r = 0; r < 4 && ok; ++r) {
      if (!std::getline(f, line)) { ok = false; break; }
      std::istringstream is(line);
      for (int c = 0; c < 4; ++c) if (!(is >> M(r, c))) { ok = false; break; }
    }
    if (!ok) break;
    out.push_back(M);
  }
  return out;
}

// One 16-bit grayscale depth PNG, millimetres, as SceneNN ships them.
bool loadDepthPng(const std::string& path, int* w, int* h, std::vector<uint16_t>* mm) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return false;
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) { std::fclose(fp); return false; }
  png_infop info = png_create_info_struct(png);
  if (!info || setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, info ? &info : nullptr, nullptr);
    std::fclose(fp); return false;
  }
  png_init_io(png, fp);
  png_read_info(png, info);
  const int W = (int)png_get_image_width(png, info);
  const int H = (int)png_get_image_height(png, info);
  if (png_get_bit_depth(png, info) != 16 ||
      png_get_color_type(png, info) != PNG_COLOR_TYPE_GRAY) {
    png_destroy_read_struct(&png, &info, nullptr); std::fclose(fp); return false;
  }
  png_set_swap(png);  // PNG is big-endian on the wire; we want host order
  mm->assign((size_t)W * H, 0);
  std::vector<png_bytep> rows((size_t)H);
  for (int y = 0; y < H; ++y) rows[y] = (png_bytep)(mm->data() + (size_t)y * W);
  png_read_image(png, rows.data());
  png_destroy_read_struct(&png, &info, nullptr);
  std::fclose(fp);
  *w = W; *h = H;
  return true;
}

// One velodyne scan: float32 x, y, z, intensity, in file order. The order is
// load-bearing -- the .topk rows are per point and follow it.
bool loadVelodyneBin(const std::string& path, std::vector<float>* xyzi) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamsize n = f.tellg();
  if (n <= 0 || n % (4 * sizeof(float)) != 0) return false;
  f.seekg(0);
  xyzi->resize((size_t)n / sizeof(float));
  f.read(reinterpret_cast<char*>(xyzi->data()), n);
  return static_cast<bool>(f);
}

std::string frameName(const std::string& dir, const char* prefix,
                      int idx, const char* ext, int width) {
  char fmt[32], buf[64];
  std::snprintf(fmt, sizeof(fmt), "%%s%%0%dd%%s", width);
  std::snprintf(buf, sizeof(buf), fmt, prefix, idx, ext);
  return dir + "/" + buf;
}

}  // namespace

class ReplayPub : public rclcpp::Node {
public:
  ReplayPub() : Node("scovox_replay_pub"), tf_(*this) {
    dataset_    = declare_parameter<std::string>("dataset", "scenenn");
    data_dir_   = declare_parameter<std::string>("data_dir", "");
    traj_path_  = declare_parameter<std::string>("traj", "");
    intr_path_  = declare_parameter<std::string>("intrinsics", "");
    stride_     = std::max(1, (int)declare_parameter<int>("stride", 2));
    rate_hz_    = declare_parameter<double>("rate_hz", 5.0);
    max_frames_ = (int)declare_parameter<int>("max_frames", 0);
    world_frame_  = declare_parameter<std::string>("world_frame", "map");
    sensor_frame_ = declare_parameter<std::string>("sensor_frame", "");
    // The node looks the ray origin up in its own `base_frame`, separately from
    // the depth frame. Publishing one transform and pointing both at it keeps
    // the observer exactly on the trajectory, as the drivers have it.
    if (sensor_frame_.empty())
      sensor_frame_ = (dataset_ == "kitti") ? "velodyne" : "camera_optical";
    depth_topic_ = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    seg_topic_   = declare_parameter<std::string>("seg_topic",   "/seg/labels");
    info_topic_  = declare_parameter<std::string>("camera_info_topic", "/camera/depth/camera_info");
    cloud_topic_ = declare_parameter<std::string>("pointcloud_topic", "/ouster/points");
    tf_lead_sec_ = declare_parameter<double>("tf_lead_sec", 2.0);    // How long to stay alive after the last frame, so reliable QoS can finish
    // retransmitting it before the writer goes away.
    linger_sec_ = declare_parameter<double>("linger_sec", 5.0);


    if (data_dir_.empty() || traj_path_.empty())
      throw std::runtime_error("data_dir and traj are required");
    traj_ = loadTrajectoryLog(traj_path_);
    if (traj_.empty()) throw std::runtime_error("no poses in " + traj_path_);
    n_frames_ = (int)traj_.size();
    if (max_frames_ > 0 && max_frames_ < n_frames_) n_frames_ = max_frames_;

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    if (dataset_ == "kitti") {
      cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic_, qos);
    } else {
      if (intr_path_.empty()) throw std::runtime_error("intrinsics is required for scenenn");
      K_ = subsampled(loadAsusIni(intr_path_), stride_);
      depth_pub_ = create_publisher<sensor_msgs::msg::Image>(depth_topic_, qos);
      seg_pub_   = create_publisher<sensor_msgs::msg::Image>(seg_topic_, qos);
      info_pub_  = create_publisher<sensor_msgs::msg::CameraInfo>(info_topic_, rclcpp::QoS(10));
    }

    RCLCPP_INFO(get_logger(),
        "replay_pub: dataset=%s frames=%d rate=%.2f Hz stride=%d world=%s sensor=%s",
        dataset_.c_str(), n_frames_, rate_hz_, stride_,
        world_frame_.c_str(), sensor_frame_.c_str());
    if (dataset_ != "kitti")
      RCLCPP_INFO(get_logger(), "replay_pub: subsampled intrinsics %dx%d fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                  K_.width, K_.height, K_.fx, K_.fy, K_.cx, K_.cy);

    // Wait for a /tf subscriber before broadcasting. The broadcaster is plain
    // reliable KeepLast, NOT transient_local, so anything sent before DDS
    // discovery matches the mapper's TransformListener is simply lost -- and
    // the failure is silent and total: every frame then fails its exact-stamp
    // lookup with "frame map does not exist", which reads like a config error
    // rather than a race. Waiting is the whole fix.
    {
      int waited = 0;
      while (rclcpp::ok() && count_subscribers("/tf") == 0 && waited < 300) {
        rclcpp::sleep_for(std::chrono::milliseconds(100));
        ++waited;
      }
      if (count_subscribers("/tf") == 0)
        RCLCPP_WARN(get_logger(), "replay_pub: no /tf subscriber after 30 s — "
                                  "broadcasting anyway, but the mapper will not see it");
      else
        RCLCPP_INFO(get_logger(), "replay_pub: /tf subscriber present after %.1f s", waited * 0.1);
      // Discovery matching is not instantaneous even once the count is up.
      rclcpp::sleep_for(std::chrono::milliseconds(500));
    }

    t0_ = now() + rclcpp::Duration::from_seconds(tf_lead_sec_);
    broadcastAllTf();
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz_),
                               std::bind(&ReplayPub::tick, this));
  }

private:
  // Frame index as it appears in this dataset's filenames, and therefore as the
  // node must derive it from the stamp.
  int fileIndex(int f) const { return (dataset_ == "kitti") ? f : f + 1; }

  // The stamp does two jobs: it is the exact-stamp TF key, and its low 16
  // bits ARE the frame id the mapper reads to pick a `.topk` blob
  // (`stamp.nanosec & 0xFFFF`). So it is built field by field rather than
  // from a double -- rclcpp::Time's seconds argument is an int32_t, and
  // handing it a fractional second truncates every frame in a second to the
  // same instant, leaving the low bits carrying the only ordering there is.
  rclcpp::Time stampFor(int f) const {
    const int64_t ns = t0_.nanoseconds() +
                       static_cast<int64_t>(std::llround(1.0e9 * (double)f / rate_hz_));
    const int32_t  sec  = static_cast<int32_t>(ns / 1000000000LL);
    uint32_t       nsec = static_cast<uint32_t>(ns % 1000000000LL);
    // Clear the low 16 bits, then write the file index into them. Frames are
    // 1/rate apart -- never less than a millisecond -- so perturbing the stamp
    // by at most 65 us can neither reorder two frames nor collide them.
    nsec = (nsec & ~0xFFFFu) | static_cast<uint32_t>(fileIndex(f));
    return rclcpp::Time(sec, nsec, RCL_ROS_TIME);
  }

  // Every pose, before the first frame. See the header note on TF UP FRONT.
  void broadcastAllTf() {
    static const Eigen::Matrix3f kR =
        (Eigen::Matrix3f() << 0,0,1, -1,0,0, 0,-1,0).finished();
    const bool rgbd = (dataset_ != "kitti");
    std::vector<geometry_msgs::msg::TransformStamped> all;
    all.reserve((size_t)n_frames_);
    for (int f = 0; f < n_frames_; ++f) {
      const Eigen::Matrix4f& M = traj_[(size_t)f];
      Eigen::Matrix3f R = M.block<3,3>(0,0);
      // The node post-multiplies an RGB-D pose by kR; undo it here so what the
      // node ends up integrating with is the trajectory's own rotation.
      if (rgbd) R = R * kR.transpose();
      const Eigen::Quaternionf q(R);
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp = stampFor(f);
      t.header.frame_id = world_frame_;
      t.child_frame_id  = sensor_frame_;
      t.transform.translation.x = M(0,3);
      t.transform.translation.y = M(1,3);
      t.transform.translation.z = M(2,3);
      t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y();
      t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
      all.push_back(t);
    }
    tf_.sendTransform(all);
    RCLCPP_INFO(get_logger(), "replay_pub: broadcast %zu poses; first frame in %.1f s",
                all.size(), tf_lead_sec_);
  }

  void tick() {
    if (f_ >= n_frames_) {
      if (!done_) {
        done_ = true;
        done_at_ = now();
        RCLCPP_INFO(get_logger(), "replay_pub: published %d frames (%d missing on disk)",
                    f_ - missing_, missing_);
      }
      // Exit rather than idle. The runner waits on this process to know the
      // feed is over, and the mapper still has a backlog to drain afterwards,
      // so a publisher that never returns stalls the whole run. Leaving is
      // safe: reliable samples already sit in the READER's history and are
      // owned by it, and every TF the mapper needs is in its own buffer.
      // The grace period covers retransmission of the last few samples.
      if ((now() - done_at_).seconds() >= linger_sec_) {
        RCLCPP_INFO(get_logger(), "replay_pub: lingered %.1f s; exiting", linger_sec_);
        rclcpp::shutdown();
      }
      return;
    }
    const int f = f_++;
    if (dataset_ == "kitti") publishCloud(f); else publishRgbd(f);
  }

  void publishRgbd(int f) {
    int W = 0, H = 0;
    std::vector<uint16_t> mm;
    // `data_dir` is the UNIT directory for both sensors, exactly as the replay
    // drivers take `--scene` / `--seq`. SceneNN's depth lives one level down.
    const std::string p =
        frameName(data_dir_ + "/frames/depth", "depth", fileIndex(f), ".png", 5);
    if (!loadDepthPng(p, &W, &H, &mm)) {
      ++missing_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "replay_pub: cannot read %s", p.c_str());
      return;
    }
    const auto stamp = stampFor(f);
    // Subsample here rather than letting the node stride: the staged .topk
    // blobs are at this size, and the node indexes them with its own loop
    // variables. See the header note on SUBSAMPLING.
    sensor_msgs::msg::Image d;
    d.header.stamp = stamp; d.header.frame_id = sensor_frame_;
    d.width = (uint32_t)K_.width; d.height = (uint32_t)K_.height;
    d.encoding = "16UC1"; d.is_bigendian = 0; d.step = d.width * 2;
    d.data.resize((size_t)d.step * d.height);
    auto* out = reinterpret_cast<uint16_t*>(d.data.data());
    for (int v = 0; v < K_.height; ++v)
      for (int u = 0; u < K_.width; ++u)
        out[(size_t)v * K_.width + u] = mm[(size_t)(v * stride_) * W + (size_t)(u * stride_)];

    // The node's RGB-D callback needs a same-size seg image in a colour
    // encoding to fire at all, but never reads its pixels while topk is on.
    // This is a placeholder, and it is pure overhead a real robot would not
    // send -- which is why the profile's headline is integrate_ms.
    sensor_msgs::msg::Image s;
    s.header = d.header;
    s.width = d.width; s.height = d.height;
    s.encoding = "rgb8"; s.is_bigendian = 0; s.step = s.width * 3;
    s.data.assign((size_t)s.step * s.height, 0);

    sensor_msgs::msg::CameraInfo ci;
    ci.header = d.header;
    ci.width = d.width; ci.height = d.height;
    ci.k = {K_.fx, 0, K_.cx, 0, K_.fy, K_.cy, 0, 0, 1};
    ci.p = {K_.fx, 0, K_.cx, 0, 0, K_.fy, K_.cy, 0, 0, 0, 1, 0};
    ci.distortion_model = "plumb_bob";

    info_pub_->publish(ci);
    depth_pub_->publish(d);
    seg_pub_->publish(s);
  }

  void publishCloud(int f) {
    std::vector<float> xyzi;
    const std::string p = frameName(data_dir_ + "/velodyne", "", fileIndex(f), ".bin", 6);
    if (!loadVelodyneBin(p, &xyzi)) {
      ++missing_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "replay_pub: cannot read %s", p.c_str());
      return;
    }
    const size_t n = xyzi.size() / 4;
    sensor_msgs::msg::PointCloud2 c;
    c.header.stamp = stampFor(f);
    c.header.frame_id = sensor_frame_;
    c.height = 1; c.width = (uint32_t)n;
    c.is_dense = false; c.is_bigendian = false;
    const char* names[4] = {"x", "y", "z", "intensity"};
    c.fields.resize(4);
    for (int i = 0; i < 4; ++i) {
      c.fields[(size_t)i].name = names[i];
      c.fields[(size_t)i].offset = (uint32_t)(4 * i);
      c.fields[(size_t)i].datatype = sensor_msgs::msg::PointField::FLOAT32;
      c.fields[(size_t)i].count = 1;
    }
    c.point_step = 16;
    c.row_step = c.point_step * c.width;
    // Raw file order, unfiltered: the node applies its own range gate, and the
    // .topk rows are per point in exactly this order.
    c.data.resize((size_t)c.row_step);
    std::memcpy(c.data.data(), xyzi.data(), c.data.size());
    cloud_pub_->publish(c);
  }

  std::string dataset_, data_dir_, traj_path_, intr_path_;
  std::string world_frame_, sensor_frame_;
  std::string depth_topic_, seg_topic_, info_topic_, cloud_topic_;
  int    stride_{2}, max_frames_{0}, n_frames_{0}, f_{0}, missing_{0};
  double rate_hz_{5.0}, tf_lead_sec_{2.0};
  bool   done_{false};
  double linger_sec_{5.0};
  rclcpp::Time done_at_;
  Intrinsics K_;
  std::vector<Eigen::Matrix4f> traj_;
  rclcpp::Time t0_;
  tf2_ros::TransformBroadcaster tf_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_, seg_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<ReplayPub>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("scovox_replay_pub"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
