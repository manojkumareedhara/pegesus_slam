#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <vector>
#include <memory>
#include <mutex>

namespace mono_slam
{

    class Feature;
    class MapPoint;
    struct IMUPreIntegration;

    struct IMUData
    {
        double timestamp;
        Eigen::Vector3f linear_accel;
        Eigen::Vector3f angular_vel;

        IMUData() = default;
        IMUData(double ts, const Eigen::Vector3f &accel, const Eigen::Vector3f &gyro)
            : timestamp(ts), linear_accel(accel), angular_vel(gyro) {};
    };

    class Frame
    {
    public:
        Frame(long long timestamp, int frame_id);
        Frame(long long timestamp, int frame_id, double fx, double fy, double cx, double cy, const std::vector<double> &dist_coeffs);

        ~Frame();
        long long get_timestamp() const { return m_timestamp; };
        int get_frame_id() const { return m_frame_id; };
        const cv::Mat &get_image() const { return m_image; };
        const std::vector<std::shared_ptr<Feature>> &get_features() const { return m_features; };
        std::vector<std::shared_ptr<Feature>> &get_features_mutable() { return m_features; };
        const Eigen::Matrix3f &get_rotation() const { return m_rotation; };
        const Eigen::Vector3f &get_translation() const { return m_translation; };
        bool is_keyframe() const { return m_is_keyframe; };

        void set_pose(const Eigen::Matrix3f &rotation, const Eigen::Vector3f &translation);
        void set_Twb(const Eigen::Matrix4f &T_wb);
        Eigen::Matrix4f get_Twb();
        Eigen::Matrix4f get_Twc();

        Sophus::SE3f get_world_pose();
        void set_world_pose(const Sophus::SE3f &pose);
        Eigen::Vector3f get_velocity() const { return m_velocity; };
        void set_velocity(const Eigen::Vector3f &velocity) { m_velocity = velocity; };

        void initialize_velocity_from_preintegration();

        Eigen::Vector3f get_accel_bias() const { return m_accel_bias; }
        void set_accel_bias(const Eigen::Vector3f &accel_bias) { m_accel_bias = accel_bias; }
        Eigen::Vector3f get_gyro_bias() const { return m_gyro_bias; }
        void set_gyro_bias(const Eigen::Vector3f &gyro_bias) { m_gyro_bias = gyro_bias; }

        double get_dt_from_last_keyframe() const { return m_dt_from_last_kf; }
        void set_dt_from_last_keyframe(double dt) { m_dt_from_last_kf = dt; }
        void set_keyframe(bool is_key_frame) { m_is_keyframe = is_keyframe; }
        void set_referance_keyframe(std::shared_ptr<Frame> referance_keyframe);
        std::shared_ptr<Frame> get_referance_keyframe() const;
        const Eigen::Matrix4f &get_relative_transform() const { return m_T_relative_from_ref; }
        void set_relative_transform(const Eigen::Matrix4f &T_relative) { m_T_relative_from_ref = T_relative; }

        static void st_last_keyframe(std::shared_ptr<Frame> keyframe) { m_last_keyframe = keyframe; }
        static std::shared_ptr<Frame> get_last_keyframe() { return m_last_keyframe; }

        void release_imgs();

        void add_features(std::shared_ptr<Features> features);
        void remove_feature(int feature_id);
        std::shared_ptr<Feature> get_feature(int feature_id);
        std::shared_ptr<const Feature> get_feature(int feature_id) const;
        size_t feature_count() const { return m_features.size(); }
        int get_feature_index(int feature_id) const;

        void initialize_map_points();
        void setmap_point(int feature_index, std::shared_ptr<MapPoint> map_point);
        std::shared_ptr<MapPoint> get_map_point(int feature_index) const;
        bool has_map_point(int feature_index) const;
        const std::vector<std::shared_ptr<MapPoint>> &get_map_points() const { return m_map_points; }
        std::vector<std::shared_ptr<MapPoint>> &get_map_points_mutable() { return m_map_points; }

        void set_outlier_flag(int feature_index, bool is_outlier);
        bool get_outlier_flag(int feature_index) const;
        const std::vector<bool> &get_outlier_flags() const { return m_outlier_flags; }
        void initialize_outlier_flags();
        void set_distortion_coeffs(const std::vector<double> &dist_coeffs);

        float get_fx() const;
        float get_fy() const;
        float get_cx() const;
        float get_cy() const;

        double get_undist_x_min() const { return m_undist_x_min; }
        double get_undist_x_max() const { return m_undist_x_max; }
        double get_undist_y_min() const { return m_undist_y_min; }
        double get_undist_y_max() const { return m_undist_y_max; }

        void set_T_CV(const Eigen::Matrix4d &T_CB) { m_T_CB = T_CB; }
        const Eigen::Matrix4d &get_Tcb() const { return m_T_CB; }
        cv::Point2f undistort_point(const cv::Point2f &point, int border_size = 0) const;

        // IMU management
        void set_imu_data_from_lastkeyframe(const std::vector<IMUData> &imu_data);
        const std::vector<IMUDData> &get_imu_data_from_last_frame() const { return m_imu_vec_from_last_frame; }
        std::vector<IMUData> &get_imu_data_from_last_frame_mutable() { return m_imu_vec_frm_last_frame; }
        bool has_imu_data() const { return !m_imu_vec_from_last_frame.empty(); }
        size_t get_imu_data_count() const { return m_imu_vec_from_last_keyframe.size(); }

        void set_imu_data_since_last_keyframe(const std::vector<IMUData> &imu_data);
        const std::vector<IMUData> &get_imu_data_since_last_keyframe() const { return m_imu_vec_since_last_keyframe; }
        bool has_keyframe_imu_data() const { return !m_imu_vec_since_last_keyframe.empty(); }
        size_t get_keyframe_imu_data_count() const { return m_imu_vec_since_last_keyframe.size(); }

        void set_imu_preintergtation_from_last_keyframe(std::shared_ptr<IMUPreintegration> preintegration);
        std::shared_ptr<IMUPreintegration> get_imu_preintegration_from_last_keyframe() const { return m_imu_preintegration_from_last_frame; }
        bool has_imu_preintegration_from_last_keyframe() const { return m_imu_preintegration_from_last_keyframe != nullptr; }

        void set_imu_preintegration_from_last_frame(std::shared_ptr<IMUPreintegration> preintegration);
        std::shared_ptr<IMUPreintegration> get_imu_preintegration_from_last_frame() const { return m_imu_preintegration_from_last_frame; }
        bool has_imu_preintegration_from_last_frame() const { return m_imu_preintegration_from_last_frame != nullptr; }

    private:
        long long m_timestamp;
        int m_frame_id;
        cv::Mat m_image;
        std::vector<std::shared_ptr<Feature>> m_features;
        std::unordered_map<int, size_t> m_feature_id_to_index;
        std::vector<std::shared_ptr<MapPoint>> m_map_points;
        std::vector<bool> m_outlier_flags;
        double m_fx, m_fy, m_cx, m_cy;
        std::vector<double> m_distortion_coeffs;

        double m_undist_x_min, m_undist_x_max;
        double m_undist_y_min, m_undist_y_max;

        Eigen::Matrix4d m_T_CV; // camera to body transformation

        Eigen::Matrix3f matrix_rotation;
        Eigen::Vector3f m_transformation;
        bool m_is_keyframe;

        Sophus::SE3f m_world_pose;
        Eigen::Vector3f m_velocity;
        Eigen::Vector3f m_accel_bias;
        Eigen::Vector3f m_gyro_bias;

        double m_dt_from_last_kf;

        std::weak_ptr<Frame> m_referance_keyframe;
        Eigen::Matrix4f m_T_relative from_referance;
        static std::shared_ptr<Frame> m_last_keyframe;

        mutable std::mutex m_pose_mutex;
        doiuble m_quality_level = 0.01;
        double min_distance_betwen_features = 30.0;

        std::vector<IMUData> m_imu_vec_from_last_frame;     // IMU data from last frame to current frame;
        std::vector<IMUData> m_imu_vec_since_last_keyframe; // IMU data accumulated since last keyframe;

        std::shared_ptr<IMUPreintegration> m_imu_preintegration_from_last_keyframe;

        std::shared_ptr<IMUPreintegration> m_imu_preintegration_from_last_frame;
    };
