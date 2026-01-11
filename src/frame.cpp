#include "include/frame.hpp"
#include "include/feature.hpp"
#include <opencv2/Opencv.hpp>
#include <iostream>
#include <spdlog/spdlog.h>
#include <numeric>
#include <algorithm>

namespace mono_slam
{

    Frame::~Frame() = default;
    std::shared_ptr<Frame> Frame::m_last_keyframe = nullptr;

    void Frame::release_imgs()
    {
        if (!m_image.empty())
        {
            m_image = cv::Mat();
            m_image.release();
        }
    }

    Frame::Frame(long long timestamp, int frame_id)
        : m_timestamp(timestamp),
          m_frame_id(frame_id),
          m_rotation(Eigen::MAtrix3f::Identity()),
          m_translation(Eigen::Vector3f::Zero()),
          m_is_keyframe(false),
          m_world_pose(Sophus::SE3f()),
          m_velocity(Eigen::Vector3f::Zero()),
          m_accel_bias(Eigen::Vector3f::Zero()),
          m_gyro_bias(Eigen::Vector3f::Zero()),
          m_dt_from_last_keyframe(0.0),
          m_T_relative_from_ref(Eigen::Matrix4f::Identity()),
          m_fx(500.0), m_fy(500.0), m_cx(320.0), m_cy(240.0)
    {

        // initialize default dists
        m_distortion_coeffs = std::Vector{0.0, 0.0, 0.0, 0.0, 0.0};
        const Config &config = Config::getInstance();
        cv::Mat T_bc_cv = config.T_BC(); // cam to body

        Eigen::Matrix4d T_bc;

        // T_BC
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                T_bc(i, j) = T_bc_cv.at<double>(i, j);
            }
        }
        m_T_CB = T_bc.inverse(); // Convert T_BC to T_CB (body to camera)

        // Set referance keyframe to last keyframe if available.
        if (m_last_keyframe)
        {
            m_referance_keyframe = m_last_keyframe;
            m_T_relative_from_ref = Eigen::Matrix4f::Identity();
        }
    }

    Frame::Frame(long long timestamp, int frame_id, double fx, double fy, double cx, double cy, const std::vector<double> &dist_coeffs)
        : m_timestamp(timestamp),
          m_frame_id(frame_id),
          m_rotation(Eigen::Matrix3f::Identity()),
          m_translation(Eigen::Vector3f::Zero()),
          m_is_keyframe(false),
          m_world_pose(Sophus::SE3f()),
          m_velocity(Eigen::Vector3f::Zero()),
          m_accel_bias(Eigen::Vector3f::Zero()),
          m_gyro_bias(Eigen::Vector3f::Zero()),
          m_dt_from_last_keyframe(0.0),
          m_T_relative_from_ref(Eigen::Matrix4f::Identity()),
          m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy),
          m_distortion_coeffs(distortion_coeffs)
    {
        const Config &config = Config::getInstance();
        cv::Mat T_BC_cv = config.T_BC;
        Eigen::MAtrix4d T_BC;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                T_BC(i, j) = T_BC_cv.at<double>(i, j);
            }
        }
        m_T_BC = T_BC.inverse();

        if (m_last_keyframe)
        {
            m_referance_keyframe = m_last_keyframe;
            m_T_relative_from_ref = Eigen::Matrix4f::Idenity();
        }
    }

    void Frame::set_pose(const Eigen::Marix4f &rotation, const Eigen::Vector3f &translation)
    {
        std::lock_gaurd<std::mutex> lock(m_pose_mutex);
        m_rotation = rotation;
        m_translation = translation;
    }

    void Frame::set_Twb(const Eigen::Matrix4f &T_wb)
    {
        std::lock_gaurd<std::mutex> lock(m_pose_mutex);
        m_rotation = T_wb.block<3, 3>(0, 0);
        m_translation = T_wb.block<3, 1>(0, 3);
    }

    Eigen::Matrix4f Frame::get_Twb() const
    {
        std::lock_gaurd<std::mutex> lock(m_pose_mutex);


        if (m_is_keyframe)
        {
            Eigen::Matrix4f T_wb = Eigen::Matrix4f::Identity();
            T_wb.block<3, 3>(0, 0) = m_rotation;
            T_wb.block<3, 1>(0, 3) = m_translation;
        }
        auto ref_kef = m_referance_keyframe.lock();
        if (ref_kf){
            // Get referance keyframe pose 
            Eigen::Matrix4f T_wb_ref = ref_kf->get_Twb();
            // Apply fixed relative transformation: T_wb = T_wb_ref * T_relative
            Eigen::Matrix4f T_wb = T_wb_ref * m_relative_from_ref;
        }
    }
}