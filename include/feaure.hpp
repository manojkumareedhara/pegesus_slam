#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

name_space mono_slam
{
    class Feature
    {
    private:
        int m_feature_id;
        cv::Point2f m_pixel_coord;
        cv::Point3f m_undistorted_coord;
        Eigen::Vector3f m_3d_point;
        Eigen::Vector3f m_normalized_coord;
        int m_total_observations;
        float m_depth;
        float m_reprojection_error;
        bool m_is_valid;
        bool m_has_3d_point;

    public:
        Feature(int feature_id, const cv::Point2f &pixel_coord);
        ~Feature() = default;

        // Getters
        int get_feature_id() const { return m_feature_id; };
        cv::Point2f get_pixel_coord() const { return m_pixel_coord; };
        cv::Point3f get_undistorted_coord() const { return m_undistorted_coord; };
        Eigen::Vector3f get_3d_point() const { return m_3d_point; };
        Eigen::Vector3f get_normalized_point() const { return m_normalized_point; };
        int get_total_observations() const { return m_total_observations; };
        float get_depth() const { return m_depth; };
        float get_reprojection_error() const { return m_reprojection_error; };
        bool is_valid() const { return m_is_valid; };
        bool has_3d_point() const { return m_has_3d_point; };

        // Setters
        void set_pixel_coord(const cv::Point2f &coord) { m_pixel_coord = coord; };
        void set_undistorted_coord(const cv::Point3f &coord) { m_undistorted_coord = coord; };
        void set_3d_point(const Eigen::Vector3f &point)
        {
            m_3d_point = point;
            m_has_3d_point = true;
            m_depth = point.z();
        };
        void set_normalized_point(const Eigen::Vector3f &point) { m_normalized_point = point; };
        void set_total_observations(int count) { m_total_observations = count; };
        void set_depth(float depth)
        {
            m_depth = depth;
        };
        void set_reprojection_error(float reprojection_error) { m_reprojection_error = reprojection_error; };
        void set_valid(bool valid) { m_is_valid = valid; };

        // Operations
        void increment_total_observations() { m_total_observations++; };
    }
}