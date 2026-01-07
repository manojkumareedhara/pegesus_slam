#include "feaure.hpp"
namespace mono_slam
{

    Feature::Feature(int feature_id, const cv::Point2f &pixel_coord) : m_feature_id(feature_id),
                                                                       m_pixel_coord(pixel_coord),
                                                                       m_undistorted_coord(cv::Point3f(-1, -1, -1)),
                                                                       m_3d_point(Eigen::Vector3f::Zero()),
                                                                       m_normalized_coord(Eigen::Vector3f::Zero()),
                                                                       m_total_observations(1),
                                                                       m_depth(-1.0f),
                                                                       m_reprojection_error(-1.0f),
                                                                       m_is_valid(true),
                                                                       m_has_3d_point(false)
    {
    }
}