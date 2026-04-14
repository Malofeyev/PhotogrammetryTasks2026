#include <phg/sfm/defines.h>
#include "calibration.h"

#include <ceres/ceres.h>

class CostFunctor {
public:

    CostFunctor(double k1, double k2, double px, double py) 
        : k1(k1), k2(k2), px(px), py(py) {} 
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        T r2 = x[0] * x[0] + x[1] * x[1];
        T radial = (T)1.0 + (T)k1 * r2 + (T)k2 * r2 * r2;
        residual[0] = x[0] - (T)px / radial;
        residual[1] = x[1] - (T)py / radial;
        return true;
    }
private:
    double k1, k2, px, py;
};
phg::Calibration::Calibration(int width, int height)
    : width_(width)
    , height_(height)
    , cx_(0)
    , cy_(0)
    , k1_(0)
    , k2_(0)
{
    // 50mm guess

    double diag_35mm = 36.0 * 36.0 + 24.0 * 24.0;
    double diag_pix = (double) width * (double) width + (double) height * (double) height;

    f_ = 50.0 * std::sqrt(diag_pix / diag_35mm);
}

cv::Matx33d phg::Calibration::K() const {
    return {f_, 0., cx_ + width_ * 0.5, 0., f_, cy_ + height_ * 0.5, 0., 0., 1.};
}

int phg::Calibration::width() const {
    return width_;
}

int phg::Calibration::height() const {
    return height_;
}

cv::Vec3d phg::Calibration::project(const cv::Vec3d &point) const
{
    double x = point[0] / point[2];
    double y = point[1] / point[2];

    // TODO 11: добавьте учет радиальных искажений (k1_, k2_) (после деления на Z, но до умножения на f)
    double r2 = x * x + y * y;
    double radial = 1.0 + k1_ * r2 + k2_ * r2 * r2;

    x *= radial;
    y *= radial;

    x *= f_;
    y *= f_;

    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    double x = pixel[0] - cx_ - width_ * 0.5;
    double y = pixel[1] - cy_ - height_ * 0.5;

    x /= f_;
    y /= f_;

    // TODO 12: добавьте учет радиальных искажений, когда реализуете - подумайте: почему строго говоря это - не симметричная формула формуле из project? (но лишь приближение)
    // потому что при y /= radial значение y, который ищем, входит в radial
    /*double distorted_x = x;
    double distorted_y = y;
    for (int i = 0; i < 10; ++i) {
        double r2 = x * x + y * y;
        double rad_distortion = 1. + k1_ * r2 + k2_ * r2 * r2;
        x = distorted_x / rad_distortion;
        y = distorted_y / rad_distortion
    }*/
    CostFunctor *f = new CostFunctor(k1_, k2_, x, y);
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 2, 2>(f);
    ceres::LossFunction* loss_function = new ceres::TrivialLoss();
    double cur_point[] = {x, y}; 

    ceres::Problem problem;
    problem.AddResidualBlock(cost_function, loss_function, cur_point);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // Почему Conjugate gradients не срабатывают?
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    
    return cv::Vec3d(cur_point[0], cur_point[1], 1.0);
}
