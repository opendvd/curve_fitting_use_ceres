#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>

class curve_factor {
public:
    curve_factor(double x, double y) : _x(x), _y(y){}

    template <typename T>
    bool operator() (const T* const abc, T* residual) const {
        residual[0] = T(_y)  - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x, _y;
};

int main(int argc, char** argv) {
    double a = 1.0, b = 2.0, c = 3.0;
    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;
    double abc[3] = {0, 0, 0};

    // generate data
    std::vector<double> x_data, y_data;
    std::clog << "generating data begin " << std::endl;
    for (int i = 0; i < N; i++) {
        double x = i / 100.;
        double y = exp(a*x*x + b*x + c) + rng.gaussian(w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }
    std::clog << "generating data done! " << std::endl;

    // create problem
    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<curve_factor, 1, 3>(new curve_factor(x_data[i], y_data[i]));
        problem.AddResidualBlock(cost_function, NULL, abc);
    }

    // configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    std::clog << "===========Ceres's auto_diff results is==========" << std::endl;
    for (auto a:abc) std::cout << a << std::endl;


    return 0;
}