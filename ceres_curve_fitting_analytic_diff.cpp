#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>

// In some cases, using automatic differentiation is not possible.
// For example, it may be the case that it is more efficient to compute the derivatives in closed form
// instead of relying on the chain rule used by the automatic differentiation code.

class curve_factor : public ceres::SizedCostFunction<1, 3> {
public:
    curve_factor(double x, double y) : _x(x), _y(y){}
    virtual ~curve_factor() {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        const double abc[3] = {parameters[0][0], parameters[0][1], parameters[0][2]};
        double y = ceres::exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);

        residuals[0] = y - _y; // notice: estimated value - measurement value

        if(jacobians) {
#if 1
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobi_abc(jacobians[0]);
            jacobi_abc << _x * _x * y, _x * y, y;
#else
            jacobians[0][0] = _x * _x * y;
            jacobians[0][1] = _x * y;
            jacobians[0][2] = y;
#endif
        }

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
//        curve_factor *cost_function = new curve_factor(x_data[i], y_data[i]); // or
        ceres::CostFunction *cost_function = new curve_factor(x_data[i], y_data[i]); // or
        problem.AddResidualBlock(cost_function, NULL, abc);
    }

    // configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    std::clog << "===========Ceres's analytic_diff results is==========" << std::endl;
    for (auto a:abc) std::cout << a << std::endl;


    return 0;
}