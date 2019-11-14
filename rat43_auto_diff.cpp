#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>

// URL: http://www.ceres-solver.org/analytical_derivatives.html

class Rat43CostFunctor {
public:
    Rat43CostFunctor(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* abcd, T* residuals) const {
        const T b1 = abcd[0];
        const T b2 = abcd[1];
        const T b3 = abcd[2];
        const T b4 = abcd[3];
        residuals[0] = b1 * ceres::pow(1.0 + exp(b2 -  b3 * x_), -1.0 / b4) - y_;
        return true;
    }

private:
    const double x_, y_;
};

int main(int argc, char** argv) {
    double b1 = 1.0, b2 = 2.0, b3 = 3.0, b4 = 4.0;
    int N = 100;
    double w_sigma = 0.010;
    cv::RNG rng;
    double abcd[4] = {0, 0, 0, 10}; // b4 can't be zero !!!

    // generate data
    std::vector<double> x_data, y_data;
    std::clog << "generating data begin " << std::endl;
    for (int i = 0; i < N; i++) {
        double x = i / 100.;
        double y = b1 *  ceres::pow(1.0 + ceres::exp(b2 - b3 * x), -1.0 / b4) + rng.gaussian(w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }
    std::clog << "generating data done! " << std::endl;

    // create problem
    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
//        curve_factor *cost_function = new curve_factor(x_data[i], y_data[i]); // or
        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<Rat43CostFunctor, 1, 4>(new Rat43CostFunctor(x_data[i], y_data[i])); // or
        problem.AddResidualBlock(cost_function, NULL, abcd);
    }

    // configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    std::clog << "===========Ceres's rat43 auto_diff results is==========" << std::endl;
    for (auto a:abcd) std::cout << a << std::endl;


    return 0;
}