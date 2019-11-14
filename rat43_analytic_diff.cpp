#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>

// URL: http://www.ceres-solver.org/analytical_derivatives.html

class rat43_factor : public ceres::SizedCostFunction<1, 4> {
public:
    rat43_factor(double x, double y) : x_(x), y_(y) {}
    virtual ~rat43_factor() {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        const double b1 = parameters[0][0];
        const double b2 = parameters[0][1];
        const double b3 = parameters[0][2];
        const double b4 = parameters[0][3];

        const double t1 = exp(b2 -  b3 * x_);
        const double t2 = 1 + t1;
        const double t3 = pow(t2, -1.0 / b4);
        residuals[0] = b1 * t3 - y_;

        if(jacobians) {

            const double t4 = pow(t2, -1.0 / b4 - 1);
#if 1
            Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> jacobi_abc(jacobians[0]);
            const double j0 = t3;
            const double j1 = -b1 * t1 * t4 / b4;
            const double j2 = -x_ * j1;
            const double j3 = b1 * log(t2) * t3 / (b4 * b4);
            jacobi_abc << j0, j1, j2, j3;

//            jacobi_abc << t3, -b1 * t1 * t4 / b4,
//                          -x_ * jacobi_abc[1], // this is error, can't use jacobi_abc[1] before finish.
//                           b1 * log(t2) * t3 / (b4 * b4);

#else
            jacobians[0][0] = t3;
            jacobians[0][1] = -b1 * t1 * t4 / b4;
            jacobians[0][2] = -x_ * jacobians[0][1];
            jacobians[0][3] = b1 * log(t2) * t3 / (b4 * b4);

#endif
        }

        return true;
    }

private:
    const double x_, y_;
};

int main(int argc, char** argv) {
    double b1 = 1.0, b2 = 2.0, b3 = 3.0, b4 = 4.0;
    int N = 100;
    double w_sigma = 0.01;
    cv::RNG rng;
    double abcd[4] = {0, 0, 0, 10}; // b4 can't be zero !!!

    // generate data
    std::vector<double> x_data, y_data;
    std::clog << "generating data begin " << std::endl;
    for (int i = 0; i < N; i++) {
        double x = i / 100.;
        double y = b1 *  pow(1.0 + exp(b2 - b3 * x), -1.0 / b4) + rng.gaussian(w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }
    std::clog << "generating data done! " << std::endl;

    // create problem
    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
//        curve_factor *cost_function = new curve_factor(x_data[i], y_data[i]); // or
        ceres::CostFunction *cost_function = new rat43_factor(x_data[i], y_data[i]); // or
        problem.AddResidualBlock(cost_function, NULL, abcd);
    }

    // configure solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    std::clog << "===========Ceres's rat43 analytic_diff results is==========" << std::endl;
    for (auto a:abcd) std::cout << a << std::endl;


    return 0;
}