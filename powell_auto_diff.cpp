#include <iostream>
#include <ceres/ceres.h>

class factor1 {
public:
    template <typename T> bool operator()(const T* const x1, const T* const x2, T* residual) const {
        // f1 = x1 + 10 * x2;
        residual[0] = x1[0] + T(10.0) * x2[0];
        return true;
    }
};

class factor2 {
public:
    template <typename T> bool operator()(const T* const x3, const T* const x4, T* residual) const {
        // f2 = sqrt(5) (x3 - x4)
        residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
        return true;
    }
};

class factor3 {
public:
    template <typename T> bool operator()(const T* const x2, const T* const x3, T* residual) const {
        // f3 = (x2 - 2 x3)^2
        residual[0] = (x2[0] - T(2.0) * x3[0]) * (x2[0] - T(2.0) * x3[0]);
        return true;
    }
};

class factor4 {
public:
    template <typename T> bool operator()(const T* const x1, const T* const x4, T* residual) const {
        // f4 = sqrt(10) (x1 - x4)^2
        residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
        return true;
    }
};

int main(int argc, char** argv) {

    double x1 = 3., x2 = -1., x3 = 0.0, x4 = 1.0;

    ceres::CostFunction *cost_function1 = new ceres::AutoDiffCostFunction<factor1, 1, 1, 1>(new factor1);
    ceres::CostFunction *cost_function2 = new ceres::AutoDiffCostFunction<factor2, 1, 1, 1>(new factor2);
    ceres::CostFunction *cost_function3 = new ceres::AutoDiffCostFunction<factor3, 1, 1, 1>(new factor3);
    ceres::CostFunction *cost_function4 = new ceres::AutoDiffCostFunction<factor4, 1, 1, 1>(new factor4);

    ceres::Problem problem;
    problem.AddResidualBlock(cost_function1, NULL, &x1, &x2);
    problem.AddResidualBlock(cost_function2, NULL, &x3, &x4);
    problem.AddResidualBlock(cost_function3, NULL, &x2, &x3);
    problem.AddResidualBlock(cost_function4, NULL, &x1, &x4);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    std::clog << "===========Ceres's powell auto_diff results is==========" << std::endl;
    std::cout << "x1 = " << x1 << " x2 = " << x2 << " x3 = " << x3 << " x4 = " << x4 << std::endl;

    return 0;
}