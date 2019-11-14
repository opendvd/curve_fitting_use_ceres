#include <iostream>
#include <ceres/ceres.h>

class factor1 : public ceres::SizedCostFunction<1, 1, 1> {
public:
    virtual ~factor1() {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        double x1 = parameters[0][0];
        double x2 = parameters[0][1];

        residuals[0] = x1 + 10.0 * x2;

        if(jacobians) {
            jacobians[0][0] = 1.0;
            jacobians[0][1] = 10.0;
        }

        return true;
    }
};

class factor2 : public ceres::SizedCostFunction<1, 1, 1> {
public:
    virtual ~factor2() {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        double x3 = parameters[0][0];
        double x4 = parameters[0][1];

        residuals[0] = sqrt(5.0) * (x3 - x4);
        if(jacobians) {
            jacobians[0][0] = sqrt(5);
            jacobians[0][1] = -sqrt(5);
        }
        return true;

    }
};

class factor3 : public ceres::SizedCostFunction<1, 1, 1> {
public:
    virtual ~factor3() {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        double x2 = parameters[0][0];
        double x3 = parameters[0][1];

        residuals[0] = (x2 - 2.0 * x3) * (x2 - 2.0 * x3);
        if(jacobians) {
            jacobians[0][0] = 2.0 * (x2 - 2.0 * x3);
            jacobians[0][1] = -4.0 * (x2 - 2.0 * x3);
        }
        return true;

    }
};

class factor4 : public ceres::SizedCostFunction<1, 1, 1> {
public:
    virtual ~factor4() {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        double x1 = parameters[0][0];
        double x4 = parameters[0][1];

        residuals[0] = sqrt(10.0) * (x1 - x4) * (x1 - x4);
        if(jacobians) {
            jacobians[0][0] = 2.0 * sqrt(10) * (x1 - x4);
            jacobians[0][1] = -2.0 * sqrt(10) * (x1 - x4);
        }
        return true;

    }
};

int main(int argc, char** argv) {

    double x1 = 3., x2 = -1., x3 = 0.0, x4 = 1.0;

    ceres::CostFunction *cost_function1 = new factor1;
    ceres::CostFunction *cost_function2 = new factor2;
    ceres::CostFunction *cost_function3 = new factor3;
    ceres::CostFunction *cost_function4 = new factor4;

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

    std::cout << summary.FullReport() << std::endl;
    std::clog << "===========Ceres's powell auto_diff results is==========" << std::endl;
    std::cout << "x1 = " << x1 << " x2 = " << x2 << " x3 = " << x3 << " x4 = " << x4 << std::endl;

    return 0;
}