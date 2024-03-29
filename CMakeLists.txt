cmake_minimum_required(VERSION 2.8)
project(ceres_curve_fitting)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O0")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
message(STATUS "Ceres localization is: ${CERES_INCLUDE_DIRS}")

set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

add_executable(ceres_curve_fitting_auto_diff ceres_curve_fitting_auto_diff.cpp)
target_link_libraries(ceres_curve_fitting_auto_diff ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(ceres_curve_fitting_numeric_diff ceres_curve_fitting_numeric_diff.cpp)
target_link_libraries(ceres_curve_fitting_numeric_diff ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(ceres_curve_fitting_analytic_diff ceres_curve_fitting_analytic_diff.cpp)
target_link_libraries(ceres_curve_fitting_analytic_diff ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(ceres_curve_fitting_loss_function ceres_curve_fitting_loss_function.cpp)
target_link_libraries(ceres_curve_fitting_loss_function ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(powell_auto_diff powell_auto_diff.cpp)
target_link_libraries(powell_auto_diff ${CERES_LIBRARIES})

add_executable(powell_analytic_diff powell_analytic_diff.cpp)
target_link_libraries(powell_analytic_diff ${CERES_LIBRARIES})

add_executable(rat43_analytic_diff rat43_analytic_diff.cpp)
target_link_libraries(rat43_analytic_diff ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(rat43_auto_diff rat43_auto_diff.cpp)
target_link_libraries(rat43_auto_diff ${CERES_LIBRARIES} ${OpenCV_LIBS})