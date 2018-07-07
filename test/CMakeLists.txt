cmake_minimum_required(VERSION 3.1)

project(wf-test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(gtest REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

include_directories(
	${GTEST_INCLUDE_DIR}
	${OpenCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)

link_directories(${OpenCL_LIBRARY} ${OpenCV_LIBRARY})

add_executable(wf-test
    test.cpp
    ../hough.cpp       ../hough.h
    ../hough_test.cpp
	../cameramodel.h
	../linepool_test.cpp ../linepool.h
	../kalmanline.h		kalmanline_test.cpp
    ../cl_utils.cpp    ../cl_utils.h
    ../cl2.hpp
	../defs.h
)

target_link_libraries(${PROJECT_NAME}
	${GTEST_LIBRARIES}
    ${OpenCL_LIBRARY}
    ${OpenCV_LIBS}
	Eigen3::Eigen
)

configure_file(../kernel.cl kernel.cl COPYONLY)
configure_file(../hough.cl hough.cl COPYONLY)
