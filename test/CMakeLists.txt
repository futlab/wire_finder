cmake_minimum_required(VERSION 3.1)

project(wf-test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
	${OpenCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

link_directories(${OpenCL_LIBRARY} ${OpenCV_LIBRARY})

add_executable(wf-test
    test.cpp
    ../hough.cpp       ../hough.h
    ../hough_test.cpp
    ../cl_utils.cpp    ../cl_utils.h
    ../cl2.hpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCL_LIBRARY}
    ${OpenCV_LIBS}
)

configure_file(../kernel.cl kernel.cl COPYONLY)
configure_file(../hough.cl hough.cl COPYONLY)
