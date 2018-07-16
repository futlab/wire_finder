cmake_minimum_required(VERSION 3.1)

project(wf-test)

option(WITH_GUI "With GUI" OFF)
option(KERNEL_FILL "Manual fill with kernel" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

if(${WITH_GUI})
  add_definitions(-DWITH_GUI)
endif()

if(${KERNEL_FILL})
  add_definitions(-DKERNEL_FILL)
endif()

include_directories(
    ${GTEST_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)

link_directories(${OpenCL_LIBRARY} ${OpenCV_LIBRARY})

add_executable(wf-test
    test.cpp		   
    gtest_utils.h
    cl_utils_test.cpp       ../cl_utils.cpp         ../cl_utils.h
    cameramodel_test.cpp    ../cameramodel.h
    kalmanline_test.cpp     ../kalmanline.h
    hough_test.cpp          ../hough.h          ../hough.cpp
    ../linepool_test.cpp    ../linepool.h
    ../cl2.hpp
    ../defs.h
)

target_link_libraries(${PROJECT_NAME}
    ${GTEST_LIBRARIES}
    ${OpenCL_LIBRARY}
    ${OpenCV_LIBS}
)

if (UNIX)
    target_link_libraries(${PROJECT_NAME}
        pthread
    )
endif()

configure_file(../kernel.cl kernel.cl COPYONLY)
configure_file(../hough.cl hough.cl COPYONLY)
