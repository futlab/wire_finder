cmake_minimum_required(VERSION 3.1)

project(cltest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

find_package(ZED 2 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

if(NOT DEFINED ${Eigen3_INCLUDE_DIRS})
    find_package(Eigen3 REQUIRED)
endif()

include_directories(
	${OpenCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
        ${ZED_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
)

link_directories(${OpenCL_LIBRARY} ${OpenCV_LIBRARY})

add_executable(${PROJECT_NAME}
    main.cpp		defs.h
    hough.cpp           hough.h
    cmplines.cpp	cmplines.h
    linepool.cpp
	linepool.h
	cameramodel.h
    hough_test.cpp
    cl_utils0.cpp   cl_utils0.h
    cl_utils.cpp    cl_utils.h
    cl2.hpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCL_LIBRARY}
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
)

add_executable(info
    info.cpp
    cl_utils.cpp    cl_utils.h
)

target_link_libraries(info
    ${OpenCL_LIBRARY}
    ${OpenCV_LIBS}
)

configure_file(kernel.cl kernel.cl COPYONLY)
configure_file(hough.cl hough.cl COPYONLY)
configure_file(cmplines.cl cmplines.cl COPYONLY)
