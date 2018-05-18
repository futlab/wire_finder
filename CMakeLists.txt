cmake_minimum_required(VERSION 3.1)

project(cltest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

find_package(ZED 2 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(${PROJECT_NAME}
    main.cpp
    cl_utils.cpp cl_utils.h
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCL_LIBRARY}
)

configure_file(kernel.cl kernel.cl COPYONLY)
