cmake_minimum_required(VERSION 3.5)
project(g1_control VERSION 0.0.1)

## Set compiler to use c++ 17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Chosse build type
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

add_subdirectory(thirdparty)

# 设置源文件路径
set(SOURCE_FILES
    unitree_legged_sdk_bin/g1_control.cpp
)

# 包含头文件目录，将 DDS 指向 thirdparty/include，unitree_sdk2 使用系统路径
#include_directories(
#    ${CMAKE_CURRENT_LIST_DIR}/thirdparty/include    # 包含 DDS 头文件
#    /usr/local/include                              # 系统中已安装的 unitree_sdk2 头文件
#)

# 查找系统架构
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
    set(ARCH amd64)
elseif("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
    set(ARCH arm64)
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 链接系统路径中的 unitree_sdk2 库
find_library(UNITREE_SDK_LIB unitree_sdk2 REQUIRED)
find_package(Threads REQUIRED)


# 设置所需的库，链接 unitree_sdk2 和其他依赖
set(EXTRA_LIBS -pthread ${UNITREE_SDK_LIB} lcm)

# 设置可执行文件的输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 添加目标可执行文件
add_executable(g1_control ${SOURCE_FILES})

# 链接库
target_link_libraries(g1_control
    ${EXTRA_LIBS}
)