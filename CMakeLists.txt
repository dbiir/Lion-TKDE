cmake_minimum_required(VERSION 3.2)
project(star)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -O0  -g -ggdb3 -pthread -Wall -Wno-long-long -Wno-unused-variable -Wno-address-of-packed-member -Wno-variadic-macros -pedantic -std=c++14 -Wno-unknown-pragmas") #

set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_EXE_LINKER_FLAGS "-static")
# set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

# additional target to perform clang-format run, requires clang-format

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)

# get all project files
file(GLOB_RECURSE ALL_SOURCE_FILES benchmark/*.h common/*.h core/*.h core/*.cpp protocol/*.h bench*.cpp)

include_directories(${CMAKE_SOURCE_DIR})

file(GLOB_RECURSE COMMON_SOURCE_FILES common/*.cpp benchmark/**/*.cpp)
add_library(common STATIC ${COMMON_SOURCE_FILES} ${ALL_SOURCE_FILES})


# if(APPLE)
#     find_package(glog REQUIRED)
#     find_package(gflags REQUIRED)
#     target_link_libraries(common ${jemalloc_lib} glog::glog gflags)# absl::flat_hash_map)
# else()
    # find_package(glog REQUIRED)
    # set(gflags_NAMES libgflags.a)
    # find_package(gflags REQUIRED)
    find_library(GLOG_LIBRARIES glog)
    find_library(GFLAGS_LIBRARIES gflags)
    find_library(GMOCK_LIBRARIES gmock)
    find_library(GTEST_LIBRARIES gtest)
    find_library(METIS_LIBRARIES metis)
    find_library(TFlowC_LIBRARIES tensorflow HINTS /usr/local/lib)
    find_library(TFlowC_FRAMEWORK_LIBRARIES tensorflow_framework HINTS /usr/local/lib)


    target_link_libraries(common ${jemalloc_lib} ${glog} ${gflags} 
    ${TFlowC_LIBRARIES} ${TFlowC_FRAMEWORK_LIBRARIES}
    ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES}) 
    # target_link_libraries(common libgflags.a)
    # --[ Eigen3
    find_package(Eigen3 REQUIRED)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# endif()


# add_executable(bench_tpcc bench_tpcc.cpp)
# target_link_libraries(bench_tpcc common)

add_executable(bench_ycsb bench_ycsb.cpp)
target_link_libraries(bench_ycsb common
                    ${jemalloc_lib} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES}
                    ${TFlowC_LIBRARIES} ${TFlowC_FRAMEWORK_LIBRARIES}
                    ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES}
                    ${METIS_LIBRARIES}) # -static

add_executable(bench_pps bench_pps.cpp)
target_link_libraries(bench_pps common
                    ${jemalloc_lib} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES}
                    ${TFlowC_LIBRARIES} ${TFlowC_FRAMEWORK_LIBRARIES}
                    ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES}
                    ${METIS_LIBRARIES}) # -static


add_executable(lion bench_ycsb.cpp)
target_link_libraries(lion common 
                    ${jemalloc_lib} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} 
                    ${TFlowC_LIBRARIES} ${TFlowC_FRAMEWORK_LIBRARIES}
                    ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES} 
                    ${METIS_LIBRARIES}) # -static

add_executable(lion_metis metis_test.cpp)
target_link_libraries(lion_metis common 
                    ${jemalloc_lib} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} 
                    ${TFlowC_LIBRARIES} ${TFlowC_FRAMEWORK_LIBRARIES}
                    ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES} 
                    ${METIS_LIBRARIES}) # -static

# add_executable(bench_tpcc bench_tpcc.cpp)
# target_link_libraries(bench_tpcc common 
#                     ${jemalloc_lib} ${glog} ${gflags} 
#                     ${TFlowC_LIBRARIES} ${TFlowC_FRAMEWORK_LIBRARIES}
#                     ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES} 
#                     ${METIS_LIBRARIES}) # -static




# add_executable(bench_metis metis_test.cpp)
# target_link_libraries(bench_metis common 
#                     ${jemalloc_lib} ${glog} ${gflags} 
#                     ${TFlowC_LIBRARIES} ${TFlowC_FRAMEWORK_LIBRARIES}
#                     ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES} 
#                     ${METIS_LIBRARIES}) # -static

# # 
add_subdirectory(test)
