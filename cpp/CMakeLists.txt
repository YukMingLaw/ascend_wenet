cmake_minimum_required(VERSION 3.5)

project(WENET_ASCEND)


add_definitions(-std=c++11)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
add_definitions(-DENABLE_DVPP_INTERFACE)

set(ACL_PATH /usr/local/Ascend/ascend-toolkit/latest)
INCLUDE_DIRECTORIES(
        ./
        ${ACL_PATH}/acllib/include/
        /usr/local/kenlm/include/kenlm
        /usr/local/fst/include/
    )


set(ACL_LIB_PATH ${ACL_PATH}/acllib/lib64/stub)
link_directories(${ACL_LIB_PATH})

set(KENLM_LIB_PATH /usr/local/kenlm/lib)
link_directories(${KENLM_LIB_PATH})

set(FST_LIB_PATH /usr/local/fst/lib)
link_directories(${FST_LIB_PATH})

file(GLOB_RECURSE SRC_DIR_LIST "swig_decoders/*.cpp" "common/*.cpp" "feature_extrator/*.cc" "encoder/*.cpp" "decoder/*.cpp")

add_executable(${PROJECT_NAME}
        main.cpp
        wenet.cpp
        ${SRC_DIR_LIST}
        )

target_link_libraries(${PROJECT_NAME}
        ascendcl
        pthread
        kenlm
        fst
        -Wl,-z,relro,-z,now,-z,noexecstack -pie -s
        stdc++)
