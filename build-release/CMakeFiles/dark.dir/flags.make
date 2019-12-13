# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# compile C with /usr/bin/cc
# compile CUDA with /usr/local/cuda/bin/nvcc
# compile CXX with /usr/bin/c++
C_FLAGS = -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings  -fPIC   -fopenmp

C_DEFINES = -DCUDNN -DGPU -DLIB_EXPORTS=1 -DOPENCV -DUSE_CMAKE_LIBS -Ddark_EXPORTS

C_INCLUDES = -I"/content/drive/My Drive/Colab Notebooks/darknet/include" -I"/content/drive/My Drive/Colab Notebooks/darknet/src" -I"/content/drive/My Drive/Colab Notebooks/darknet/3rdparty/stb/include" -I/usr/local/cuda/targets/x86_64-linux/include -isystem /usr/include/opencv 

CUDA_FLAGS = -gencode arch=compute_60,code=sm_60 --compiler-options " -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings -DGPU -DCUDNN -DOPENCV -fPIC -fopenmp -Ofast "  -Xcompiler=-fPIC  

CUDA_DEFINES = -DCUDNN -DGPU -DLIB_EXPORTS=1 -DOPENCV -DUSE_CMAKE_LIBS -Ddark_EXPORTS

CUDA_INCLUDES = -I"/content/drive/My Drive/Colab Notebooks/darknet/include" -I"/content/drive/My Drive/Colab Notebooks/darknet/src" -I"/content/drive/My Drive/Colab Notebooks/darknet/3rdparty/stb/include" -I/usr/local/cuda/targets/x86_64-linux/include -isystem=/usr/include/opencv 

CXX_FLAGS = -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings  -fPIC -fvisibility=hidden   -fopenmp -std=gnu++11

CXX_DEFINES = -DCUDNN -DGPU -DLIB_EXPORTS=1 -DOPENCV -DUSE_CMAKE_LIBS -Ddark_EXPORTS

CXX_INCLUDES = -I"/content/drive/My Drive/Colab Notebooks/darknet/include" -I"/content/drive/My Drive/Colab Notebooks/darknet/src" -I"/content/drive/My Drive/Colab Notebooks/darknet/3rdparty/stb/include" -I/usr/local/cuda/targets/x86_64-linux/include -isystem /usr/include/opencv 

