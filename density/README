Source code:
  density_cpu : single thread cpu density 
  atomic_double : multiple thread cpu density 
  density_no_stream_gpu : gpu without using stream 
  density_has_stream_gpu : gpu with using stream

Compile: 

The GPU files require the CUB library [https://nvlabs.github.io/cub/]

g++ -O3 -std=c++1z ./density_cpu.cpp -o density_cpu 
g++ -fopenmp -O2 -std=c++1z ./atomic_double.cpp -o atomic_double 
/usr/local/cuda-8.0/bin/nvcc --default-stream per-thread -w -arch sm_60 -O2 -I ./cub-1.6.4 density_no_stream_gpu.cu -o density_no_stream_gpu "-std=c++11" -lcusparse 
/usr/local/cuda-8.0/bin/nvcc --default-stream per-thread -w -arch sm_60 -I ./cub-1.6.4 -O2 density_has_stream_gpu.cu -o density_has_stream_gpu "-std=c++11" -lcusparse 

Run:
 ./binary input_file
