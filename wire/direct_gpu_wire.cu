#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <random>
#include <cmath>
#include <cassert>
#include <iomanip>  
#include <sstream>

#include <cuda_runtime.h>
#include "cusparse.h"


#define Gamma 100.0

using namespace std;

#define D2H cudaMemcpyDeviceToHost 
#define H2D cudaMemcpyHostToDevice 


#define cudaAlloc(ptr,sz,type) do{ \
  cudaMalloc((void **)&ptr, sz*sizeof(type)); \
}while(0)



char* testcase_name[] = {
  "./exp/mgc_superblue16_a", // 0
  "./exp/mgc_superblue11_a", // 1
  "./exp/mgc_superblue12",   // 2
  "./exp/mgc_des_perf_1",    // 3
  "./exp/mgc_des_perf_a",    // 4
  "./exp/mgc_edit_dist_a",   // 5
  "./exp/mgc_edit_dist_2",   // 6
  "./exp/mgc_matrix_mult_1", // 7
  "./exp/mgc_pci_bridge32_a",// 8
  "./exp/mgc_pci_bridge32_b",// 9
  "./exp/mgc_fft_1",         // 10
  "./exp/mgc_fft_2",         // 11
  "./exp/mgc_matrix_mult_a", // 12
  "./exp/mgc_matrix_mult_b", // 13
  "./exp/mgc_des_perf_b",    // 14
  "./exp/mgc_fft_a",         // 15
  "./exp/mgc_fft_b",         // 16
};


int ID = 0;


void checkCUDAerror(){
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}





__global__
void pin_exp_sum(double *pin , size_t num , double *out, bool neg )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < num )
    if( !neg )
      out[i] = exp( pin[i]/Gamma );
    else
      out[i] = exp( -pin[i]/Gamma );
}







__global__
void exp_sum(double *pin_x , double *pin_y , size_t num , double *out )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < num ){

   // out[4*i]   = exp( pin_x[i]/Gamma );
   // out[4*i+1] = exp( -pin_x[i]/Gamma );
   // out[4*i+2] = exp( pin_y[i]/Gamma );
   // out[4*i+3] = exp( -pin_y[i]/Gamma );

    out[i]   = exp( pin_x[i]/Gamma );
    out[i+num] = exp( -pin_x[i]/Gamma );
    out[i+2*num] = exp( pin_y[i]/Gamma );
    out[i+3*num] = exp( -pin_y[i]/Gamma );
  }
}

void compute_exp( size_t pin_size, double *pin_x, double *pin_y, double* pin_exp_d ){
  dim3 DimGrid(pin_size/256+1,1,1);
  dim3 DimBlock(256,1,1);
  exp_sum<<<DimGrid,DimBlock>>>( pin_x, pin_y, pin_size, pin_exp_d);
  checkCUDAerror();
  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()
}

__global__
void reciprocal_kernel(double *mat , size_t sz )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    if( mat[i] != 0.0 )
      mat[i] = 1.0/mat[i];
  }
}


void gpu_reciprocal( double *mat , size_t sz ){
  dim3 DimGrid(sz/256+1,1,1);
  dim3 DimBlock(256,1,1);
  reciprocal_kernel<<<DimGrid,DimBlock>>>(mat , sz);
  checkCUDAerror();
}


__global__
void neg_odd_col_kernel(double *mat , size_t pin_num )
{
  auto i =  blockIdx.x*blockDim.x + threadIdx.x;
  if( i < pin_num ){
    mat[i+pin_num] = -mat[i+pin_num];
    mat[i+3*pin_num] = -mat[i+3*pin_num];
  }

  //auto i = blockIdx.y * col + blockIdx.x*blockDim.x + threadIdx.x;
  //if( blockIdx.x*blockDim.x + threadIdx.x < col )
  //  mat[i] = -mat[i];
}



void gpu_neg_odd_row( double *mat, size_t pin_num ){
  dim3 DimGrid( (pin_num/256+1),1,1);
  dim3 DimBlock(256,1,1);
  neg_odd_col_kernel<<<DimGrid,DimBlock>>>(mat , pin_num);
  checkCUDAerror();
  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()
}



//__global__ void Dev_dot(double x[], double y[], double z[], int n) {
//   /* Use tmp to store products of vector components in each block */
//   /* Can't use variable dimension here                            */
//   __shared__ double tmp[MAX_BLOCK_SZ];
//   int t = blockDim.x * blockIdx.x + threadIdx.x;
//   int loc_t = threadIdx.x;
//   
//   if (t < n) tmp[loc_t] = x[t]*y[t];
//   __syncthreads();
//
//   /* This uses a tree structure to do the additions */
//   for (int stride = blockDim.x/2; stride >  0; stride /= 2) {
//      if (loc_t < stride)
//         tmp[loc_t] += tmp[loc_t + stride];
//      __syncthreads();
//   }
//
//   /* Store the result from this cache block in z[blockIdx.x] */
//   if (threadIdx.x == 0) {
//      z[blockIdx.x] = tmp[0];
//   }
//}  /* Dev_dot */    






__global__
void matrix_dot_product_kernel(double* A, double* B, double* C, size_t sz )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    //C[i] =  A[i*4]  *B[i*4];
    //C[i] += A[i*4+1]*B[i*4+1];
    //C[i] += A[i*4+2]*B[i*4+2];
    //C[i] += A[i*4+3]*B[i*4+3];

    C[i]  = A[i]  *B[i];
    C[i] += A[i+1*sz]*B[i+1*sz];
    C[i] += A[i+2*sz]*B[i+2*sz];
    C[i] += A[i+3*sz]*B[i+3*sz];
  }
}
// B is transpose
void gpu_matrix_dot_product( double *A , double *B, double *C, size_t pin_size ){
  dim3 DimGrid( (pin_size/256+1),1,1);
  dim3 DimBlock(256,1,1);
  matrix_dot_product_kernel<<<DimGrid,DimBlock>>>(A,B,C,pin_size);
  checkCUDAerror();
}



void read_pin_location( vector<double> &pin_x , vector<double> &pin_y ){
  int pin_num;
  double x,y,w,h;
  ifstream fptr;
  vector<double> x_v;
  vector<double> y_v;
  string tname(testcase_name[ID]);
  tname += "_gpu_density_info";
  cout << "GPU density Info = " << tname << '\n';

  double Gx,Gy;

  //fptr.open("./tune_placer/gpu_density_info");
  fptr.open(tname.c_str());

  fptr >> pin_num >> Gx >> Gy;
  for( size_t i = 0 ; i < pin_num ; ++ i ){
    fptr >> x >> y >> w >> h;

    x_v.emplace_back(x);
    y_v.emplace_back(y);
  }
  std::move(x_v.begin(), x_v.end(), std::back_inserter(pin_x)); 
  std::move(y_v.begin(), y_v.end(), std::back_inserter(pin_y)); 
  fptr.close();
}




__global__
void compute_wire_exp_kernel(int* start, int* end, int* pinInWire,
                             double* wire_x, double* wire_y, 
                             double* wire_x_neg, double* wire_y_neg,
                             double *pin_x, double *pin_y, double* pin_x_neg, double* pin_y_neg, size_t num )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < num ){
    for( auto id = start[i] ; id < end[i] ; ++id ){
      if( id == start[i] ){
        wire_x[i] = pin_x[pinInWire[id]];
        wire_y[i] = pin_y[pinInWire[id]];
        wire_x_neg[i] = pin_x_neg[pinInWire[id]];
        wire_y_neg[i] = pin_y_neg[pinInWire[id]];
      }
      else{
        wire_x[i] += pin_x[pinInWire[id]];
        wire_y[i] += pin_y[pinInWire[id]];
        wire_x_neg[i] += pin_x_neg[pinInWire[id]];
        wire_y_neg[i] += pin_y_neg[pinInWire[id]];
      }
    }
  }
}

void compute_wire_exp( size_t wire_size, double *pin_x, double *pin_y, double* pin_x_neg, double* pin_y_neg,
                       int *start, int *end,
                       double *wire_x, double *wire_y, double *wire_x_neg, double *wire_y_neg, int *pinInWire ){
  dim3 DimGrid(wire_size/1024+1,1,1);
  dim3 DimBlock(1024,1,1);
  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);

  cudaEventRecord(start_t);

  compute_wire_exp_kernel<<<DimGrid,DimBlock>>>( start, end, pinInWire,
                                                 wire_x, wire_y, wire_x_neg, wire_y_neg, pin_x, pin_y, pin_x_neg, pin_y_neg, wire_size );
  cudaEventRecord(stop_t);

  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()

  cudaEventSynchronize(stop_t);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_t, stop_t);

  std::cout << "Compute wire exp :" << milliseconds << " milli sec" << std::endl;

  checkCUDAerror();
}





__global__
void compute_pin_exp_kernel(double* x, double* y, double *pin_x, double *pin_y, double* pin_x_neg, double* pin_y_neg, size_t num )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < num ){
    pin_x[i] = exp( x[i]/Gamma );
    pin_y[i] = exp( y[i]/Gamma );
    pin_x_neg[i] = exp( -x[i]/Gamma );
    pin_y_neg[i] = exp( -y[i]/Gamma );
  }
}

void compute_pin_exp( size_t pin_size, double* x, double* y, double *pin_x, double *pin_y, double* pin_x_neg, double* pin_y_neg ){
  dim3 DimGrid(pin_size/1024+1,1,1);
  dim3 DimBlock(1024,1,1);

  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);

  cudaEventRecord(start_t);

  compute_pin_exp_kernel<<<DimGrid,DimBlock>>>( x, y, pin_x, pin_y, pin_x_neg, pin_y_neg, pin_size );

  cudaEventRecord(stop_t);

  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()
  cudaEventSynchronize(stop_t);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_t, stop_t);

  std::cout << "Compute pin exp :" << milliseconds << " milli sec" << std::endl;
  checkCUDAerror();
}


__global__
void compute_grad_kernel(int* start, int *end, int *pinInWire, 
                         double* pin_x,  double* pin_y,  double* pin_x_neg,  double *pin_y_neg,
                         double *wire_x, double *wire_y, double* wire_x_neg, double* wire_y_neg, int num, double *x, double *y )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < num ){
    double grad_x = 0.0;
    double grad_y = 0.0;
    double grad_x_neg = 0.0;
    double grad_y_neg = 0.0;
    for( auto id = start[i] ; id < end[i] ; ++id ){
      grad_x += 1.0/wire_x[pinInWire[id]];
      grad_y += 1.0/wire_y[pinInWire[id]];
      grad_x_neg -= 1.0/wire_x_neg[pinInWire[id]];
      grad_y_neg -= 1.0/wire_y_neg[pinInWire[id]];
    }

    x[i] = grad_x*pin_x[i] + grad_x_neg*pin_x_neg[i];
    y[i] = grad_y*pin_y[i] + grad_y_neg*pin_y_neg[i];
    //if( i < 10 )
    //  printf("%lf , %lf , %lf , %lf,  %.12e\n", pin_y[i], pin_y_neg[i], grad_y, grad_y_neg, y[i] );


    //x[i] = grad_x*pin_x[i] - grad_x_neg*pin_x_neg[i] +
    //       grad_y*pin_y[i] - grad_y_neg*pin_y_neg[i] ;
    //y[i] = 1.0/grad_y*pin_y[i] - 1.0/grad_y_neg*pin_y_neg[i];
  }
}





int main(int argc, char *argv[] ){
  if(argc < 2 ){
    printf("Error : No testcase ID\n");
    exit(1);
  }
  istringstream ss(argv[1]);
  ss >> ID;
  printf("Argv[1] (ID) = %d\n" , ID);



  string tname(testcase_name[ID]);
  tname += "_gpu_wire_info";

  ofstream of;
  ifstream fptr;
  fptr.open(tname.c_str());

  vector<vector<int>> wire;
  vector<double> pin_x;
  vector<double> pin_y;
  size_t wire_num;
  size_t pin_num;
  fptr >> wire_num >> pin_num;
  wire.resize(wire_num);
  
  size_t pin_sum = 0;
  for( size_t i = 0 ; i < wire_num ; ++ i ){
    size_t id;
    double x;
    fptr >> id;
    wire[i].resize(id);
    pin_sum += id;
    for( size_t j = 0 ; j < wire[i].size() ; ++ j ){
      fptr >> wire[i][j] >> x;
    }
  }
  fptr.close();

  read_pin_location( pin_x, pin_y );

  int *start = (int*)malloc(sizeof(int)*wire.size() );
  int *end = (int*)malloc(sizeof(int)*wire.size() );
  int *pinInWire = (int*)malloc(sizeof(int)*pin_sum);
  int acc = 0;
  pin_sum = 0;

  for( size_t i = 0 ; i < wire.size() ; ++ i ){
    start[i] = acc;
    end[i] = acc + wire[i].size();
    acc += wire[i].size();
    for( size_t j = 0 ; j < wire[i].size() ; ++ j ){
      pinInWire[pin_sum] = wire[i][j];
      pin_sum += 1;
    }
  }



  auto begin_t = std::chrono::high_resolution_clock::now();
  auto end_t = std::chrono::high_resolution_clock::now();



  double *pin_x_d;
  double *pin_y_d;
  double *pin_exp_x_d;
  double *pin_exp_y_d;
  double *pin_exp_x_neg_d;
  double *pin_exp_y_neg_d;
  cudaMalloc((void **)&pin_x_d, pin_x.size() * sizeof(double) );
  cudaMalloc((void **)&pin_y_d, pin_x.size() * sizeof(double) );
  cudaMalloc((void **)&pin_exp_x_d, pin_x.size() * sizeof(double) );
  cudaMalloc((void **)&pin_exp_y_d, pin_x.size() * sizeof(double) );
  cudaMalloc((void **)&pin_exp_x_neg_d, pin_x.size() * sizeof(double) );
  cudaMalloc((void **)&pin_exp_y_neg_d, pin_x.size() * sizeof(double) );


  // Copy pin coordinates to GPU and compute exponential values
  begin_t = std::chrono::high_resolution_clock::now();
  cudaMemcpy(pin_x_d, pin_x.data(), pin_x.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(pin_y_d, pin_y.data(), pin_x.size() * sizeof(double), cudaMemcpyHostToDevice);
  compute_pin_exp( pin_x.size(), pin_x_d, pin_y_d, pin_exp_x_d, pin_exp_y_d, pin_exp_x_neg_d, pin_exp_y_neg_d );
  end_t = std::chrono::high_resolution_clock::now();


  auto total_run_time = std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();


  int *start_d;
  int *end_d;
  int *pinInWire_d;
  cudaMalloc((void **)&start_d, wire.size() * sizeof(int));
  cudaMalloc((void **)&end_d, wire.size() * sizeof(int));
  cudaMalloc((void **)&pinInWire_d, pin_sum* sizeof(int));
  cudaMemcpy(start_d, start, wire.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(end_d,   end,   wire.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(pinInWire_d, pinInWire, pin_sum * sizeof(int), cudaMemcpyHostToDevice);


  double *wire_exp_x_d;
  double *wire_exp_y_d;
  double *wire_exp_x_neg_d;
  double *wire_exp_y_neg_d;
  cudaMalloc((void **)&wire_exp_x_d, wire.size() * sizeof(double));
  cudaMalloc((void **)&wire_exp_y_d, wire.size() * sizeof(double));
  cudaMalloc((void **)&wire_exp_x_neg_d, wire.size() * sizeof(double));
  cudaMalloc((void **)&wire_exp_y_neg_d, wire.size() * sizeof(double));

  // Compute wire exp sum in GPU
  begin_t = std::chrono::high_resolution_clock::now();
  compute_wire_exp( wire.size(), pin_exp_x_d, pin_exp_y_d, pin_exp_x_neg_d, pin_exp_y_neg_d, start_d, end_d, 
                    wire_exp_x_d, wire_exp_y_d, wire_exp_x_neg_d, wire_exp_y_neg_d, pinInWire_d );
  end_t = std::chrono::high_resolution_clock::now();
  total_run_time += std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();


  start = static_cast<int*>( realloc( start, sizeof(int)*pin_x.size() ) );
  end   = static_cast<int*>( realloc( end,   sizeof(int)*pin_x.size() ) );
  vector<vector<int>> wireInPin;
  wireInPin.resize( pin_x.size() );
  for( int i = 0 ; i < wire.size() ; ++ i )
    for( int j = 0 ; j < wire[i].size() ; ++ j )
      wireInPin[wire[i][j]].push_back( i );
  pin_sum = 0;
  for( int i = 0 ; i < wireInPin.size(); ++ i ){
    start[i] = pin_sum;
    end[i] = start[i] + wireInPin[i].size();
    memcpy(&pinInWire[start[i]] , wireInPin[i].data(), sizeof(int)*wireInPin[i].size() );
    pin_sum += wireInPin[i].size();
  }
  cudaMemcpy(pinInWire_d, pinInWire, pin_sum * sizeof(int), cudaMemcpyHostToDevice);

  int* s;
  int* e;
  cudaMalloc((void **)&s, pin_x.size() * sizeof(double) );
  cudaMalloc((void **)&e, pin_x.size() * sizeof(double) );
  cudaMemcpy(s, start, pin_x.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(e, end, pin_x.size() * sizeof(int), cudaMemcpyHostToDevice);


  double *grad_x;
  double *grad_y;
  cudaMalloc((void **)&grad_x, pin_x.size() * sizeof(double) );
  cudaMalloc((void **)&grad_y, pin_x.size() * sizeof(double) );

  dim3 DimGrid(pin_x.size()/1024+1,1,1);
  dim3 DimBlock(1024,1,1);

  cudaEvent_t start_t, stop_t;
  cudaEventCreate(&start_t);
  cudaEventCreate(&stop_t);

  cudaEventRecord(start_t);

  // Compute gradient in GPU
  begin_t = std::chrono::high_resolution_clock::now();
  compute_grad_kernel<<<DimGrid,DimBlock>>>( s, e, pinInWire_d, pin_exp_x_d, pin_exp_y_d, pin_exp_x_neg_d, pin_exp_y_neg_d, 
                                            wire_exp_x_d, wire_exp_y_d, wire_exp_x_neg_d, wire_exp_y_neg_d, 
                                            pin_x.size(), grad_x, grad_y );
  end_t = std::chrono::high_resolution_clock::now();
  total_run_time += std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();


  cudaEventRecord(stop_t);
  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()
  cudaEventSynchronize(stop_t);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_t, stop_t);
  std::cout << "Compute grad :" << milliseconds << " milli sec" << std::endl;
  checkCUDAerror();

  double *x_host = static_cast<double*>(malloc(sizeof(double)*pin_x.size()));
  double *y_host = static_cast<double*>(malloc(sizeof(double)*pin_x.size()));
  // Copy gradient from GPU to CPU
  begin_t = std::chrono::high_resolution_clock::now();
  cudaMemcpy(x_host, grad_x, pin_x.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(y_host, grad_y, pin_x.size() * sizeof(double), cudaMemcpyDeviceToHost);
  end_t = std::chrono::high_resolution_clock::now();
  std::cout << "Copy grad D2H: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t-begin_t).count() << " milli sec" << std::endl;
  total_run_time += std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();

  std::cout << "Total Direct Wire GPU Run time = " << total_run_time << " micro sec" << std::endl;


  // This is for debugging
  //of.open("./gpu_result");
  //of.setf(ios::fixed,ios::floatfield);
  //of.precision(12);
  //for( int i = 0; i < pin_x.size() ; ++ i ){
  //  if( i < 10 )
  //    //printf("%.25e %.25e\n", x_host[i], y_host[i]);
  //    cout << x_host[i] << " " << y_host[i] << '\n';
  //  of << y_host[i] << '\n';
  //}
  //of.close();
  //delete [] y_host;
}


