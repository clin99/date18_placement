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
#include <string>
#include <sstream>

#include <cuda_runtime.h>
#include "cusparse.h"


#define Gamma 100.0

using namespace std;
#define D2H cudaMemcpyDeviceToHost 
#define H2D cudaMemcpyHostToDevice 



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

#define GoKernel(action, kernel_name, ...) do{ \
  cudaEvent_t start_t, stop_t; \
  cudaEventCreate(&start_t); \
  cudaEventCreate(&stop_t); \
  cudaEventRecord(start_t); \
  kernel_name<<<DimGrid,DimBlock>>>(__VA_ARGS__); \
  cudaEventRecord(stop_t); \
  cudaStreamSynchronize(0); \
  checkCUDAerror(); \
  cudaEventSynchronize(stop_t); \
  float milliseconds = 0; \
  cudaEventElapsedTime(&milliseconds, start_t, stop_t); \
  std::cout.precision(7); \
  std::cout.width(40); cout << std::right << #action << " Run Time : " << milliseconds << " milli sec\n"; \
}while(0)



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
  GoKernel( "Exponent Sum Time : " , exp_sum, pin_x, pin_y, pin_size, pin_exp_d );
}

__global__
void stream_exp_sum(double *pin_x , double *pin_y , size_t num , double *out, int start, size_t sz )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x + start;
  //if( i < num ){
	if(  blockIdx.x*blockDim.x + threadIdx.x < num ){
    out[i]   = exp( pin_x[i]/Gamma );
    out[i+sz] = exp( -pin_x[i]/Gamma );
    out[i+2*sz] = exp( pin_y[i]/Gamma );
    out[i+3*sz] = exp( -pin_y[i]/Gamma );
  }
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
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  neg_odd_col_kernel<<<DimGrid,DimBlock>>>(mat , pin_num);
  checkCUDAerror();
  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()

  cudaEventRecord(stop);
  checkCUDAerror();

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Negate Odd Row : " << milliseconds << " milli sec" << std::endl;
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
void matrix_dot_product_kernel(double* A, double* B, double* C, double *D, size_t sz )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    //C[i] =  A[i*4]  *B[i*4];
    //C[i] += A[i*4+1]*B[i*4+1];
    //C[i] += A[i*4+2]*B[i*4+2];
    //C[i] += A[i*4+3]*B[i*4+3];

    C[i]  = A[i]  *B[i];
    C[i] -= A[i+1*sz]*B[i+1*sz];
    D[i]  = A[i+2*sz]*B[i+2*sz];
    D[i] -= A[i+3*sz]*B[i+3*sz];
    //if( i < 10 )
    //  printf("%lf , %lf, %lf ,%lf , %.12e\n", B[i+2*sz], B[i+3*sz] , A[i+2*sz], A[i+3*sz] , D[i] );
  }
}


// B is transpose
                           //( midVal, pin_exp_d, gradient, pin_x.size() ); 
void gpu_matrix_dot_product( double *A , double *B, double *C, double *D, size_t pin_size ){
  dim3 DimGrid( (pin_size/256+1),1,1);
  dim3 DimBlock(256,1,1);

  GoKernel("Dot Product Run Time : " , matrix_dot_product_kernel, A, B, C, D, pin_size );
}


__global__
void stream_matrix_dot_product_kernel(double* A, double* B, double* C, double *D, size_t sz, size_t start, size_t num )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x + start;
  if( blockIdx.x*blockDim.x + threadIdx.x < num ){
    C[i]  = A[i]  *B[i];
    C[i] -= A[i+1*sz]*B[i+1*sz];
    D[i]  = A[i+2*sz]*B[i+2*sz];
    D[i] -= A[i+3*sz]*B[i+3*sz];
  }
}


void read_pin_location( vector<double> &pin_x , vector<double> &pin_y ){
  int pin_num;
  double x,y,w,h;
  ifstream fptr;
  vector<double> x_v;
  vector<double> y_v;
  string tname(testcase_name[ID]);
  tname += "_gpu_density_info";

  double Gx, Gy;
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
void print_GPU_kernel(int *array, size_t sz )
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    printf("%d : %d\n" , i , array[i]);
  }
}


void print_GPU( int *array, size_t sz ){
  dim3 DimGrid( (sz/256+1),1,1);
  dim3 DimBlock(256,1,1);
  print_GPU_kernel<<<DimGrid,DimBlock>>>(array,sz);
  checkCUDAerror();
  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

int main( int argc, char *argv[] ){
  if(argc < 2 ){
    printf("Error : No testcase ID\n");
    exit(1);
  }
  istringstream ss(argv[1]);
  ss >> ID;
  printf("Argv[1] (ID) = %d\n" , ID);



  string tname(testcase_name[ID]);
  tname += "_gpu_wire_info";
	printf("Wire info = %s\n" , tname.c_str() );


  ofstream of;
  ifstream fptr;
  //fptr.open("./blue12");
  //fptr.open("./tune_placer/gpu_wire_info");
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

  //for( size_t i = 0 ; i < pin.size() ; ++ i )
  //  pin[i] = double(i%10);
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

  begin_t = std::chrono::high_resolution_clock::now();
  double* pin_d;
  double* out_d;

  int *start_d;
  int *end_d;
  int *pinInWire_d;
  double *pin_exp_d;

  cudaMalloc((void **)&pin_d, pin_x.size() * sizeof(double) * 2);
  cudaMalloc((void **)&out_d, wire.size()  * sizeof(double) * 4); // 4 : xi, -xi, yi, -yi

  cudaMalloc((void **)&pin_exp_d, pin_x.size() * sizeof(double) * 4);

  
  cudaMalloc((void **)&start_d, wire.size() * sizeof(int));
  cudaMalloc((void **)&end_d, wire.size() * sizeof(int));
  cudaMalloc((void **)&pinInWire_d, pin_sum* sizeof(int));

  cudaMemcpy(start_d, start, wire.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(end_d,   end,   wire.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(pinInWire_d, pinInWire, pin_sum * sizeof(int), cudaMemcpyHostToDevice);


  // Create CUDA streams and set up the range to be process by each stream
  int num_stream = 16;
	cudaStream_t stream[num_stream];
	int my_start[num_stream];
	int my_end[num_stream];
	double* pin_x_h;
	double* pin_y_h;
  assert( cudaSuccess == cudaMallocHost( (void**)&pin_x_h, pin_x.size()*sizeof(double) ) );
  assert( cudaSuccess == cudaMallocHost( (void**)&pin_y_h, pin_y.size()*sizeof(double) ) );
  memcpy( pin_x_h, pin_x.data(), pin_x.size()*sizeof(double) );
  memcpy( pin_y_h, pin_y.data(), pin_x.size()*sizeof(double) );

	double* pin_x_d;
	double* pin_y_d;
  cudaMalloc( (void**)&pin_x_d, pin_x.size()*sizeof(double) );
  cudaMalloc( (void**)&pin_y_d, pin_y.size()*sizeof(double) );

	for( int i = 0 ; i < num_stream ; ++ i ){
  	cudaStreamCreate(&stream[i]);
    if( i == 0 )
			my_start[i] = 0;
		else
			my_start[i] = my_end[i-1];
		
		if( i != num_stream - 1 )
			my_end[i] = my_start[i] + pin_x.size()/num_stream;
		else
			my_end[i] = pin_x.size();
	}

  cout << "Start GPU computing...\n";

  // Copy pin coordinates to GPU and compute exponential values
  begin_t = std::chrono::high_resolution_clock::now();

	for( int i  = 0 ; i < num_stream ; ++ i ){
		auto sz = my_end[i] - my_start[i];
    cudaMemcpyAsync( &pin_x_d[my_start[i]], &pin_x_h[my_start[i]], sz*sizeof(double), H2D, stream[i]);
    cudaMemcpyAsync( &pin_y_d[my_start[i]], &pin_y_h[my_start[i]], sz*sizeof(double), H2D, stream[i]);
		dim3 DimGrid(sz/1024+1,1,1);
		dim3 DimBlock(1024,1,1);
		stream_exp_sum<<<DimGrid,DimBlock,0, stream[i]>>>( pin_x_d, pin_y_d, sz, pin_exp_d, my_start[i], pin_x.size() ); 
	}
	cudaDeviceSynchronize();
  end_t = std::chrono::high_resolution_clock::now();
  auto total_run_time = std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();
  std::cout << "First : " << std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << " micro sec" << std::endl;


  // ------------------ Prepare Connectivity matrix for GPU sparse matrix multiplication ------------
	cusparseStatus_t status;
	cusparseHandle_t handle=0;
	cusparseMatDescr_t descr=0;
 
  // DEVICE 
  auto nnz = pin_sum;
  int *cooRowIndex=0;
  int *cooColIndex=0;   
  double *cooVal=0;
  double *midVal=0;
  assert( cudaMalloc((void**)&midVal,pin_x.size()*4*sizeof(double)) == cudaSuccess ); 

  // HOST
  int *cooRowIndexHostPtr=0; 
  int *cooColIndexHostPtr=0; 
  double *cooValHostPtr=0;

  // HOST
  cooRowIndexHostPtr = (int *)malloc(nnz*sizeof(int)); 
  cooColIndexHostPtr = (int *)malloc(nnz*sizeof(int));
  cooValHostPtr      = (double *)malloc(nnz*sizeof(double));
  
  // -------------------- Prepare pin(row)/wire(col) connectivity matrix -------------

	cusparseHandle_t handle_pw=0;
	cusparseMatDescr_t descr_pw=0;

  int *cooRowIndex_pw=0;
  int *cooColIndex_pw=0;   

  int *cooRowIndexHostPtr_pw;
  int *cooColIndexHostPtr_pw;
  double *cooValHostPtr_pw;

  double *cooVal_pw=0;

  cooRowIndexHostPtr_pw = (int *)malloc(nnz*sizeof(int)); 
  cooColIndexHostPtr_pw = (int *)malloc(nnz*sizeof(int));
  cooValHostPtr_pw      = (double *)malloc(nnz*sizeof(double));

  vector<vector<int>> pin2wire;
  pin2wire.resize( pin_x.size() );
  for( int i = 0 ; i < wire.size() ; ++ i ){
    for( int j = 0 ; j < wire[i].size() ; ++ j )
      pin2wire[ wire[i][j] ].push_back( i );
  }

  pin_sum = 0;
  // Row : Pin   ,  Col : wires of this pin
  for( int i = 0 ; i < pin2wire.size() ; ++ i ){
    for( int j = 0 ;  j < pin2wire[i].size() ; ++ j ){
      cooRowIndexHostPtr_pw[pin_sum] = i;
      cooColIndexHostPtr_pw[pin_sum] = pin2wire[i][j];
      cooValHostPtr_pw[pin_sum] = 1.0;
      ++ pin_sum;
    }
  }

  assert( pin_sum == nnz );

  int *cooRowIndex_pw_d=0;
  int *cooColIndex_pw_d=0;   
  double *cooVal_pw_d=0;

  assert( cudaMalloc((void**)&cooRowIndex_pw_d,nnz*sizeof(int)) == cudaSuccess ); 
  assert( cudaMalloc((void**)&cooColIndex_pw_d,nnz*sizeof(int)) == cudaSuccess );
  assert( cudaMalloc((void**)&cooVal_pw_d,     nnz*sizeof(double)) == cudaSuccess );

  assert( cudaMemcpy(cooRowIndex_pw_d, cooRowIndexHostPtr_pw, (size_t)(nnz*sizeof(int)), cudaMemcpyHostToDevice) == cudaSuccess );
  assert( cudaMemcpy(cooColIndex_pw_d, cooColIndexHostPtr_pw, (size_t)(nnz*sizeof(int)), cudaMemcpyHostToDevice) == cudaSuccess );
  assert( cudaMemcpy(cooVal_pw_d, cooValHostPtr_pw, (size_t)(nnz*sizeof(double)), cudaMemcpyHostToDevice)  == cudaSuccess );

  cout << "NNZ = " << nnz << '\n';

  assert( cusparseCreate(&handle_pw) == CUSPARSE_STATUS_SUCCESS );
  assert( cusparseCreateMatDescr(&descr_pw) == CUSPARSE_STATUS_SUCCESS );
  cusparseSetMatType(descr_pw,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_pw,CUSPARSE_INDEX_BASE_ZERO);

  int *csrRowPtr_pw=0;
  /* exercise conversion routines (convert matrix from COO 2 CSR format) */ 
  assert( cudaMalloc((void**)&csrRowPtr_pw,(pin_x.size()+1)*sizeof(int)) == cudaSuccess );
  assert( cusparseXcoo2csr(handle_pw,cooRowIndex_pw_d,nnz,pin_x.size(),csrRowPtr_pw,CUSPARSE_INDEX_BASE_ZERO) == CUSPARSE_STATUS_SUCCESS);



  // -------------------- cuSPARSE matrix multiplication --------------------------

  for( int i = 0 ; i < wire.size() ; ++ i ){
    sort(wire[i].begin(), wire[i].end());
  }

  // HOST
  size_t count = 0;
  // Row : Wire   ,  Col : Pins on this wire
  for( int i = 0 ; i < wire.size() ; ++ i ){
    for( int j = 0 ;  j < wire[i].size() ; ++ j ){
      cooRowIndexHostPtr[count] = i;
      cooColIndexHostPtr[count] = wire[i][j];
      cooValHostPtr[count] = 1.0;

      ++ count;
    }
  }
  assert( nnz == count );
  printf("count = %d , pin_sum = %d\n", count , pin_sum );

  // DEVICE
  assert( cudaMalloc((void**)&cooRowIndex,nnz*sizeof(int)) == cudaSuccess ); 
  assert( cudaMalloc((void**)&cooColIndex,nnz*sizeof(int)) == cudaSuccess );
  assert( cudaMalloc((void**)&cooVal,     nnz*sizeof(double)) == cudaSuccess );

  assert( cudaMemcpy(cooRowIndex, cooRowIndexHostPtr, (size_t)(nnz*sizeof(int)), cudaMemcpyHostToDevice) == cudaSuccess );
  assert( cudaMemcpy(cooColIndex, cooColIndexHostPtr, (size_t)(nnz*sizeof(int)), cudaMemcpyHostToDevice) == cudaSuccess );
  assert( cudaMemcpy(cooVal,      cooValHostPtr,      (size_t)(nnz*sizeof(double)), cudaMemcpyHostToDevice)  == cudaSuccess );

  /* initialize cusparse library */
  assert( cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS );
  assert( cusparseCreateMatDescr(&descr) == CUSPARSE_STATUS_SUCCESS );
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

  std::cout << "Start Sparse Matrix mulitplication ...\n";

  int *csrRowPtr=0;
  /* exercise conversion routines (convert matrix from COO 2 CSR format) */ 
  assert( cudaMalloc((void**)&csrRowPtr,(wire.size()+1)*sizeof(int)) == cudaSuccess );
  assert( cusparseXcoo2csr(handle,cooRowIndex,nnz,wire.size(), csrRowPtr,CUSPARSE_INDEX_BASE_ZERO) == CUSPARSE_STATUS_SUCCESS);

  // Sparse matrix-vector multiplication : y = alpha*A*x + beta*y
  double alpha = 1.0;
  double beta = 0.0;

  cout << "Pin num = " << pin_num << '\n';
  printf("count = %d , pin_sum = %d , nnz = %d\n", count , pin_sum , nnz );
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    begin_t = std::chrono::high_resolution_clock::now();
    /* exercise Level 3 routines (csrmv) */ 
    /* Multiply to get sum of pins for each net */
    assert( cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, wire.size(), 4, pin_num, nnz, &alpha, 
          descr, cooVal, csrRowPtr, cooColIndex, pin_exp_d, pin_num, &beta, out_d, wire.size() ) == CUSPARSE_STATUS_SUCCESS );
    cudaDeviceSynchronize(); 
    gpu_reciprocal( out_d , wire.size() * 4);
    cudaDeviceSynchronize(); 

    end_t = std::chrono::high_resolution_clock::now();
    total_run_time += std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Net Sum : " << milliseconds << " milli sec" << std::endl;
  }


  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 

    begin_t = std::chrono::high_resolution_clock::now();

    /*  Multiply pin-wire connectivity matrix & reciprocal of wire sum  */
    assert( cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, pin_x.size(), 4, wire.size(), nnz, &alpha, 
          descr, cooVal_pw_d, csrRowPtr_pw, cooColIndex_pw_d, &out_d[0], wire.size(), &beta, &midVal[0], pin_x.size() ) 
        == CUSPARSE_STATUS_SUCCESS );

    cudaEventRecord(stop);
    cudaDeviceSynchronize(); 
    end_t = std::chrono::high_resolution_clock::now();
    total_run_time += std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();


    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Sum of reciprocal of Net : " << milliseconds << " milli sec" << std::endl;
  }


  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()

  cout << "Pin size = " << pin_x.size() << "  Wire size = " << wire.size() << '\n';


  double *gradient_x=0;
  assert( cudaMalloc((void**)&gradient_x, pin_x.size()*sizeof(double)) == cudaSuccess ); 
  double *gradient_y=0;
  assert( cudaMalloc((void**)&gradient_y, pin_x.size()*sizeof(double)) == cudaSuccess ); 

  double* x_host;
  double* y_host;
  assert( cudaSuccess == cudaMallocHost( (void**)&x_host, pin_x.size()*sizeof(double) ) );
  assert( cudaSuccess == cudaMallocHost( (void**)&y_host, pin_y.size()*sizeof(double) ) );


  begin_t = std::chrono::high_resolution_clock::now();

	for( int i = 0 ; i < num_stream ; i ++ ){
		auto sz = my_end[i] - my_start[i];
    dim3 DimGrid( (sz/1024+1),1,1);
    dim3 DimBlock(1024,1,1);
    stream_matrix_dot_product_kernel<<<DimGrid,DimBlock,0,stream[i]>>>( 
				midVal, pin_exp_d, gradient_x, gradient_y, pin_x.size(), my_start[i], sz ); 
    cudaMemcpyAsync( &x_host[my_start[i]], &gradient_x[my_start[i]], sz*sizeof(double), D2H, stream[i]);
    cudaMemcpyAsync( &y_host[my_start[i]], &gradient_y[my_start[i]], sz*sizeof(double), D2H, stream[i]);
	}
	cudaDeviceSynchronize();
  end_t = std::chrono::high_resolution_clock::now();
  total_run_time += std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();

  std::cout << "Total Wire MM GPU Run time = " << total_run_time << " micro sec" << std::endl;


  // This is for debugging
  ////of.open("./gpu_result");
  ////of.setf(ios::fixed,ios::floatfield);
  ////of.precision(12);
  //for( int i = 0; i < pin_x.size() ; ++ i ){
  //  if( i < 10 )
  //    printf("%g %g\n", x_host[i], y_host[i]);
  //    //printf("%.30e %.30e\n", x_host[i], y_host[i]);
  //  //  cout << x_host[i] << "   "  << y_host[i] << '\n';
  //  //of << y_host[i] << '\n';
  //}
  ////of.close();
  ////delete [] y_host;

}


