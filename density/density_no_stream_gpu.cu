#include <cstring>
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
#include "cub/cub.cuh"


using namespace std;

#define D2H cudaMemcpyDeviceToHost 
#define H2D cudaMemcpyHostToDevice 


#define cudaAlloc(ptr,sz,type) do{ \
  cudaMalloc((void **)&ptr, sz*sizeof(type)); \
}while(0)

#define cudaICalloc(ptr,sz,type) do{ \
  cudaMalloc((void **)&ptr, sz*sizeof(type)); \
  cudaMemset(ptr,0, sizeof(type)*sz ); \
}while(0)

#define cudaDCalloc(ptr,sz,type) do{ \
  cudaMalloc((void **)&ptr, sz*sizeof(type)); \
  cudaMemset(ptr,0.0, sizeof(type)*sz ); \
}while(0)



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
  printf("%s Run Time : %lf milli sec\n", #action, milliseconds ); \
}while(0)


int DIMENSION = 1024;
//int DIMENSION = 2048;


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
void compute_bin_id_kernel(double *x, double *y, int* pid, int *bid, int sz, double Gx, double Gy, int width){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    int lx = floor(x[i]/Gx);
    int ly = floor(y[i]/Gy);
    bid[i] = width*lx + ly;
    pid[i] = i;
  }
}

void compute_bin_id( double *x , double *y, int* pin_id, int* bin_id, int sz, double& Gx, double& Gy, int width, 
      double x_unit, 
      double y_unit){

  //compute_bin_id(X_d, Y_d, pin_id_d, bin_id_d, pin_num, Gx, Gy, width, max_w/Gx, max_h/Gy );
  dim3 DimGrid( (sz/256+1),1,1);
  dim3 DimBlock(256,1,1);
  //compute_bin_id_kernel<<<DimGrid,DimBlock>>>(x,y,pin_id,bin_id,sz, Gx*ceil(xunit),Gy*ceil(yunit), width );
  compute_bin_id_kernel<<<DimGrid,DimBlock>>>(x,y,pin_id,bin_id,sz, ceil(Gx*ceil(x_unit)), ceil(Gy*ceil(y_unit)), 1024/(int)(ceil(y_unit)) );
  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()
  checkCUDAerror();
  cout << "After compute bin id \n";
}














__device__
double overlap_area( double x11, double x12, double x21, double x22, 
                     double y11, double y12, double y21, double y22){
  auto x_overlap = fmax(0.0, fmin(x12,x22) - fmax(x11,x21));
  auto y_overlap = fmax(0.0, fmin(y12,y22) - fmax(y11,y21));
  return x_overlap * y_overlap;
}


__global__
void direct_density_kernel( double* map, double* X, double* Y, double *W, double *H, int sz, double Gx, double Gy, int dim ){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    int lx = floor(X[i]/Gx);
    int ly = floor(Y[i]/Gy);
    //int lx = ceil(X[i]/Gx);
    //int ly = ceil(Y[i]/Gy);

    int ux = ceil((X[i]+W[i])/Gx);
    int uy = ceil((Y[i]+H[i])/Gy);
    
    for( int j = lx ; j < ux ; ++ j ){
      for( int k = ly ; k < uy; ++ k ){
        auto area = overlap_area( X[i], X[i]+W[i], double(j)*Gx, double(j+1)*Gx, 
                                  Y[i], Y[i]+H[i], double(k)*Gy, double(k+1)*Gy );
        atomicAdd( &map[k*dim+j], area );
        //atomicAdd( &map[k*dim+j], area );
      }
    }
  }
}


void direct_density( double *map_d, double* x_d, double* y_d, double *w_d, double *h_d , int pin_num, double Gx, double Gy ){
  printf("Density compute...\n");
  dim3 DimGrid( (pin_num/1024+1),1,1);
  dim3 DimBlock(1024,1,1);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  direct_density_kernel<<<DimGrid,DimBlock>>>( map_d,x_d,y_d,w_d,h_d,pin_num,Gx,Gy,DIMENSION );
  cudaEventRecord(stop);

  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()
  checkCUDAerror();

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Direct Density : " << milliseconds << " milli sec" << std::endl;
}




//------------------------------------------------------------------------------------------//
typedef unsigned long long int uint64_cu;


__device__ inline
uint64_cu myAtomicAdd( uint64_cu *addr, uint64_cu val ){
  uint64_cu old, newval, readback;
  old = *addr;
  newval = old + val;
  while ((readback = atomicCAS(addr, old, newval)) != old) {
    old = readback;
    newval = old + val;
  }
  return old;
}


__device__ __host__
void Double_2_HP(double r, uint64_cu* arr) {
  int N = 3;
  int K = 2;
  double dtmp = r * double(pow(2.0, 64.0 * (N - K - 1)));
  for (int i = 0; i < N - 1; ++i) {
    uint64_cu itmp = (uint64_cu)dtmp;
    dtmp = (dtmp - (double)itmp) * pow(2.0, 64.0);
    arr[i] = itmp;
  }
  arr[N - 1] = (uint64_cu)dtmp;
}

__device__
void addHPs( uint64_cu *a, uint64_cu *b ){
  int N = 3;
  int K = 2;
  //auto old = myAtomicAdd( &a[N-1], b[N-1] );
  auto old = atomicAdd( &a[N-1], b[N-1] );

  uint64_cu co = (old+b[N-1] < b[N-1]);
  for( int i = N - 2 ; i >= 1 ; --i ){
    //old = myAtomicAdd( &a[i], b[i] + co );
    old = atomicAdd( &a[i], b[i] + co );
    co = (old+b[i]+co == b[i]) ? co : (old+b[i]+co < b[i]);
  }
  //myAtomicAdd( &a[0], b[0]+co );
  atomicAdd( &a[0], b[0]+co );
}









__global__
void hp_direct_density_kernel( uint64_cu* map, double* X, double* Y, double *W, double *H , int sz, double Gx, double Gy, int dim ){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    int lx = floor(X[i]/Gx);
    int ly = floor(Y[i]/Gy);

    int ux = ceil((X[i]+W[i])/Gx);
    int uy = ceil((Y[i]+H[i])/Gy);

    int N = 3;
    int K = 2;
    uint64_cu arr[3];

    for( int j = lx ; j < ux ; ++ j ){
      for( int k = ly ; k < uy; ++ k ){
        auto area = overlap_area( X[i], X[i]+W[i], double(j)*Gx, double(j+1)*Gx, 
                                  Y[i], Y[i]+H[i], double(k)*Gy, double(k+1)*Gy );
        Double_2_HP( area, arr );
        addHPs( &map[ (k*dim+j)*N ], arr );
        //atomicAdd( &map[k*dim+j], area );
      }
    }
    
  }
}



void HP_2_Double(double &r, uint64_cu *arr ){
  int N = 3;
  int K = 2;
  r = 0.0;
  for( int i = 0 ; i < N ; ++ i ){
    r += arr[i]*pow(2, 64*(N-K-i-1));
  }
}

void hp_direct_density( double* x_d, double* y_d, double *w_d, double *h_d , int pin_num, double Gx, double Gy ){
  int N = 3;
  int K = 2;
  uint64_cu *result = (uint64_cu*)malloc(sizeof(uint64_cu)*N*DIMENSION*DIMENSION );

  uint64_cu *hp;
  cudaAlloc( hp, N*DIMENSION*DIMENSION, uint64_cu );
  cudaMemset( hp, 0, N*DIMENSION*DIMENSION*sizeof(uint64_cu) );

  dim3 DimGrid( (pin_num/1024+1),1,1);
  dim3 DimBlock(1024,1,1);
  //hp_direct_density_kernel<<<DimGrid,DimBlock>>>( hp, x_d, y_d, w_d, h_d, pin_num, Gx, Gy, DIMENSION );
  //checkCUDAerror();
  GoKernel("HP Direct Sum : " , hp_direct_density_kernel, hp, x_d, y_d, w_d, h_d, pin_num, Gx, Gy, DIMENSION );


  cudaMemcpy( result, hp, sizeof(uint64_cu)*DIMENSION*DIMENSION*N, D2H );
  printf("Show Result \n");
  int qq = 0;
  for( int i = 0 ; i < DIMENSION ; ++ i )
    for( int j = 0 ; j < DIMENSION; ++ j ){
      double tmp = 0.0;
      HP_2_Double( tmp, &result[(i*DIMENSION+j)*N] );
      if( tmp > 0.0  && qq < 30 ){
        printf("%.15e\n", tmp);
        //cout << std::setprecision(15) << tmp << '\n';
        qq ++;
      }
    }
 
  exit(1);
}

__device__
void addHP_Double( uint64_cu *a, double val ){
  uint64_cu b[3];
  Double_2_HP( val, b );

  int N = 3;
  int K = 2;
  //auto old = myAtomicAdd( &a[N-1], b[N-1] );
  auto old = atomicAdd( &a[N-1], b[N-1] );

  uint64_cu co = (old+b[N-1] < b[N-1]);
  for( int i = N - 2 ; i >= 1 ; --i ){
    //old = myAtomicAdd( &a[i], b[i] + co );
    old = atomicAdd( &a[i], b[i] + co );
    co = (old+b[i]+co == b[i]) ? co : (old+b[i]+co < b[i]);
  }
  //myAtomicAdd( &a[0], b[0]+co );
  atomicAdd( &a[0], b[0]+co );
}



















__global__
void compute_count_kernel( int *counter, double Gx, double Gy, double *W, double *H, int sz ){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    int x = fmod( W[i], Gx ) == 0.0 ? W[i]/Gx + 1 : ceil(W[i]/Gx);
    int y = fmod( H[i], Gy ) == 0.0 ? H[i]/Gy + 1 : ceil(H[i]/Gy);
    //int x = floor(W[i]/Gx) + 1;
    //int y = floor(H[i]/Gy) + 1;

    counter[i] = (ceil( (W[i]+Gx)/Gx)) * (ceil( (H[i]+Gy)/Gy));
    //counter[i] = x * y;
    assert(counter[i] > 0 );
    //counter[i] = (x+2)*(y+2);
  }
}

void compute_count(int *counter, double Gx, double Gy, double *W, double *H, int sz ){
  dim3 DimGrid( (sz/1024+1),1,1);
  dim3 DimBlock(1024,1,1);
  GoKernel( "Compute Count : ", compute_count_kernel, counter, Gx, Gy, W, H, sz );
}

__global__
void assign_id_count_kernel( int *my_id, int *my_count, int *hist, int *counter, int sz ){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < sz ){
    for( int j = 0 ; j < counter[i] ; ++j ){
      my_id[ hist[i]+j ] = i;
      my_count[ hist[i]+j ] = j;
    }
  }
}


void assign_id_count( int *my_id, int *my_count, int *hist, int *counter, int sz){
  dim3 DimGrid( (sz/1024+1),1,1);
  dim3 DimBlock(1024,1,1);
  GoKernel( "Assign ID Count : ", assign_id_count_kernel, my_id, my_count, hist, counter, sz );
}





__global__
void compute_each_density_kernel( uint64_cu* map, double* X, double* Y, double *W, double *H , int sz, double Gx, double Gy, int dim,
      int *my_id, int *my_count, int total_computation, int dimension ){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;

  if( i < total_computation ){
    int cid = my_id[i];
    int lx = floor(X[cid]/Gx);
    int ly = floor(Y[cid]/Gy);

    //int xdim = ceil( double(W[cid]/Gx) );
    //int ydim = ceil( double(H[cid]/Gy) );


    int xdim = ceil( (W[cid]+Gx)/Gx );
    int ydim = ceil( (H[cid]+Gy)/Gy );

     //int x = min( my_count[i]%xdim + lx, 1023 );
    //int y = min( my_count[i]/xdim + ly, 1023 );
    int x = my_count[i]%xdim + lx;
    int y = my_count[i]/xdim + ly;

    if( x > dimension - 1 || y > dimension - 1 )
      return ;





    int N = 3;
    int K = 2;
    uint64_cu arr[3];
    /*
       if( my_count[i] == 0 ){
       lx = floor(X[cid]/Gx);
       ly = floor(Y[cid]/Gy);
       int ux = ceil((X[cid]+W[cid])/Gx);
       int uy = ceil((Y[cid]+H[cid])/Gy);

       for( int j = lx ; j < ux ; ++ j ){
       for( int k = ly ; k < uy; ++ k ){
       auto area = overlap_area( X[cid], X[cid]+W[cid], double(j)*Gx, double(j+1)*Gx, 
       Y[cid], Y[cid]+H[cid], double(k)*Gy, double(k+1)*Gy );
       Double_2_HP( area, arr );
       addHPs( &map[ (k*dim+j)*N ], arr );
    //atomicAdd( &map[k*dim+j], area );
    }
    }
    }
    */

    auto area = overlap_area( X[cid], X[cid]+W[cid], double(x)*Gx, double(x+1)*Gx, 
        Y[cid], Y[cid]+H[cid], double(y)*Gy, double(y+1)*Gy );

    if( area <= 0.0 )
      return ;
    
    
    Double_2_HP( area, arr );
    addHPs( &map[ (y*dim+x)*N ], arr );
  }
}


void compute_each_density( double* x_d, double* y_d, double *w_d, double *h_d , int pin_num, double Gx, double Gy, 
                          int *my_id, int *my_count, int total_computation ){
  int N = 3;
  int K = 2;
  uint64_cu *result = (uint64_cu*)malloc(sizeof(uint64_cu)*N*DIMENSION*DIMENSION );

  uint64_cu *hp;
  cudaAlloc( hp, N*DIMENSION*DIMENSION, uint64_cu );
  cudaMemset( hp, 0, N*DIMENSION*DIMENSION*sizeof(uint64_cu) );


  dim3 DimGrid( (total_computation/1024+1),1,1);
  dim3 DimBlock( 1024,1,1 );

  auto begin_t = std::chrono::high_resolution_clock::now();
  GoKernel("Compute Each Density : " , compute_each_density_kernel, hp, x_d, y_d, w_d, h_d, pin_num, Gx, Gy, DIMENSION, 
                                       my_id, my_count, total_computation, DIMENSION );

  cudaMemcpy( result, hp, sizeof(uint64_cu)*DIMENSION*DIMENSION*N, D2H );
  auto end_t = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_t-begin_t).count() << std::endl;

  //exit(1);

  printf("Show Result \n");
  int qq = 0;
  for( int i = 0 ; i < DIMENSION ; ++ i )
    for( int j = 0 ; j < DIMENSION; ++ j ){
      double tmp = 0.0;
      HP_2_Double( tmp, &result[(i*DIMENSION+j)*N] );
      if( tmp > 0.0  && qq < 30 ){
        printf("%.15e\n", tmp);
        //cout << std::setprecision(15) << tmp << '\n';
        qq ++;
      }
    }
 
  exit(1);
}


int thrust_prefix_sum(int *d_in, int *d_out, int N ){

  int num_items = N;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

  cudaDeviceSynchronize(); // if we want to use printf in kernel, must have cudaDeviceSynchronize()
  checkCUDAerror();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Prefix Runtime : " << milliseconds << " milli sec" << std::endl;

  int last_num;
  int tmp;
  cudaMemcpy( &last_num, &d_out[N-1], sizeof(int), D2H );
  cudaMemcpy( &tmp,      &d_in[N-1],  sizeof(int), D2H );
  cout << "last num = " << last_num << "   tmp = " << tmp << '\n';
  return tmp + last_num;
}




__global__
void row_count_kernel( int pin_num, double *X, double *Y, double *W, double *H, int *count, double Gx, double Gy ){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < pin_num ){
    int ly = floor( Y[i]/Gy );
    int uy = ceil( (Y[i] + H[i])/Gy );
    for( int j = ly ; j < uy ; ++ j )
      atomicAdd( &count[j], 1 );
  }
}

void row_count(int pin_num, double *X, double *Y, double *W, double *H, int *count, double Gx, double Gy){
  dim3 DimGrid( (pin_num/1024+1),1,1);
  dim3 DimBlock( 1024,1,1 );
  GoKernel( "Row Count : " , row_count_kernel, pin_num, X, Y, W, H, count, Gx, Gy);
}


__device__ __inline__ 
uint get_peers(int key) {
	uint peers = 0;
	bool is_peer;
	uint unclaimed=0xffffffff; // in the beginning, no threads are claimed
	do {
		int other_key=__shfl(key,__ffs(unclaimed)-1);// get key from least unclaimed lane
		is_peer=(key==other_key); // do we have a match?
		peers=__ballot(is_peer); // find all matches
		unclaimed^=peers; // matches are no longer unclaimed
	} while (!is_peer); // repeat as long as we haven’t found our match
	return peers;
}

__device__ __inline__ 
void add_peers(uint64_cu* dest, double x, uint peers, int key ) {
  //int lane = TX & 31;
  int lane = threadIdx.x & 31;
  int first = __ffs(peers) - 1;                // find the leader
  int rel_pos = __popc(peers << (32 - lane));  // find our own place
  peers &= (0xfffffffe << lane);               // drop everything to our right
  while (__any(peers)) {        // stay alive as long as anyone is working
    int next = __ffs(peers);    // find out what to add
    double t = __shfl(x, next - 1);  // get what to add (undefined if nothing)
    if (next)  // important: only add if there really is anything
      x += t;
    int done = rel_pos & 1;  // local data was used in iteration when its LSB is set
    peers &= __ballot(!done);  // clear out all peers that were just used
    rel_pos >>= 1;             // count iterations by shifting position
  }
  if (lane == first)  // only leader threads for each key perform atomics
    addHP_Double( &dest[ key  ], x );
  //  atomicAdd(&dest[key], x);
  //F res = __shfl(x, first);  // distribute result (if needed)
  //return res;  // may also return x or return value of atomic, as needed
}





__global__
void compute_portion_density_kernel( uint64_cu* map, double* X, double* Y, double *W, double *H , double Gx, double Gy, int dim,
      int *my_id, int *my_count, int total_computation, int sz, int start, int dimension ){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;

  if( i < sz ){
    int cid = my_id[start+i];
    int lx = floor(X[cid]/Gx);
    int ly = floor(Y[cid]/Gy);

    int xdim = ceil( (W[cid]+Gx)/Gx );
    int ydim = ceil( (H[cid]+Gy)/Gy );

    int x = min( my_count[start+i]%xdim + lx, dimension - 1);
    int y = min( my_count[start+i]/xdim + ly, dimension - 1);

    int N = 3;
    int K = 2;
    uint64_cu arr[3];
    

    auto area = overlap_area( X[cid], X[cid]+W[cid], double(x)*Gx, double(x+1)*Gx, 
        Y[cid], Y[cid]+H[cid], double(y)*Gy, double(y+1)*Gy );

    //auto p = get_peers( (y*dim+x)*N );
    //add_peers( map, area, p, (y*dim+x)*N );
   
    Double_2_HP( area, arr );
    addHPs( &map[ (y*dim+x)*N ], arr );
  }
}


__global__
void hp_to_double_kernel( uint64_cu* map, double* map_d, int bin_num ){
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < bin_num ){
    int N = 3;
    int K = 2;
    map_d[i] = 0.0;
    for( int j = 0 ; j < N ; ++ j ){
      map_d[i] += map[3*i+j]*pow(2, 64*(N-K-j-1));
    }
  }
}

void convert_to_double( uint64_cu *map, double *map_d , int bin_num ){
  dim3 DimGrid( (bin_num/1024+1),1,1);
  dim3 DimBlock( 1024,1,1 );
  GoKernel( "HP to Double Kernel : " , hp_to_double_kernel, map, map_d, bin_num );
}


int main( int argc, char* argv[] ){

  if( argc < 2 ){
    printf("No input FILE\n");
    exit(1);
  }
  printf("input = %s\n" , argv[1]);


  ofstream of;
  ifstream fptr;
  //fptr.open("./tune_placer/gpu_density_info");
  fptr.open(argv[1]);
  
  double *map = static_cast<double*>(malloc(sizeof(double)*DIMENSION*DIMENSION));

  double* X;
  double* Y;
  double* W;
  double* H;
  vector<double> pin;
  size_t pin_num;
  double Gx,Gy;
  fptr >> pin_num >> Gx >> Gy;

  cout << pin_num << " " << Gx << "  " << Gy << '\n';
  X = static_cast<double*>(malloc(sizeof(double)*pin_num));
  Y = static_cast<double*>(malloc(sizeof(double)*pin_num));
  W = static_cast<double*>(malloc(sizeof(double)*pin_num));
  H = static_cast<double*>(malloc(sizeof(double)*pin_num));


  auto begin_t = std::chrono::high_resolution_clock::now();
  double max_w = -1;
  double max_h = -1;
  for( size_t i =  0 ; i < pin_num ; ++ i ){
    fptr >> X[i] >> Y[i] >> W[i] >> H[i];
    max_w = max(W[i],max_w);
    max_h = max(H[i],max_h);
  }
  cout << "Crossing : " << max_w/Gx << "   " << max_h/Gy << '\n';

  int unit_x = (int)ceil(max_w/Gx);
  int unit_y = (int)ceil(max_h/Gy);


  fptr.close();
  auto end_t = std::chrono::high_resolution_clock::now();
  std::cout << "Reading : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t-begin_t).count() << " milli sec" << std::endl;

  int *pin_id_d ;
  int *bin_id_d ;
  int *sorted_bin_id;
  int *sorted_pin_id;
  double *X_d;
  double *Y_d;
  double *W_d;
  double *H_d;
  cudaMalloc((void **)&X_d, pin_num * sizeof(double) );
  cudaMalloc((void **)&Y_d, pin_num * sizeof(double) );
  cudaMalloc((void **)&W_d, pin_num * sizeof(double) );
  cudaMalloc((void **)&H_d, pin_num * sizeof(double) );


  begin_t = std::chrono::high_resolution_clock::now();
  cudaMemcpy(X_d, X, pin_num* sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Y_d, Y, pin_num* sizeof(double), cudaMemcpyHostToDevice);
  end_t = std::chrono::high_resolution_clock::now();
  auto total_run_time = std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();

  cudaMemcpy(W_d, W, pin_num* sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(H_d, H, pin_num* sizeof(double), cudaMemcpyHostToDevice);


  double *map_d;
  cudaMalloc((void **)&map_d, DIMENSION*DIMENSION* sizeof(double) );
  cudaMemset( map_d, 0.0, DIMENSION*DIMENSION*sizeof(double));


  int *counter;
  cudaAlloc( counter, pin_num, int );
  cudaMemset( counter, 0, pin_num*sizeof(int) );
  printf("counter = %d\n" , counter );
  compute_count( counter, Gx, Gy, W_d, H_d, pin_num );
  printf("Compute Count\n");
 
  int *hist;
  cudaAlloc( hist, pin_num, int );
  cudaMemset( hist, 0, pin_num*sizeof(int) );
  auto total_computation = thrust_prefix_sum( counter, hist, pin_num );
  printf("total computation = %d\n", total_computation );

  int* my_id;
  int* my_count;
  cudaAlloc( my_id, total_computation, int );
  cudaAlloc( my_count, total_computation, int );

  assign_id_count( my_id, my_count, hist, counter, pin_num );

  //compute_each_density( X_d, Y_d, W_d, H_d, pin_num, Gx, Gy, my_id, my_count, total_computation);
 
  double *map_double;
  cudaAlloc( map_double, DIMENSION*DIMENSION, double );

  uint64_cu *hp;
  int N = 3;
  cudaAlloc( hp, N*DIMENSION*DIMENSION, uint64_cu );
  cudaMemset( hp, 0, N*DIMENSION*DIMENSION*sizeof(uint64_cu) );

  begin_t = std::chrono::high_resolution_clock::now();
  dim3 DimGrid( (total_computation/1024+1),1,1) ;
  dim3 DimBlock(1024,1,1);
  compute_portion_density_kernel<<<DimGrid,DimBlock>>>( hp, X_d, Y_d, W_d, H_d, Gx, Gy, DIMENSION,
      my_id, my_count, total_computation, total_computation, 0 , DIMENSION );
  cudaDeviceSynchronize();
  convert_to_double( hp, map_double, DIMENSION*DIMENSION );
  cudaMemcpy( map, map_double, sizeof(double)*DIMENSION*DIMENSION, D2H );
  end_t = std::chrono::high_resolution_clock::now();
  total_run_time += std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();

  std::cout << "Total Run Time : "<< std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << " micro secs\n";

  // This is for debugging
  //printf("Show Result \n");
  //int qq = 0;
  //for( int i = 0 ; i < DIMENSION ; ++ i )
  //  for( int j = 0 ; j < DIMENSION; ++ j ){
  //    if( map[i*DIMENSION+j] > 0.0  && qq < 10 ){
  //      printf("%.15e\n", map[i*DIMENSION+j]);
  //      //cout << std::setprecision(15) << tmp << '\n';
  //      qq ++;
  //    }
  //  }
 
  //std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << "\n";
  std::cout << total_run_time << "\n";
	return 0;
}


