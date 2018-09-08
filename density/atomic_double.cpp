#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iomanip>
#include <atomic>
#include <array>

using namespace std;



struct Atom
{
	std::atomic<double> _a;

	Atom():_a()
	{}

	Atom(const std::atomic<double> &a) :_a(a.load())
	{}

	Atom(const Atom &other) :_a(other._a.load())
	{}

	Atom &operator=(const Atom &other){
		_a.store(other._a.load());
    return *this;
	}

	void Add( double val ){
		 auto current = _a.load();
	   while (!_a.compare_exchange_weak(current, current + val)){}
	}

	void show(){
		cout << std::setprecision(15) << std::scientific << _a << '\n';
	}
};


double overlap_area( double x11, double x12, double x21, double x22, 
                     double y11, double y12, double y21, double y22){
  auto x_overlap = std::max(0.0, ::min(x12,x22) - ::max(x11,x21));
  auto y_overlap = std::max(0.0, ::min(y12,y22) - ::max(y11,y21));
  return x_overlap * y_overlap;
}

template<typename T>
T atomic_fetch_add(std::atomic<T> *obj, T arg) {
  T expected = obj->load();
  while(!atomic_compare_exchange_weak(obj, &expected, expected + arg))
    ;
  return expected;
}


void Double_2_HP(double r, uint64_t* arr) {
  int N = 3;
  int K = 2;
  double dtmp = r * double(pow(2.0, 64.0 * (N - K - 1)));
  for (int i = 0; i < N - 1; ++i) {
    uint64_t itmp = (uint64_t)dtmp;
    dtmp = (dtmp - (double)itmp) * pow(2.0, 64.0);
    arr[i] = itmp;
  }
  arr[N - 1] = (uint64_t)dtmp;
}


void HP_2_Double(double &r, uint64_t *arr ){
  int N = 3;
  int K = 2;
  r = 0.0;
  for( int i = 0 ; i < N ; ++ i ){
    r += arr[i]*pow(2, 64*(N-K-i-1));
  }
}

int main( int argc, char* argv[] ){


  if( argc < 2 ){
    printf("No input FILE\n");
    exit(1);
  }
  printf("input = %s\n" , argv[1]);

  //int DIMENSION = 1024;
  constexpr int DIMENSION = 2048;

  ofstream of;
  ifstream fptr;
  //fptr.open("./tune_placer/gpu_density_info");
  fptr.open(argv[1]);
  
  double **map;
  map = static_cast<double**>(malloc(sizeof(double*)*DIMENSION));
  for( size_t i = 0 ; i < DIMENSION ; ++ i ){
    map[i] = static_cast<double*>(malloc(sizeof(double)*DIMENSION));
    memset(map[i], 0.0, sizeof(double)*DIMENSION);
  }

  constexpr int N = 3;
  atomic<uint64_t>* layout = new std::atomic<uint64_t>[N*DIMENSION*DIMENSION];

	vector<Atom> den_map( DIMENSION*DIMENSION );
  #pragma omp parallel for 
	for( int i = 0 ; i < den_map.size(); ++i  ){
		den_map[i]._a = 0.0;
	}


  

  vector<double> pin;
  size_t pin_num;
  double Gx,Gy;
  fptr >> pin_num >> Gx >> Gy;
  double *X = (double*)malloc(sizeof(double)*pin_num);
  double *Y = (double*)malloc(sizeof(double)*pin_num);
  double *W = (double*)malloc(sizeof(double)*pin_num);
  double *H = (double*)malloc(sizeof(double)*pin_num);
  for( int i = 0 ; i < pin_num ; ++ i )
    fptr >> X[i] >> Y[i] >> W[i] >> H[i];
  fptr.close();

  cout << "Pin num = " << pin_num << " " << Gx << "  " << Gy << '\n';

  auto begin_t = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for( size_t i =  0 ; i < pin_num ; ++ i ){
    
    int lx = floor(X[i]/Gx);
    int ly = floor(Y[i]/Gy);

    int ux = min( int(ceil((X[i]+W[i])/Gx)) , DIMENSION );
    int uy = min( int(ceil((Y[i]+H[i])/Gy)) , DIMENSION );

    for( int j = lx ; j < ux ; ++ j ){
      for( int k = ly ; k < uy; ++ k ){
        auto area = overlap_area( X[i], X[i]+W[i], double(j)*Gx, double(j+1)*Gx, 
                                  Y[i], Y[i]+H[i], double(k)*Gy, double(k+1)*Gy );
        //if( j >= DIMENSION || k >= DIMENSION ) {
        //  printf("i = %lu  : %lf , %lf , %lf , %lf  :::  %d , %d", i , X[i], Y[i], W[i], H[i], j , k);
        //}

				den_map[j*DIMENSION + k].Add( area );
      }
    }
  }
  #pragma omp barrier
  auto end_t = std::chrono::high_resolution_clock::now();

  std::cout << " Density CPU run time : " << std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << " micro sec" << std::endl;

 
  // This is for debugging
  //of.open("./cpu_density_result");
  //of.setf(ios::fixed,ios::floatfield);
  //of.precision(5);
  // double maxd = -1.0;
  //for( int i = 0 ; i < DIMENSION ; ++ i )
  //  for( int j = 0 ; j < DIMENSION ; ++ j ){
  //    of << map[i][j] << '\n';
  //    maxd = std::max(maxd, map[i][j]);
  //  }
  //cout << "Max density = " << maxd << '\n';
  //of.close();
   
  //int cc = 0;
  //for( int i = 0 ; i < DIMENSION ; ++ i )
  //  for( int j = 0 ; j < DIMENSION ; ++ j ){
  //  //if( map[i][j] > 0.0 && cc < 10 ){
	//	if( den_map[j*DIMENSION+i]._a > 0.0 && cc < 10 ){
  //    den_map[ j*DIMENSION + i].show();
  //    //printf("%.15e\n", den_map[j*DIMENSION + i]._a );
  //    ++ cc;
  //  }
  //}

  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << std::endl;
}


