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

using namespace std;


double overlap_area( double x11, double x12, double x21, double x22, 
                     double y11, double y12, double y21, double y22){
  auto x_overlap = std::max(0.0, ::min(x12,x22) - ::max(x11,x21));
  auto y_overlap = std::max(0.0, ::min(y12,y22) - ::max(y11,y21));
  return x_overlap * y_overlap;
}

int main( int argc, char* argv[] ){

  if( argc < 2 ){
    printf("No input FILE\n");
    exit(1);
  }
  printf("input = %s\n" , argv[1]);

  //int DIMENSION = 1024;
  int DIMENSION = 2048;

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
  for( size_t i =  0 ; i < pin_num ; ++ i ){
    int lx = floor(X[i]/Gx);
    int ly = floor(Y[i]/Gy);

    //int ux = min( int(ceil((X[i]+W[i])/Gx)) , DIMENSION - 1 );
    //int uy = min( int(ceil((Y[i]+H[i])/Gy)) , DIMENSION - 1 );

    int ux = min( int(ceil((X[i]+W[i])/Gx)) , DIMENSION );
    int uy = min( int(ceil((Y[i]+H[i])/Gy)) , DIMENSION );
  
    for( int j = lx ; j < ux ; ++ j ){
      for( int k = ly ; k < uy; ++ k ){
        auto area = overlap_area( X[i], X[i]+W[i], double(j)*Gx, double(j+1)*Gx, 
                                  Y[i], Y[i]+H[i], double(k)*Gy, double(k+1)*Gy );
        map[k][j] += area;
      }
    }
  }
  auto end_t = std::chrono::high_resolution_clock::now();

  std::cout << " Density CPU run time : " << std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << " micro sec" << std::endl;

 
  // This is for debugging
  //of.open("./cpu_density_result");
  //of.setf(ios::fixed,ios::floatfield);
  //of.precision(5);
  //double maxd = -1.0;
  //for( int i = 0 ; i < DIMENSION ; ++ i )
  //  for( int j = 0 ; j < DIMENSION ; ++ j ){
 //     of << map[i][j] << '\n';
  //    maxd = std::max(maxd, map[i][j]);
  //  }
  //cout << "Max density = " << maxd << '\n';
  //of.close();
  //int cc = 0;
  //for( int i = 0 ; i < DIMENSION ; ++ i )
  //  for( int j = 0 ; j < DIMENSION ; ++ j ){
  //  if( map[i][j] > 0.0 && cc < 10 ){
  //    printf("%.15e\n", map[i][j]);
  //    ++ cc;
  //  }
  //}

  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << std::endl;



}


