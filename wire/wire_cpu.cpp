#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <random>
#include <cmath>
#include <string>
#include <cstdlib>
#include <sstream>

using namespace std;


// Set Gamma as a global constant
#define Gamma 100.0 

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



void read_pin_location( vector<double> &pin_x , vector<double> &pin_y ){
  int pin_num;
  double x,y,w,h;
  ifstream fptr;
  vector<double> x_v;
  vector<double> y_v;

  //char buff[100];
  //sprintf( buff, "%s_gpu_density_info", testcase_name[ID]);
  //string tname(buff);
  string tname(testcase_name[ID]);
  tname += "_gpu_density_info";
  cout << "GPU density Info = " << tname << '\n';


  //fptr.open("./tune_placer/gpu_density_info");
  fptr.open(tname.c_str());

  double Gx, Gy;
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




int main( int argc, char* argv[] ){
  if(argc < 2 ){
    printf("Error : No testcase ID\n");
    exit(1);
  }
  istringstream ss(argv[1]);
  ss >> ID;
  printf("Argv[1] (ID) = %d\n" , ID);

  //char buff[100];
  //sprintf( buff, "%s_gpu_wire_info", testcase_name[ID]);
  //string tname( buff );
  string tname(testcase_name[ID]);
  tname += "_gpu_wire_info";
  cout << "Wire info = " << tname << '\n';

  ofstream of;
  ifstream fptr;
  fptr.open(tname.c_str());
  vector<vector<int>> wire;
  vector<double> pin;
  size_t wire_num;
  size_t pin_num;
  fptr >> wire_num >> pin_num;
  wire.resize(wire_num);
  pin.resize(pin_num);
  
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
  cout << "Wire info parse Done\n";


  vector<double> pin_x;
  vector<double> pin_y;
  read_pin_location( pin_x, pin_y );



  auto begin_t = std::chrono::high_resolution_clock::now();
  auto end_t = std::chrono::high_resolution_clock::now();


  vector<double> exp_x;
  vector<double> exp_x_neg;
  vector<double> exp_y;
  vector<double> exp_y_neg;

  exp_x.resize( wire.size() , 0.0 );
  exp_x_neg.resize( wire.size() , 0.0 );
  exp_y.resize( wire.size() , 0.0 );
  exp_y_neg.resize( wire.size() , 0.0 );

  // Compute exponential values
  begin_t = std::chrono::high_resolution_clock::now();
  for( size_t i = 0 ; i < wire.size() ; ++ i ){
    for( size_t j = 0 ; j < wire[i].size() ; ++ j ){
      exp_x[i]     += exp(pin_x[wire[i][j]]/Gamma);
      exp_x_neg[i] += exp(-pin_x[wire[i][j]]/Gamma);
      exp_y[i]     += exp(pin_y[wire[i][j]]/Gamma);
      exp_y_neg[i] += exp(-pin_y[wire[i][j]]/Gamma);
    }
  }
  end_t = std::chrono::high_resolution_clock::now();
  std::cout << "Exp RunTime : " << std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << " micro sec" << std::endl;
  auto total_run_time = std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();

 
  vector<vector<int>> wireInPin;
  wireInPin.resize( pin_num );
  for( size_t i = 0 ; i < wire.size() ; ++ i )
    for( size_t j = 0 ; j < wire[i].size() ; ++ j )
      wireInPin[ wire[i][j] ].push_back( i );


  vector<double> x;
  vector<double> x_neg;
  vector<double> y;
  vector<double> y_neg;
  x.resize( pin_num, 0.0 );
  y.resize( pin_num, 0.0 );
  x_neg.resize( pin_num, 0.0 );
  y_neg.resize( pin_num, 0.0 );



  vector<double> grad_x;
  vector<double> grad_y;
  grad_x.resize( pin_num, 0.0 );
  grad_y.resize( pin_num, 0.0 );

  // Compute gradient
  begin_t = std::chrono::high_resolution_clock::now();
  for( size_t i = 0 ;  i < pin_num ; ++ i ){
    for( size_t j = 0 ; j < wireInPin[i].size() ; ++ j ){
      x[i] += 1.0/exp_x[ wireInPin[i][j] ];
      x_neg[i] -= 1.0/exp_x_neg[ wireInPin[i][j] ];
      y[i] += 1.0/exp_y[ wireInPin[i][j] ];
      y_neg[i] -= 1.0/exp_y_neg[ wireInPin[i][j] ];
    }
    grad_x[i] = x[i]*exp(pin_x[i]/Gamma) + x_neg[i]*exp(-pin_x[i]/Gamma);
    grad_y[i] = y[i]*exp(pin_y[i]/Gamma) + y_neg[i]*exp(-pin_y[i]/Gamma);
  }
  end_t = std::chrono::high_resolution_clock::now();
  std::cout << "Grad RunTime : " << std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count() << " micro sec" << std::endl;

  total_run_time += std::chrono::duration_cast<std::chrono::microseconds>(end_t-begin_t).count();
  std::cout << "Total CPU Run time = " << total_run_time << " micro sec" << std::endl;


  // This is for debugging
  //of.open("./cpu_result");
  //of.setf(ios::fixed,ios::floatfield);
  //of.precision(12);

  //for( int i = 0 ; i < pin_num ; ++ i ){
  //  if( i < 10 ){
  //    cout << grad_x[i] << " " << grad_y[i] << '\n';
  //  }
  //  of << grad_x[i] << '\n';
  //}
  //of.close();

}


