#include <iostream>
#include <string>
#include <vector>
#include <typeinfo>
#include "src/ir_pom_data.h"



using Real = float;



void print_array(std::vector<Real> array)
{
  for (unsigned int i = 0; i < array.size(); i++)
  {
    std::cout << array[i]
              << std::endl;
  }
}



int main ()
{
  // create data
  MEPBM::PomData<Real> my_data;

  // check that each is initialized properly
  print_array(my_data.chcrr_concentration);
  std::cout << std::endl;
  print_array(my_data.chcrr_time);
  std::cout << std::endl;

  print_array(my_data.tem_diam_time1);
  std::cout << std::endl;
  std::cout << my_data.tem_time1 << std::endl;
  std::cout << std::endl;

  print_array(my_data.tem_diam_time2);
  std::cout << std::endl;
  std::cout << my_data.tem_time2 << std::endl;
  std::cout << std::endl;

  print_array(my_data.tem_diam_time3);
  std::cout << std::endl;
  std::cout << my_data.tem_time3 << std::endl;
  std::cout << std::endl;

  print_array(my_data.tem_diam_time4);
  std::cout << std::endl;
  std::cout << my_data.tem_time4 << std::endl;
}
