#include <iostream>
#include <string>
#include <valarray>
#include <models.h>



int main()
{
  Models::TwoStep::Parameters prm(100, 10, 3, 5);
  Models::TwoStep model;

  std::valarray<double> state = { 1,0.5,0.3,0.2 };

  std::valarray<double> rhs_output = model.right_hand_side(state, prm);

  for (unsigned int i = 0; i < rhs_output.size(); i++)
  {
   std::cout << rhs_output[i]
             << std::endl;
  }
}
