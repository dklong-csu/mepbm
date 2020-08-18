#include <iostream>
#include <string>
#include <valarray>
#include <models.h>



int main()
{
  Models::TwoStepAlternative::Parameters prm(100, 70,60,40,2,3,5);
  Models::TwoStepAlternative model;

  std::valarray<double> state = { 1,.8,.6,.4,.2,.1 };

  std::valarray<double> rhs_output = model.right_hand_side(state, prm);

  for (unsigned int i = 0; i < rhs_output.size(); i++)
  {
   std::cout << rhs_output[i]
             << std::endl;
  }
}
