#include <iostream>
#include <string>
#include <valarray>
#include <models.h>



int main()
{
  Models::ThreeStepAlternative::Parameters prm(100, 90, 80, 70, 60, 2, 3, 6, 4);
  Models::ThreeStepAlternative model;

  std::valarray<double> state = { 1,.9,.8,.7,.6,.5,.4 };


  std::valarray<double> rhs_output = model.right_hand_side(state, prm);

  for (unsigned int i = 0; i < rhs_output.size(); i++)
  {
  std::cout << rhs_output[i]
            << std::endl;
  }
}
