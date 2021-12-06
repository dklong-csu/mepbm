#include "chemical_reaction.h"
#include <iostream>

/*
 * This tests the member variables of the species class.
 */

int main ()
{
  // The constructor provides the index of the Species for accessing the correct element of a vector.
  Model::Species my_chemical(3);

  // Make sure the index is 3
  std::cout << std::boolalpha << (my_chemical.index == 3) << std::endl;
}