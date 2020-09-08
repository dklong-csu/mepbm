#ifndef DATA_H_
#define DATA_H_

#include <valarray>



namespace Data
{
  // This class holds the data for the Ir-POM system
  class PomData
  {
  public:
    const std::valarray<double> chcrr_concentration, chcrr_time;
    const std::valarray<double> tem_diam_time1, tem_diam_time2, tem_diam_time3, tem_diam_time4;
    const double tem_time1, tem_time2, tem_time3, tem_time4;

    // constructor
    PomData();
  };
}


#endif /* DATA_H_ */
