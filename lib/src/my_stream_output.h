#ifndef MEPBM_MY_STREAM_OUTPUT_H
#define MEPBM_MY_STREAM_OUTPUT_H

#include <sampleflow/consumer.h>


namespace MEPBM {
  /// Similar to the StreamOutput class in SampleFlow but outputs all data for each sample. Notably the log likelihood value.
  template<typename InputType>
  class MyStreamOutput : public SampleFlow::Consumer<InputType> {
  public:
    MyStreamOutput(std::ostream &output_stream)
        :
        output_stream(output_stream) {}

    ~MyStreamOutput() {
      this->disconnect_and_flush();
    }


    virtual
    void
    consume(InputType sample, SampleFlow::AuxiliaryData aux_data) override {
      std::lock_guard<std::mutex> lock(mutex);

      output_stream << "Sample: " << sample << std::endl;
      for (const auto &data: aux_data) {
        // Output the key of each pair:
        output_stream << "   " << data.first;

        // Then see if we can interpret the value via a known type:
        if (const bool *p = boost::any_cast<bool>(&data.second))
          output_stream << " -> " << (*p ? "true" : "false") << std::endl;
        else if (const double *p = boost::any_cast<double>(&data.second))
          output_stream << " -> " << *p << std::endl;
        else
          output_stream << std::endl;
      }
    }


  private:
    mutable std::mutex mutex;
    std::ostream &output_stream;
  };
}

#endif //MEPBM_MY_STREAM_OUTPUT_H
