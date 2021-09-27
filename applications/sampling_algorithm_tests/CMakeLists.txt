CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)


ADD_EXECUTABLE(sampler_test_uniformproposal_unifromprior_TEMonly
        sampler_test_uniformproposal_unifromprior_TEMonly.cpp)
TARGET_LINK_LIBRARIES(sampler_test_uniformproposal_unifromprior_TEMonly
        libmepbm)
