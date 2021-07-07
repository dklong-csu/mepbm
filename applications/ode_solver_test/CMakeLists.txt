CMAKE_MINIMUM_REQUIRED (VERSION 3.1)

FIND_PACKAGE (Threads)

ADD_EXECUTABLE(eigen_fast_test
        eigen_fast.cpp)
TARGET_LINK_LIBRARIES(eigen_fast_test
        libmepbm)

ADD_EXECUTABLE(eigen_accurate_test
        eigen_accurate.cpp)
TARGET_LINK_LIBRARIES(eigen_accurate_test
        libmepbm)

ADD_EXECUTABLE(sundials_dense_test
        sundials_dense.cpp)
TARGET_LINK_LIBRARIES(sundials_dense_test
        libmepbm)

ADD_EXECUTABLE(sundials_spbcgs_test
        sundials_spbcgs.cpp)
TARGET_LINK_LIBRARIES(sundials_spbcgs_test
        libmepbm)

ADD_EXECUTABLE(sundials_spfgmr_test
        sundials_spfgmr.cpp)
TARGET_LINK_LIBRARIES(sundials_spfgmr_test
        libmepbm)

ADD_EXECUTABLE(sundials_spgmr_test
        sundials_spgmr.cpp)
TARGET_LINK_LIBRARIES(sundials_spgmr_test
        libmepbm)

ADD_EXECUTABLE(sundials_sptfqmr_test
        sundials_sptfqmr.cpp)
TARGET_LINK_LIBRARIES(sundials_sptfqmr_test
        libmepbm)