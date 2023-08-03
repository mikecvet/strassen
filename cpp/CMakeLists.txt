PROJECT (STRASSEN)

INCLUDE_DIRECTORIES(${STRASSEN}/src)

FIND_LIBRARY(MATH m)
FIND_LIBRARY(PTHREAD pthread)

SET(CMAKE_CXX_FLAGS "-O2 -rdynamic -fforce-addr -march=native -Wall")

ADD_EXECUTABLE(test_strassen_matrix
  src/util/timer.cpp
  src/test/test_strassen_matrix.cpp
  )

TARGET_LINK_LIBRARIES(test_strassen_matrix ${MATH} ${PTHREAD})
