#ifndef MATRIX_MULTIPLIER_HPP_
#define MATRIX_MULTIPLIER_HPP_

namespace strassen
{
  /**
   * A matrix_multiplier object performs matrix multiplication on two arrays with given row and column
   * bounds, representing matrices.
   */
  template <typename T>
  class matrix_multiplier
  {
  public:    
    virtual T* mult (const T *a, const T *b, size_t arows, size_t acols, size_t brows, size_t bcols) = 0;
    virtual matrix_multiplier<T>* copy () const = 0;
    virtual ~matrix_multiplier<T>() {}
  };
}

#endif /* MATRIX_MULTIPLIER_HPP_ */
