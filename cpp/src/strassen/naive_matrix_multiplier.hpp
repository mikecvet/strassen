#ifndef NAIVE_MATRIX_MULTIPLIER_HPP_
#define NAIVE_MATRIX_MULTIPLIER_HPP_

#include "matrix_multiplier.hpp"

namespace strassen
{
  /**
   * A naive_matrix_multiplier multiples two given matrices using the naive O(n^3) multiplication algorithm.
   */
  template <typename T>
  class naive_matrix_multiplier : public strassen::matrix_multiplier<T>
  {
  public:
    naive_matrix_multiplier ();
    virtual ~naive_matrix_multiplier ();
    
    T* mult (const T *a, const T *b, size_t arows, size_t acols, size_t brows, size_t bcols);
    matrix_multiplier<T>* copy () const;
  };

  template <typename T>
  naive_matrix_multiplier<T>::naive_matrix_multiplier ()
  {
  }

  template <typename T>
  naive_matrix_multiplier<T>::~naive_matrix_multiplier ()
  {
  }

  template <typename T>
  matrix_multiplier<T>*
  naive_matrix_multiplier<T>::copy () const
  {
    return (new naive_matrix_multiplier<T> ());
  }

  template <typename T>
  T*
  naive_matrix_multiplier<T>::mult (const T *A, const T *B,
                                    size_t arows, size_t acols,
                                    size_t brows, size_t bcols)
  {
    if (arows == bcols)
      {
        T t;
        size_t m = arows;
        size_t n = acols;
        size_t im;	
        
        T *C = (T *) malloc (m * m * sizeof (T));
        const T *a_row = NULL;

        for (size_t i = 0; i < m; i++)
          {
            im = i * m;
            a_row = &A[i * acols];	    

            for (size_t j = 0; j < m; j++)
              {
                t = 0;
                
                for (size_t k = 0; k < n; k++)
                  {
                    t += (a_row[k] * B[(k * bcols) + j]);
                  }

                C[im++] = t;
              }
          }
        
          return C;
      }
    else
      {
        fprintf (stderr, "a.rows %lu != b.cols%lu\n", arows, bcols);
        exit (1);
      }

    return NULL;
  }
}

#endif /* NAIVE_MATRIX_MULTIPLIER_HPP_ */
