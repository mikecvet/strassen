/**
 *Copyright (C) 2011 by Michael Cvet <http://mikecvet.wordpress.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 **/

#ifndef TRANSPOSE_MATRIX_MULTIPLIER_HPP_
#define TRANSPOSE_MATRIX_MULTIPLIER_HPP_

#include "matrix_multiplier.hpp"

namespace strassen
{
  /**
   * A transpose_matrix_multiplier multiplies two given matrices using the naive multiplication algorithm,
   * with an optimization. Instead of iterating over the rows of one matrix and the columns of the other,
   * take the transpose of the second matrix and perform a row-by-row multiplication. This greatly improves
   * multiplication performance because better cache usage.
   */
  template <typename T>
  class transpose_matrix_multiplier : public strassen::matrix_multiplier<T>
  {
  public:
    transpose_matrix_multiplier ();
    ~transpose_matrix_multiplier ();
    
    T* mult (const T *a, const T *b, size_t arows, size_t acols, size_t brows, size_t bcols);
    T* transpose (const T *A, size_t rows, size_t cols);
    matrix_multiplier<T>* copy () const;
  };

  template <typename T>
  transpose_matrix_multiplier<T>::transpose_matrix_multiplier ()
  {
  }

  template <typename T>
  transpose_matrix_multiplier<T>::~transpose_matrix_multiplier ()
  {
  }

  template <typename T>
  matrix_multiplier<T>*
  transpose_matrix_multiplier<T>::copy () const
  {
    return (new transpose_matrix_multiplier<T> ());
  }

  template <typename T>
  T*
  transpose_matrix_multiplier<T>::mult (const T *A, const T *b,
					size_t arows, size_t acols,
					size_t brows, size_t bcols)
  {
    if (arows == bcols)
      {
        T t;
        size_t m = arows;
        size_t n = acols;
        size_t im;
        size_t tmp = brows;
        brows = bcols;
        bcols = tmp;
        
        /* B is a malloc'd array containing the transpose of the matrix represented by b. */
        T *B = transpose (b, brows, bcols);
        T *C = (T *) malloc (m * m * sizeof (T));
        const T *a_row = NULL;
        T *b_row = NULL;
	
        for (size_t i = 0; i < m; i++)
          {
            im = i * m;
            a_row = &A[i * acols];	    

            for (size_t j = 0; j < m; j++)
              {
                t = 0;
                b_row = &B[j * bcols];
                
                for (size_t k = 0; k < n; k++)
                  {
                    t += (a_row[k] * b_row[k]);
                  }

                C[im++] = t;
              }
          }
        
        free (B);
        return C;
      }
    else
      arows = 0;

    return NULL;
  }

  template <typename T>
  T*
  transpose_matrix_multiplier<T>::transpose (const T *A, size_t rows, size_t cols)
  {    
    if (A)
      {
        T *row = NULL;
        T *m = (T *) malloc (rows * cols * sizeof (T));
        
        for (size_t i = 0; i < rows; i++)
          {
            row = &m[i * cols];
            
            for (size_t j = 0; j < cols; j++)
              {
                row[j] = A[j * rows + i];
              }
          }

        return m;
      }

    return NULL;    
  }
}

#endif /* TRANSPOSE_MATRIX_MULTIPLIER_HPP_ */
