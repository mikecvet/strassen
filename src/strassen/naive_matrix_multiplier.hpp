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

#ifndef NAIVE_MATRIX_MULTIPLIER_HPP_
#define NAIVE_MATRIX_MULTIPLIER_HPP_

#include "matrix_multiplier.hpp"

namespace strassen
{
  template <typename T>
  class naive_matrix_multiplier : public strassen::matrix_multiplier<T>
  {
  public:
    naive_matrix_multiplier ();
    ~naive_matrix_multiplier ();
    
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
		    t += (a_row[k] * B[(k * brows) + j]);
		  }

		C[im++] = t;
	      }
	  }
	
	return C;
      }
    else
      arows = 0;

    return NULL;
  }
}

#endif /* NAIVE_MATRIX_MULTIPLIER_HPP_ */
