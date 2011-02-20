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

#ifndef STRASSEN_MATRIX_MULTIPLIER_HPP_
#define STRASSEN_MATRIX_MULTIPLIER_HPP_

#include <cmath>
#include "matrix_multiplier.hpp"

namespace strassen
{
  template <typename T>
  class strassen_matrix_multiplier : public strassen::matrix_multiplier<T>
  {
  protected:
    static double __log2;
    transpose_matrix_multiplier<T> __tmm;

    T* __pad   (const T *m, size_t rows, size_t cols, size_t n);
    T* __unpad (const T *m, size_t rows, size_t cols, size_t n);
    T* __mult  (const T *A, const T *B, size_t n);

    bool __zeroes (const T *A, size_t n);

    void __submatrix_add (T *C, const T *A,
			  size_t a_row_start, size_t a_col_start, 
			  size_t b_row_start, size_t b_col_start, 
			  size_t m, size_t n);    

    void __submatrix_sub (T *C, const T *A, 
			  size_t a_row_start, size_t a_col_start, 
			  size_t b_row_start, size_t b_col_start, 
			  size_t m, size_t n);
    
    void __submatrix_cpy (T *C, const T *A, size_t start, size_t end, size_t m, size_t n);

    void __submatrix_add (T *C, const T *A, size_t row_start, size_t col_start, size_t m, size_t n);
    void __submatrix_add (T *C, const T *A, const T *B, size_t row_start, size_t col_start, size_t m, size_t n);
    void __submatrix_sub (T *C, const T *A, size_t row_start, size_t col_start, size_t m, size_t n);
    void __submatrix_sub (T *C, const T *A, const T *B, size_t row_start, size_t col_start, size_t m, size_t n);
    
  public:
    strassen_matrix_multiplier ();
    virtual ~strassen_matrix_multiplier ();
    
    virtual T* mult (const T *a, const T *b, size_t arows, size_t acols, size_t brows, size_t bcols);
    virtual matrix_multiplier<T>* copy () const;
  };

  template <typename T>
  double strassen_matrix_multiplier<T>::__log2 = std::log (2.0);  

  template <typename T>
  strassen_matrix_multiplier<T>::strassen_matrix_multiplier ()
  {
  }

  template <typename T>
  strassen_matrix_multiplier<T>::~strassen_matrix_multiplier ()
  {
  }

  template <typename T>
  matrix_multiplier<T>*
  strassen_matrix_multiplier<T>::copy () const
  {
    return (new strassen_matrix_multiplier<T> ());
  }

  template <typename T>
  T*
  strassen_matrix_multiplier<T>::mult (const T *m, const T *n,
				       size_t arows, size_t acols,
				       size_t brows, size_t bcols)
  {
    if (acols == brows)
      {
	if (arows == acols && brows == bcols && !(arows & (arows - 1)))
	  {
	    T *C = __mult (m, n, arows);
	    return C;
	  }
	else
	  {
	    size_t N;
	    size_t max_term = acols;

	    T *A;
	    T *B;

	    if (arows >= acols && arows >= brows)
	      max_term = arows;
	    else if (acols >= arows && acols >= bcols)
	      max_term = acols;
	    else if (brows >= bcols && brows >= arows)
	      max_term = brows;
	    else if (bcols >= brows && bcols >= acols)
	      max_term = bcols;
	    
	    N = std::pow (2, (size_t) (std::log (max_term) / __log2) + 1);
	    
	    A = __pad (m, arows, acols, N);
	    B = __pad (n, brows, bcols, N);

	    T *C = __mult (A, B, N);
	    T *D = __unpad (C, arows, arows, N);
	    
	    free (A);
	    free (B);
	    free (C);
	    
	    return D;
	  }
      }

    return NULL;
  }

  template <typename T>
  T*
  strassen_matrix_multiplier<T>::__mult (const T *A, const T *B, size_t n)
  {
    if (n <= 128)
      {
	return (__tmm.mult (A, B, n, n, n, n));
      }

    size_t m = n / 2;

    size_t tl_row_start = 0;
    size_t tl_col_start = 0;

    size_t tr_row_start = 0;
    size_t tr_col_start = m;

    size_t bl_row_start = m;
    size_t bl_col_start = 0;

    size_t br_row_start = m;
    size_t br_col_start = m;

    T *C = (T *) malloc (n * n * sizeof (T));

    T* AA[7];
    T* BB[7];
    T* MM[7];

    if ((!A[0] && !A[1] && __zeroes (A, n)) || (!B[0] && !B[1] && __zeroes (B, n)))
      {
	memset (C, 0, n * n * sizeof (T));
	return C;
      }
    
    for (uint32_t i = 0; i < 7; i++)
      {
	AA[i] = (T *) malloc (m * m * sizeof (T));
	BB[i] = (T *) malloc (m * m * sizeof (T));
      }

    __submatrix_add (AA[0], A, tl_row_start, tl_col_start, br_row_start, br_col_start, m, n);
    __submatrix_add (AA[1], A, bl_row_start, bl_col_start, br_row_start, br_row_start, m, n);
    __submatrix_cpy (AA[2], A, tl_row_start, tl_col_start, m, n);
    __submatrix_cpy (AA[3], A, br_row_start, br_col_start, m, n);
    __submatrix_add (AA[4], A, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m, n);
    __submatrix_sub (AA[5], A, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m, n);
    __submatrix_sub (AA[6], A, tr_row_start, tr_col_start, br_row_start, br_col_start, m, n);

    __submatrix_add (BB[0], B, tl_row_start, tl_col_start, br_row_start, br_col_start, m, n);
    __submatrix_cpy (BB[1], B, tl_row_start, tl_col_start, m, n);
    __submatrix_sub (BB[2], B, tr_row_start, tr_col_start, br_row_start, br_col_start, m, n);
    __submatrix_sub (BB[3], B, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m, n);
    __submatrix_cpy (BB[4], B, br_row_start, br_col_start, m, n);
    __submatrix_add (BB[5], B, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m, n); 
    __submatrix_add (BB[6], B, bl_row_start, bl_col_start, br_row_start, br_col_start, m, n);

    MM[0] = __mult (AA[0], BB[0], m);
    MM[1] = __mult (AA[1], BB[1], m);
    MM[2] = __mult (AA[2], BB[2], m);
    MM[3] = __mult (AA[3], BB[3], m);
    MM[4] = __mult (AA[4], BB[4], m);
    MM[5] = __mult (AA[5], BB[5], m);
    MM[6] = __mult (AA[6], BB[6], m);      

    __submatrix_add (C, MM[0], MM[3], tl_row_start, tl_col_start, m, n);
    __submatrix_sub (C, MM[4], tl_row_start, tl_col_start, m, n);
    __submatrix_add (C, MM[6], tl_row_start, tl_col_start, m, n);

    __submatrix_add (C, MM[2], MM[4], tr_row_start, tr_col_start, m, n);

    __submatrix_add (C, MM[1], MM[3], bl_row_start, bl_col_start, m,  n);

    __submatrix_sub (C, MM[0], MM[1], br_row_start, br_col_start, m, n);
    __submatrix_add (C, MM[2], br_row_start, br_col_start, m, n);
    __submatrix_add (C, MM[5], br_row_start, br_col_start, m, n);

    for (uint32_t i = 0; i < 7; i++)
      {
	free (AA[i]);
	free (BB[i]);
	free (MM[i]);
      }

    return C;
  }

  template <typename T>
  bool
  strassen_matrix_multiplier<T>::__zeroes (const T *A, size_t n)
  {
    size_t N = n * n;

    for (size_t i = 0; i < N; i++)
      {
	if (A[i])
	  return false;
      }

    return true;
  }

  template <typename T>
  void
  strassen_matrix_multiplier<T>::__submatrix_add (T *C, const T *A,
						  size_t a_row_start, size_t a_col_start, 
						  size_t b_row_start, size_t b_col_start, 
						  size_t m, size_t n)
  {
    size_t im;
    size_t a_row_indx;
    size_t b_row_indx;
    const T *a_row;
    const T *b_row;

    for (size_t i = 0; i < m; i++)
      {
	im = i * m;
	a_row_indx = ((a_row_start + i) * n) + a_col_start;
	b_row_indx = ((b_row_start + i) * n) + b_col_start;

	a_row = &A[a_row_indx];
	b_row = &A[b_row_indx];

	for (size_t j = 0; j < m; j++)
	  {
	    C[im + j] = a_row[j] + b_row[j];
	  }
      }
  }

  template <typename T>
  void
  strassen_matrix_multiplier<T>::__submatrix_add (T *C, const T *A, 
						  size_t row_start, size_t col_start, 
						  size_t m, size_t n)
  {
    size_t im;
    size_t row_indx;
    T *row;

    for (size_t i = 0; i < m; i++)
      {
	im = i * m;
	row_indx = ((row_start + i) * n) + col_start;
	row = &C[row_indx];

	for (size_t j = 0; j < m; j++)
	  {
	    row[j] += A[im + j];
	  }
      }
  }

  template <typename T>
  void
  strassen_matrix_multiplier<T>::__submatrix_add (T *C, const T *A, const T *B, 
						  size_t row_start, size_t col_start, 
						  size_t m, size_t n)
  {
    size_t im;
    size_t row_indx;
    T *row;

    for (size_t i = 0; i < m; i++)
      {
	im = i * m;
	row_indx = ((row_start + i) * n) + col_start;
	row = &C[row_indx];

	for (size_t j = 0; j < m; j++)
	  {
	    row[j] = A[im + j] + B[im + j];
	  }
      }
  }

  template <typename T>
  void
  strassen_matrix_multiplier<T>::__submatrix_sub (T *C, const T *A,
						  size_t a_row_start, size_t a_col_start, 
						  size_t b_row_start, size_t b_col_start, 
						  size_t m, size_t n)
  {
    size_t im;
    size_t a_row_indx;
    size_t b_row_indx;
    const T *a_row;
    const T *b_row;

    for (size_t i = 0; i < m; i++)
      {
	im = i * m;
	a_row_indx = ((a_row_start + i) * n) + a_col_start;
	b_row_indx = ((b_row_start + i) * n) + b_col_start;

	a_row = &A[a_row_indx];
	b_row = &A[b_row_indx];

	for (size_t j = 0; j < m; j++)
	  {
	    C[im + j] = a_row[j] - b_row[j];
	  }
      }
  }
  
  template <typename T>
  void
  strassen_matrix_multiplier<T>::__submatrix_cpy (T *C, const T *A,
						  size_t row_start, size_t col_start, 
						  size_t m, size_t n)
  {
    size_t im;
    size_t row_indx;
    const T *row;

    for (size_t i = 0; i < m; i++)
      {
	im = i * m;
	row_indx = ((row_start + i) * n) + col_start;

	row = &A[row_indx];

	for (size_t j = 0; j < m; j++)
	  {
	    C[im + j] = row[j];
	  }
      }
  } 

  template <typename T>
  void
  strassen_matrix_multiplier<T>::__submatrix_sub (T *C, const T *A, 
						  size_t row_start, size_t col_start, 
						  size_t m, size_t n)
  {
    size_t im;
    size_t row_indx;
    T *row;

    for (size_t i = 0; i < m; i++)
      {
	im = i * m;
	row_indx = ((row_start + i) * n) + col_start;
	row = &C[row_indx];

	for (size_t j = 0; j < m; j++)
	  {
	    row[j] -= A[im + j];
	  }
      }
  }
 
  template <typename T>
  void
  strassen_matrix_multiplier<T>::__submatrix_sub (T *C, const T *A, const T *B, 
						  size_t row_start, size_t col_start, 
						  size_t m, size_t n)
  {
    size_t im;
    size_t row_indx;
    T *row;

    for (size_t i = 0; i < m; i++)
      {
	im = i * m;
	row_indx = ((row_start + i) * n) + col_start;
	row = &C[row_indx];

	for (size_t j = 0; j < m; j++)
	  {
	    row[j] = A[im + j] - B[im + j];
	  }
      }
  }

  template <typename T>
  T*
  strassen_matrix_multiplier<T>::__pad (const T *m, size_t rows, size_t cols, size_t n)
  {
    size_t in;
    size_t ir;
    T *M = (T *) malloc (n * n * sizeof (T));
    
    for (size_t i = 0; i < rows; i++)
      {
	in = i * n;
	ir = i * rows;

	for (size_t j = 0; j < cols; j++)
	  {
	    M[in + j] = m[ir + j];
	  }
	
	for (size_t j = cols; j < n; j++)
	  {
	    M[in + j] = 0;
	  }
      }

    for (size_t i = rows; i < n; i++)
      {
	in = i * n;

	for (size_t j = 0; j < n; j++)
	  {
	    M[in + j] = 0;
	  }
      }

    return M;
  }
  
  
  template <typename T>
  T*
  strassen_matrix_multiplier<T>::__unpad (const T *m, size_t rows, size_t cols, size_t n)
  {
    size_t in;
    size_t ir;
    T *M = (T *) malloc (rows * cols * sizeof (T));

    for (size_t i = 0; i < rows; i++)
      {
	in = i * n;
	ir = i * rows;

	for (size_t j = 0; j < cols; j++)
	  {
	    M[ir + j] = m[in + j];
	  }
      }

    return M;
  }
}

#endif /* STRASSEN_MATRIX_MULTIPLIER_HPP_ */
