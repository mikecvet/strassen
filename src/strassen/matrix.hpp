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

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transpose_matrix_multiplier.hpp"
#include "strassen_matrix_multiplier.hpp"

namespace strassen
{
  template <typename T>
  class matrix
  {
    friend class iterator;
    
  private:
    size_t _rows;
    size_t _cols;
    T *__matrix;

    strassen::matrix_multiplier<T> *__mm;

    T& __at (size_t i, size_t j);
    void __mult (T *a, size_t arows, size_t acols, T k);
    void __mult (T *A, size_t arows, size_t acols, const matrix<T> &b);
    void __add (T *A, size_t arows, size_t acols, const matrix<T> &b);
    void __sub (T *A, size_t arows, size_t acols, const matrix<T> &b);
    
    bool __equal (const matrix<T> &m);

  public:
    matrix (matrix_multiplier<T> *mm = new strassen_matrix_multiplier<T> ());
    matrix (size_t h, size_t w, matrix_multiplier<T> *mm = new strassen_matrix_multiplier<T> ());
    matrix (const matrix<T> &m);
    ~matrix ();

    void clear ();
    void zeroes ();
    void random (uint32_t max = 0);

    T& at (size_t i, size_t j);
    void mult (T k);
    void mult (const matrix<T> &m);
    void add (const matrix<T> &m);
    void sub (const matrix<T> &m);
    bool equal (const matrix<T> &m);

    size_t rows () const;
    size_t cols () const;
    T* raw_data_copy () const;

    T& operator () (size_t i, size_t j);
    matrix<T>& operator * (T k);
    matrix<T>& operator * (const matrix<T> &m);
    matrix<T>& operator + (const matrix<T> &m);
    matrix<T>& operator - (const matrix<T> &m);
    matrix<T>& operator = (const matrix<T> &m);
    bool operator == (const matrix<T> &m);

    class iterator
    {
    private:
      T *__m;
      size_t __rows;
      size_t __cols;
      size_t __indx;
      size_t __n;

    public:
      iterator (matrix<T> &m);
      ~iterator ();

      T val ();
      size_t row ();
      size_t col ();
      bool ok ();
      void operator ++ ();      
    };
  };

  template <typename T>
  matrix<T>::matrix (matrix_multiplier<T> *mm)
    : _rows (0),
      _cols (0),
      __matrix (NULL),
      __mm (mm)
  {    
  }
  
  template <typename T>
  matrix<T>::matrix (size_t rows, size_t cols, matrix_multiplier<T> *mm)
    : _rows (rows),
      _cols (cols),
      __mm (mm)
  {
    __matrix = (T *) malloc (_rows * _cols * sizeof (T));
  }
  
  template <typename T>
  matrix<T>::matrix (const matrix<T> &m)
    : _rows (m.rows ()),
      _cols (m.cols ()),
      __mm (m.__mm->copy ())
  {
    __matrix = m.raw_data_copy ();
  }
  
  template <typename T>
  matrix<T>::~matrix ()
  {
    if (__matrix)
      {
	free (__matrix);
	__matrix = NULL;
	delete __mm;
      }
  }

  template <typename T>
  void
  matrix<T>::clear ()
  {
    if (__matrix)
      {
	_rows = 0;
	_cols = 0;
	
	free (__matrix);
	__matrix = NULL;
      }
  }

  template <typename T>
  T&
  matrix<T>::at (size_t i, size_t j)
  {
    return (__at (i, j));
  }

  template <typename T>
  void
  matrix<T>::zeroes ()
  {
    memset (__matrix, 0, _rows * _cols * sizeof (T));
  }

  template <typename T>
  void
  matrix<T>::random (uint32_t max)
  {
    size_t n = _rows * _cols;

    srand (time (NULL));

    if (!max)
      {
	for (size_t i = 0; i < n; i++)
	  __matrix[i] = rand ();
      }
    else
      {
	for (size_t i = 0; i < n; i++)
	  __matrix[i] = (rand () % max) + 1;
      }
  }

  template <typename T>
  size_t
  matrix<T>::rows () const
  {
    return _rows;
  }

  template <typename T>
  size_t
  matrix<T>::cols () const
  {
    return _cols;
  }

  template <typename T>
  T*
  matrix<T>::raw_data_copy () const
  {
    T *t = (T *) malloc (_rows * _cols * sizeof (T));
    memcpy (t, __matrix, (_rows * _cols * sizeof (T)));

    return t;
  }

  template <typename T>
  void
  matrix<T>::mult (T k)
  {
    __mult (__matrix, _rows, _cols, k);
  }

  template <typename T>
  void
  matrix<T>::mult (const matrix<T> &m)
  { 
    T *C = __mm -> mult (__matrix, m.__matrix, _rows, _cols, m.rows (), m.cols ());

    if (C)
      {
	free (__matrix);
	
	__matrix = C; 
	_cols = _rows;
      }
  }

  template <typename T>
  void
  matrix<T>::add (const matrix<T> &m)
  {
    __add (__matrix, _rows, _cols, m);
  }

  template <typename T>
  void
  matrix<T>::sub (const matrix<T> &m)
  {
    __sub (__matrix, _rows, _cols, m);
  }

  template <typename T>
  bool
  matrix<T>::equal (const matrix<T> &m)
  {
    return (__equal (m));
  }

  template <typename T>
  T&
  matrix<T>::operator () (size_t i, size_t j)
  {
    return (__at (i , j));
  }

  template <typename T>
  matrix<T>&
  matrix<T>::operator * (T k)
  {
    __mult (__matrix, _rows, _cols, k);
    return (*this);
  }

  template <typename T>
  matrix<T>&
  matrix<T>::operator * (const matrix<T> &m)
  {
    __mult (__matrix, _rows, _cols, m);
    return (*this);
  }

  template <typename T>
  matrix<T>&
  matrix<T>::operator + (const matrix<T> &m)
  {
    __add (__matrix, _rows, _cols, m);
    return (*this);
  }
  
  template <typename T>
  matrix<T>&
  matrix<T>::operator - (const matrix<T> &m)
  {
    __sub (__matrix, _rows, _cols, m);
    return (*this);
  }
  
  template <typename T>
  matrix<T>&
  matrix<T>::operator = (const matrix<T> &m)
  {
    if (__matrix)
      free (__matrix);

    _rows = m.rows ();
    _cols = m.cols ();
    __matrix = m.raw_data_copy ();

    return (*this);
  }

  template <typename T>
  bool
  matrix<T>::operator == (const matrix<T> &m)
  {
    return (__equal (m));
  }

  template <typename T>
  T&
  matrix<T>::__at (size_t i, size_t j)
  {
    return (__matrix[(i * _cols) + j]);
  }

  template <typename T>
  void
  matrix<T>::__mult (T *A, size_t arows, size_t acols, T k)
  {
    size_t n = arows * acols;
    
    for (size_t i = 0; i < n; i++)
      A[i] = A[i] * k;
  }

  template <typename T>
  void
  matrix<T>::__add (T *A, size_t arows, size_t acols, const matrix<T> &b)
  {
    if (_rows == b.rows () && _cols == b.cols ())
      {
	size_t n = _rows * _cols;
	T *B = b.__matrix;

	for (size_t i = 0; i < n; i++)
	  __matrix[i] = A[i] + B[i];
      }
    //else throw exception
  }

  template <typename T>
  void
  matrix<T>::__sub (T *A, size_t arows, size_t acols, const matrix<T> &b)
  {
    if (_rows == b.rows () && _cols == b.cols ())
      {
	size_t n = _rows * _cols;
	T *B = b.__matrix;

	for (size_t i = 0; i < n; i++)
	  __matrix[i] = A[i] - B[i];
      }
    //else throw exception
  }

  template <typename T>
  bool
  matrix<T>::__equal (const matrix<T> &m)
  {
    if (_rows == m.rows () && _cols == m.cols ())
      {
	size_t n = _rows * _cols;
	T *B = m.__matrix;

	for (size_t i = 0; i < n; i++)	  
	  {
	    if (__matrix[i] != B[i])
	      {
		fprintf (stderr, "failure: %d\n",__matrix[i]);
		return false;
	      }
	  }
	
	return true;
      }
    else
      return false;
  }

  template <typename T>
  matrix<T>::iterator::iterator (matrix<T> &m)
    : __m (m.__matrix),
      __rows (m.rows ()),
      __cols (m.cols ()),
      __indx (0)
  {
    __n = __rows * __cols;
  }

  template <typename T>
  matrix<T>::iterator::~iterator ()
  {    
  }

  template <typename T>
  T
  matrix<T>::iterator::val ()
  {
    return (__m[__indx]);
  }

  template <typename T>
  size_t
  matrix<T>::iterator::row ()
  {
    return (__indx / __rows);
  }

  template <typename T>
  size_t
  matrix<T>::iterator::col ()
  {
    return (__indx % __cols);
  }

  template <typename T>
  bool
  matrix<T>::iterator::ok ()
  {
    return (__indx < __n);
  }

  template <typename T>
  void
  matrix<T>::iterator::operator ++ ()
  {
    ++__indx;
  }
}

#endif /* MATRIX_HPP_ */
