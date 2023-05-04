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
  /**
   * The matrix class is essentially a wrapper around an array of type T which maintains row and column
   * information. Various matrix operations are defined. The work of actually multiplying two matrices
   * is done by the matrix_multiplier<T> field present in the class. This defaults to a 
   * strassen::strassen_matrix_multiplier<T> unless specified otherwise.
   */
  template <typename T>
  class matrix
  {
    friend class iterator;
    
  private:
    size_t _rows;   /* Number of rows in our matrix */
    size_t _cols;   /* Number of columns in our matrix */
    T *__matrix;    /* Our actual matrix data */

    /* A matrix_multiplier performs one of several matrix multiplication algorithms */
    strassen::matrix_multiplier<T> *__mm;

    /* Returns a reference to the matrix element (i, j) */
    T& __at (size_t i, size_t j);

    /* Scalar multiplication */
    void __mult (T *a, size_t arows, size_t acols, T k);

    /* Adds the given matrix to this matrix */
    void __add (T *A, size_t arows, size_t acols, const matrix<T> &b);

    /* Subtracts the given matrix from this matrix */
    void __sub (T *A, size_t arows, size_t acols, const matrix<T> &b);
    
    /* Determines equality */
    bool __equal (const matrix<T> &m);

  public:
    /* Declare a new, empty matrix */
    matrix (matrix_multiplier<T> *mm = new strassen_matrix_multiplier<T> ());
    /* Declare a new matrix with dimensions defined */
    matrix (size_t h, size_t w, matrix_multiplier<T> *mm = new strassen_matrix_multiplier<T> ());
    matrix (const matrix<T> &m);
    ~matrix ();

    /* Clear the contents of this matrix */
    void clear ();
    /* Initialize this matrix to zero */
    void zeroes ();
    /* Initialize this matrix with random numbers bounded by the parameter; max = 0 means no bound */
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

    /* Iterator class allows iteration over the internal matrix data structures */
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

  /**
   * Initialize this matrix to random values. Values are bounded by the max parameter
   * unless it is zero, in which case they are not bounded.
   */
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

  /**
   * Return a new array containing a copy of the data in our matrix.
   */
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

  /**
   * Main matrix multiplication function.
   * 
   * Given another matrix m, use the __mm object to multiply this matrix by m. Depending on 
   * how this object was constructed, it may perform one of naive, transpose-naive, strassen, or
   * parallel-strassen multiplication algorithms. Returns a new array of type T. If the return
   * is NULL, something went wrong so nothing will be changed.
   */
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
    mult (m);
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
