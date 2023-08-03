#ifndef STRASSEN_MATRIX_MULTIPLIER_HPP_
#define STRASSEN_MATRIX_MULTIPLIER_HPP_

#include <cmath>
#include "matrix_multiplier.hpp"

namespace strassen
{
  const size_t STRASSEN_THRESHOLD = 128;

  /**
   * The strassen_matrix_multiplier class multiplies two matrices over a given size together using the Strassen
   * Algorithm for matrix multiplication. 
   */
  template <typename T>
  class strassen_matrix_multiplier : public strassen::matrix_multiplier<T>
  {
  protected:
    static double __log2;

    /* A transpose_matrix_multiplier for use on submatrices with size below the STRASSEN_THRESOLD defined above */
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

  /**
   * Perform a strassen multiplication of the given two matrices. 
   */
  template <typename T>
  T*
  strassen_matrix_multiplier<T>::mult (const T *m, const T *n,
				       size_t arows, size_t acols,
				       size_t brows, size_t bcols)
  {
    /* Make sure this is a valid multiplication */
    if (acols == brows)
      {
        /* Check to see if these matrices are already square and have dimensions of a power of 2. If not,
        * the matrices must be resized and padded with zeroes to meet this criteria. */
        if (arows == acols && brows == bcols && !(arows & (arows - 1)))
          {
            T *C = __mult (m, n, arows);
            return C;
          }
        else
          {
            size_t N;
            size_t max_term = acols;

            T *A = NULL;
            T *B = NULL;
            T *C = NULL;

            if (arows >= acols && arows >= brows)
              max_term = arows;
            else if (acols >= arows && acols >= bcols)
              max_term = acols;
            else if (brows >= bcols && brows >= arows)
              max_term = brows;
            else if (bcols >= brows && bcols >= acols)
              max_term = bcols;
            
            /* Find the nearest power of 2 greater than the largest dimension of these matrices */
            N = std::pow (2, (size_t) (std::log (max_term) / __log2) + 1);
            
            /* If m needs padding, pad it */
            if (arows != acols || arows & (arows - 1))
              A = __pad (m, arows, acols, N);
            
            /* If n needs padding, pad it */
            if (brows != bcols || brows & (brows - 1))
              B = __pad (n, brows, bcols, N);

            /* __mult does the actual multiplication work */
            if (A && B)
              C = __mult (A, B, N);
            else if (A)
              C = __mult (A, n, N);
            else if (B)
              C = __mult (m, B, N);

            /* Extract the non-zero elements out of C and put them into a new matrix D which is 
            * of the size arows x bcols */
            T *D = __unpad (C, arows, arows, N);
            
            if (A)
              free (A);

            if (B)
              free (B);

            free (C);
            
            return D;
          }
      }

    return NULL;
  }

  /**
   * Performs the actual strassen multiplication.
   *
   * The Strassen algorithm works by breaking the given matrices A and B into submatrices and 
   * performing operations on those quadrants. This function will break apart A and B into those
   * submatrices and recursively multiply them together using the same method.
   */
  template <typename T>
  T*
  strassen_matrix_multiplier<T>::__mult (const T *A, const T *B, size_t n)
  {
    /* If the given matrices are small, its more efficient to use the transpose naive algorithm. */
    if (n <= STRASSEN_THRESHOLD)
      {
	      return (__tmm.mult (A, B, n, n, n, n));
      }

    size_t m = n / 2;

    /* Top left submatrix */
    size_t tl_row_start = 0;
    size_t tl_col_start = 0;

    /* Top right submatrix */
    size_t tr_row_start = 0;
    size_t tr_col_start = m;

    /* Bottom left submatrix */
    size_t bl_row_start = m;
    size_t bl_col_start = 0;

    /* Bottom right submatrix */
    size_t br_row_start = m;
    size_t br_col_start = m;

    /* The output matrix */
    T *C = (T *) malloc (n * n * sizeof (T));

    T* AA[7]; /* Submatrix blocks for A */
    T* BB[7]; /* Submatrix blocks for B */
    T* MM[7]; /* Products of above submatrices */

    /* Make sure that neither A or B consist entirely of zeroes. If so, easy; nullify the
     * contents of C and return. */
    if ((!A[0] && !A[1] && __zeroes (A, n)) || (!B[0] && !B[1] && __zeroes (B, n)))
      {
        memset (C, 0, n * n * sizeof (T));
        return C;
      }
    
    /* Make room for the submatrices */
    for (uint32_t i = 0; i < 7; i++)
      {
        AA[i] = (T *) malloc (m * m * sizeof (T));
        BB[i] = (T *) malloc (m * m * sizeof (T));
      }

    /*
     * The output matrix C is expressed in terms of the block matrices M1..M7
     *
     * C1,1 = M1 + M4 - M5 + M7
     * C1,2 = M3 + M5
     * C2,1 = M2 + M4
     * C2,2 = M1 - M2 + M3 + M6
     * 
     * Each of the block matrices M1..M7 is composed of quadrants from A and B as follows:
     * 
     * M1 = AA[0] * BB[0] = (A1,1 + A2,2)(B1,1 + B2,2)
     * M2 = AA[1] * BB[1] = (A2,1 + A2,2)(B1,1)
     * M3 = AA[2] * BB[2] = (A1,1)(B1,2 - B2,2)
     * M4 = AA[3] * BB[3] = (A2,2)(B2,1 - B1,1)
     * M5 = AA[4] * BB[4] = (A1,1 + A1,2)(B2,2)
     * M6 = AA[5] * BB[5] = (A2,1 - A1,1)(B1,1 + B1,2)
     * M7 = AA[6] * BB[6] = (A1,2 - A2,2)(B2,1 + B2,2)
     */

    /* AA[0] = (A1,1 + A2,2) */
    __submatrix_add (AA[0], A, tl_row_start, tl_col_start, br_row_start, br_col_start, m, n);
    /* AA[1] = (A2,1 + A2,2) */
    __submatrix_add (AA[1], A, bl_row_start, bl_col_start, br_row_start, br_col_start, m, n);
    /* AA[2] = (A1,1) */
    __submatrix_cpy (AA[2], A, tl_row_start, tl_col_start, m, n);
    /* AA[3] = (A2,2) */
    __submatrix_cpy (AA[3], A, br_row_start, br_col_start, m, n);
    /* AA[4] = (A1,1 + A1,2) */
    __submatrix_add (AA[4], A, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m, n);
    /* AA[5] = (A2,1 - A1,1) */
    __submatrix_sub (AA[5], A, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m, n);
    /* AA[6] = (A1,2 - A2,2) */
    __submatrix_sub (AA[6], A, tr_row_start, tr_col_start, br_row_start, br_col_start, m, n);

    /* BB[0] = (B1,1 + B2,2) */
    __submatrix_add (BB[0], B, tl_row_start, tl_col_start, br_row_start, br_col_start, m, n);
    /* BB[1] = (B1,1) */
    __submatrix_cpy (BB[1], B, tl_row_start, tl_col_start, m, n);
    /* BB[2] = (B1,2 - B2,2) */
    __submatrix_sub (BB[2], B, tr_row_start, tr_col_start, br_row_start, br_col_start, m, n);
    /* BB[3] = (B2,1 - B1,1) */
    __submatrix_sub (BB[3], B, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m, n);
    /* BB[4] = (B2,2) */
    __submatrix_cpy (BB[4], B, br_row_start, br_col_start, m, n);
    /* BB[5] = (B1,1 + B1,2) */
    __submatrix_add (BB[5], B, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m, n); 
    /* BB[6] = (B2,1 + B2,2) */
    __submatrix_add (BB[6], B, bl_row_start, bl_col_start, br_row_start, br_col_start, m, n);

    MM[0] = __mult (AA[0], BB[0], m);
    MM[1] = __mult (AA[1], BB[1], m);
    MM[2] = __mult (AA[2], BB[2], m);
    MM[3] = __mult (AA[3], BB[3], m);
    MM[4] = __mult (AA[4], BB[4], m);
    MM[5] = __mult (AA[5], BB[5], m);
    MM[6] = __mult (AA[6], BB[6], m);

    /* C1,1 = M1 + M4 - M5 + M7 */
    __submatrix_add (C, MM[0], MM[3], tl_row_start, tl_col_start, m, n);
    __submatrix_sub (C, MM[4], tl_row_start, tl_col_start, m, n);
    __submatrix_add (C, MM[6], tl_row_start, tl_col_start, m, n);

    /* C1,2 = M3 + M5 */
    __submatrix_add (C, MM[2], MM[4], tr_row_start, tr_col_start, m, n);

    /* C2,1 = M2 + M4 */
    __submatrix_add (C, MM[1], MM[3], bl_row_start, bl_col_start, m,  n);

    /* C2,2 = M1 - M2 + M3 + M6 */
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

  /**
   * Returns true only if the given matrix consists entirely of zeroes
   */
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

  /**
   * Assigns to the matrix C the sum of two quandrants of A using the given offsets
   */
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

  /**
   * Adds to the matrix C to the specified quadrant of A
   */
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

  /**
   * Assigns to specified quadrant of C the sum of A and B
   */
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

  /**
   * Assigns to C the difference of the two given quadrants of A
   */
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

  /**
   * Subtracts A from the given quadrant of C
   */
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
 
  /**
   * Assigns to the specified quadrant of C the difference of A and B
   */
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
  
  /**
   * Copies to C the given quadrant of A
   */
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

  /**
   * Returns a new n x n matrix containing the contents of m, but with extra elements 
   * padded with zeroes.
   */
  template <typename T>
  T*
  strassen_matrix_multiplier<T>::__pad (const T *m, size_t rows, size_t cols, size_t n)
  {
    size_t in;
    size_t ic;
    T *M = (T *) malloc (n * n * sizeof (T));
    
    for (size_t i = 0; i < rows; i++)
      {
        in = i * n;
        ic = i * cols;

        for (size_t j = 0; j < cols; j++)
          {
            M[in + j] = m[ic + j];
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
  
  /**
   * Given a matrix m, returns a new rows x cols matrix containing the nonzero elements from m.
   */
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
