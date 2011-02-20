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

#ifndef PARALLEL_STRASSEN_MATRIX_MULTIPLIER_HPP_
#define PARALLEL_STRASSEN_MATRIX_MULTIPLIER_HPP_

#include <cmath>
#include "matrix_multiplier.hpp"
#include "strassen_matrix_multiplier.hpp"
#include "transpose_matrix_multiplier.hpp"

namespace strassen
{  
  template <typename T>
  class psmm_pair
  {
  public:
    T *A;
    T *B;
    T *C;
    size_t m;
    void *pmm;
    int j;
  };

  template <typename T>
  class parallel_strassen_matrix_multiplier : public strassen::strassen_matrix_multiplier<T>
  {
  private:
    bool __loop;    
    size_t __cntr;
    size_t __nthreads;
    pthread_t *__threads;
    pthread_mutex_t *__lock;
    pthread_cond_t *__cond[7];
    pthread_cond_t *__main_cond;
    psmm_pair<T>* __thread_data[7];

    strassen_matrix_multiplier<T> __smm;
    transpose_matrix_multiplier<T> __tmm;

    T* __mult (const T *A, const T *B, size_t n, uint32_t id);

  public:
    parallel_strassen_matrix_multiplier ();
    ~parallel_strassen_matrix_multiplier ();
    
    T* mult (const T *a, const T *b, size_t arows, size_t acols, size_t brows, size_t bcols);
    matrix_multiplier<T>* copy () const;

    void thread_loop (int id);
  };

  template <typename T>
  void*
  psmm_thread_entry (void *p)
  {
    psmm_pair<T> *pmm = ((psmm_pair<T> *) p);

    ((parallel_strassen_matrix_multiplier<T> *) pmm -> pmm) -> thread_loop (pmm -> j);
    //free (pmm);
    
    return NULL;
  }

  template <typename T>
  parallel_strassen_matrix_multiplier<T>::parallel_strassen_matrix_multiplier ()
  {
    __loop = true;
    __nthreads = 8;
    __cntr = __nthreads - 1;
    __threads = (pthread_t *) malloc ((__nthreads - 1) * sizeof (pthread_t));
    __lock = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
    __main_cond = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
    
    pthread_mutex_init (__lock, NULL);
    pthread_cond_init (__main_cond, NULL);

    for (size_t i = 0; i < (__nthreads - 1); i++)
      {
	__cond[i] = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (__cond[i], NULL);

	__thread_data[i] = (psmm_pair<T> *) malloc (sizeof (psmm_pair<T>));
	__thread_data[i] -> pmm = (void *) this;
	__thread_data[i] -> j = i + 1;
	pthread_create (&__threads[i], NULL, psmm_thread_entry<T>, (void *) __thread_data[i]);
      }
    
    sleep (1);
  }

  template <typename T>
  parallel_strassen_matrix_multiplier<T>::~parallel_strassen_matrix_multiplier ()
  {
    __loop = false;

    pthread_mutex_lock (__lock);

    for (size_t i = 0; i < (__nthreads - 1); i++)
      {
	pthread_cond_signal (__cond[i]);

	pthread_join (__threads[i], NULL);

	free (__thread_data[i]);

	pthread_cond_destroy (__cond[i]);
	free (__cond[i]);
      }

    pthread_mutex_unlock (__lock);

    free (__threads);
    pthread_mutex_destroy (__lock);
    pthread_cond_destroy (__main_cond);

    free (__lock);
    free (__main_cond);    
  }

  template <typename T>
  matrix_multiplier<T>*
  parallel_strassen_matrix_multiplier<T>::copy () const
  {
    return (new parallel_strassen_matrix_multiplier<T> ());
  }


  template <typename T>
  T*
  parallel_strassen_matrix_multiplier<T>::mult (const T *m, const T *n,
						size_t arows, size_t acols,
						size_t brows, size_t bcols)
  {
    if (acols == brows)
      {
	if (arows == acols && brows == bcols && !(arows & (arows - 1)))
	  {
	    T *C = __mult (m, n, arows, 0);
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
	    
	    N = std::pow (2, (size_t) (std::log (max_term) / strassen_matrix_multiplier<T>::__log2) + 1);
	    
	    A = __pad (m, arows, acols, N);
	    B = __pad (n, brows, bcols, N);

	    T *C = __mult (A, B, N, 0);
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
  parallel_strassen_matrix_multiplier<T>::__mult (const T *A, const T *B, size_t n, uint32_t id)
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
    T* MM[7] = {NULL};
    
    if (id)
      {
	if (!A[0] && !A[1])
	  {
	    if (__zeroes (A, n))
	      {
		memset (C, 0, n * n * sizeof (T));
		return C;
	      }
	  }
	
	if (!B[0] && !B[1])
	  {
	    if (__zeroes (B, n))
	      {
		memset (C, 0, n * n * sizeof (T));
		return C;
	      }
	  }
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
    
    if (m <= 256)
      {
	MM[0] = __tmm.mult (AA[0], BB[0], m, m, m, m);
	MM[1] = __tmm.mult (AA[1], BB[1], m, m, m, m);
	MM[2] = __tmm.mult (AA[2], BB[2], m, m, m, m);
	MM[3] = __tmm.mult (AA[3], BB[3], m, m, m, m);
	MM[4] = __tmm.mult (AA[4], BB[4], m, m, m, m);
	MM[5] = __tmm.mult (AA[5], BB[5], m, m, m, m);
	MM[6] = __tmm.mult (AA[6], BB[6], m, m, m, m);
      }
    else
      {
	if (!id)
	  {	    
	    pthread_mutex_lock (__lock);
	    __cntr = 7;
	    pthread_mutex_unlock (__lock);

	    for (uint32_t i = 0; i < (__nthreads - 1); i++)
	      {
		__thread_data[i]->A = AA[i];
		__thread_data[i]->B = BB[i];
		__thread_data[i]->C = MM[i];
		__thread_data[i]->m = m;

		pthread_cond_signal (__cond[i]);
	      }	
	    
	    pthread_mutex_lock (__lock);
	    
	    while (__cntr)
	      pthread_cond_wait (__main_cond, __lock);

	    MM[0] = __thread_data[0] -> C;
	    MM[1] = __thread_data[1] -> C;
	    MM[2] = __thread_data[2] -> C;
	    MM[3] = __thread_data[3] -> C;
	    MM[4] = __thread_data[4] -> C;
	    MM[5] = __thread_data[5] -> C;
	    MM[6] = __thread_data[6] -> C;
	    
	    pthread_mutex_unlock (__lock);
	  }
	else
	  {
	    MM[0] = __mult (AA[0], BB[0], m, id);
	    MM[1] = __mult (AA[1], BB[1], m, id);
	    MM[2] = __mult (AA[2], BB[2], m, id);
	    MM[3] = __mult (AA[3], BB[3], m, id);
	    MM[4] = __mult (AA[4], BB[4], m, id);
	    MM[5] = __mult (AA[5], BB[5], m, id);
	    MM[6] = __mult (AA[6], BB[6], m, id);
	  }
      }

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
  void
  parallel_strassen_matrix_multiplier<T>::thread_loop (int id)
  {
    while (__loop)
      {	
	pthread_mutex_lock (__lock);
	--__cntr;

	if (!__cntr)
	  pthread_cond_broadcast (__main_cond);
	
	pthread_cond_wait (__cond[id - 1], __lock);
	pthread_mutex_unlock (__lock);

	if (!__loop)
	  break;
       	
	__thread_data[id-1]->C = __mult (__thread_data[id-1]->A, 
					 __thread_data[id-1]->B, 
					 __thread_data[id-1]->m, 
					 id);	
      }
  }
}

#endif /* PARALLEL_STRASSEN_MATRIX_MULTIPLIER_HPP_ */
