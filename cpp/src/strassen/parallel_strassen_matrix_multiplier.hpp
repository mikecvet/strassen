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
  
  /**
   * A parallel implementation of strassen_matrix_multiplier.
   *
   * The main thread performs the initial division of the input matrices A and B into respective their 7 submatrix 
   * elements. These top-level divisions are then spread across 7 threads which will continue to recursively
   * multiply their matrices in parallel. When completed, the main thread aggregates their work and returns
   * the completed matrix.
   */
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
    psmm_pair<T>* __thread_data[7];   /* Data needed for each thread; referenced by thread ID */

    /* Re-entrant strassen_matrix_multiplier used to do actual work */
    strassen_matrix_multiplier<T> __smm;
    /* Re-entrant transpose_matrix_multiplier used to do actual work */
    transpose_matrix_multiplier<T> __tmm;

    T* __mult (const T *A, const T *B, size_t n, uint32_t id);

  public:
    parallel_strassen_matrix_multiplier ();
    virtual ~parallel_strassen_matrix_multiplier ();
    
    T* mult (const T *a, const T *b, size_t arows, size_t acols, size_t brows, size_t bcols);
    matrix_multiplier<T>* copy () const;

    /* Thread entry function for this class */
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

  /**
   * Initializes a few threads and necessary synchronization primitives used for concurrent calculations
   */
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

  /**
   * Perform a strassen multiplication of the given two matrices. 
   */
  template <typename T>
  T*
  parallel_strassen_matrix_multiplier<T>::mult (const T *m, const T *n,
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
            /* Call __mult with an ID of 0 */
            T *C = __mult (m, n, arows, 0);
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
            N = std::pow (2, (size_t) (std::log (max_term) / strassen_matrix_multiplier<T>::__log2) + 1);

            /* If m needs padding, pad it */
            if (arows != acols || arows & (arows - 1))
              A = this->__pad (m, arows, acols, N);
            
            /* If n needs padding, pad it */
            if (brows != bcols || brows & (brows - 1))
              B = this->__pad (n, brows, bcols, N);

            /* __mult does the actual multiplication work - call with ID of 0 to identify this as the
            * main thread. */
            if (A && B)
              C = this->__mult (A, B, N, 0);
            else if (A)
              C = this->__mult (A, n, N, 0);
            else if (B)
              C = this->__mult (m, B, N, 0);

            /* Extract the non-zero elements out of C and put them into a new matrix D which is 
            * of the size arows x bcols */
            T *D = this->__unpad (C, arows, arows, N);
            
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
  parallel_strassen_matrix_multiplier<T>::__mult (const T *A, const T *B, size_t n, uint32_t id)
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
    T* MM[7] = {NULL};  /* Products of above submatrices */
    
    if (id)
      {
        /* Make sure that neither A or B consist entirely of zeroes. If so, easy; nullify the
        * contents of C and return. */
        if ((!A[0] && !A[1] && this->__zeroes (A, n)) || (!B[0] && !B[1] && this->__zeroes (B, n)))
          {
            memset (C, 0, n * n * sizeof (T));
            return C;
          }
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
    this->__submatrix_add (AA[0], A, tl_row_start, tl_col_start, br_row_start, br_col_start, m, n);
    /* AA[1] = (A2,1 + A2,2) */
    this->__submatrix_add (AA[1], A, bl_row_start, bl_col_start, br_row_start, br_row_start, m, n);
    /* AA[2] = (A1,1) */
    this->__submatrix_cpy (AA[2], A, tl_row_start, tl_col_start, m, n);
    /* AA[3] = (A2,2) */
    this->__submatrix_cpy (AA[3], A, br_row_start, br_col_start, m, n);
    /* AA[4] = (A1,1 + A1,2) */
    this->__submatrix_add (AA[4], A, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m, n);
    /* AA[5] = (A2,1 - A1,1) */
    this->__submatrix_sub (AA[5], A, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m, n);
    /* AA[6] = (A1,2 - A2,2) */
    this->__submatrix_sub (AA[6], A, tr_row_start, tr_col_start, br_row_start, br_col_start, m, n);

    /* BB[0] = (B1,1 + B2,2) */
    this->__submatrix_add (BB[0], B, tl_row_start, tl_col_start, br_row_start, br_col_start, m, n);
    /* BB[1] = (B1,1) */
    this->__submatrix_cpy (BB[1], B, tl_row_start, tl_col_start, m, n);
    /* BB[2] = (B1,2 - B2,2) */
    this->__submatrix_sub (BB[2], B, tr_row_start, tr_col_start, br_row_start, br_col_start, m, n);
    /* BB[3] = (B2,1 - B1,1) */
    this->__submatrix_sub (BB[3], B, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m, n);
    /* BB[4] = (B2,2) */
    this->__submatrix_cpy (BB[4], B, br_row_start, br_col_start, m, n);
    /* BB[5] = (B1,1 + B1,2) */
    this->__submatrix_add (BB[5], B, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m, n); 
    /* BB[6] = (B2,1 + B2,2) */
    this->__submatrix_add (BB[6], B, bl_row_start, bl_col_start, br_row_start, br_col_start, m, n);
    
    /* If the thread ID is zero, this is the main thread */
    if (!id)
      {	    
        pthread_mutex_lock (__lock);
        __cntr = 7;
        pthread_mutex_unlock (__lock);

        /* Copy the above submatrix data into the global thread data structures. Each thread
        * from 1 - 7 corresponds to an AA[i], BB[i], and MM[i] which they will process in parallel. */
        for (uint32_t i = 0; i < (__nthreads - 1); i++)
          {
            __thread_data[i]->A = AA[i]; /* The A submatrix data */
            __thread_data[i]->B = BB[i]; /* The B submatrix data */
            __thread_data[i]->C = MM[i]; /* The M data */
            __thread_data[i]->m = m;     /* The current size of the submatrices */

            /* Wake this thread up */
            pthread_cond_signal (__cond[i]);
          }	
            
        pthread_mutex_lock (__lock);
            
        /* Wait here for all the threads to complete their work */
        while (__cntr)
          pthread_cond_wait (__main_cond, __lock);

        /* Copy back the completed data */
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
        /* This is a worker thread - do the M multiplications as necessary */
        MM[0] = __mult (AA[0], BB[0], m, id);
        MM[1] = __mult (AA[1], BB[1], m, id);
        MM[2] = __mult (AA[2], BB[2], m, id);
        MM[3] = __mult (AA[3], BB[3], m, id);
        MM[4] = __mult (AA[4], BB[4], m, id);
        MM[5] = __mult (AA[5], BB[5], m, id);
        MM[6] = __mult (AA[6], BB[6], m, id);
      }

    /* C1,1 = M1 + M4 - M5 + M7 */
    this->__submatrix_add (C, MM[0], MM[3], tl_row_start, tl_col_start, m, n);
    this->__submatrix_sub (C, MM[4], tl_row_start, tl_col_start, m, n);
    this->__submatrix_add (C, MM[6], tl_row_start, tl_col_start, m, n);
    
    /* C1,2 = M3 + M5 */
    this->__submatrix_add (C, MM[2], MM[4], tr_row_start, tr_col_start, m, n);

    /* C2,1 = M2 + M4 */
    this->__submatrix_add (C, MM[1], MM[3], bl_row_start, bl_col_start, m,  n);
    
    /* C2,2 = M1 - M2 + M3 + M6 */
    this->__submatrix_sub (C, MM[0], MM[1], br_row_start, br_col_start, m, n);
    this->__submatrix_add (C, MM[2], br_row_start, br_col_start, m, n);
    this->__submatrix_add (C, MM[5], br_row_start, br_col_start, m, n);

    for (uint32_t i = 0; i < 7; i++)
      {
        free (AA[i]);
        free (BB[i]);
        free (MM[i]);
      }

    return C;
  }

  /**
   * Loop for worker threads. Waits here until signalled by the main thread that
   * its thread data is ready for processing.
   */
  template <typename T>
  void
  parallel_strassen_matrix_multiplier<T>::thread_loop (int id)
  {
    while (__loop)
      {	
        pthread_mutex_lock (__lock);
        --__cntr;

        /* If this is the last thread to enter; notify the main thread that all workers are ready */
        if (!__cntr)
          pthread_cond_broadcast (__main_cond);
        
        /* Wait here for work */
        pthread_cond_wait (__cond[id - 1], __lock);
        pthread_mutex_unlock (__lock);

        if (!__loop)
          break;
              
        /* Begin recursively multiplying using the supplied thread data */
        __thread_data[id-1]->C = __mult (__thread_data[id-1]->A, 
                __thread_data[id-1]->B, 
                __thread_data[id-1]->m, 
                id);	
      }
  }
}

#endif /* PARALLEL_STRASSEN_MATRIX_MULTIPLIER_HPP_ */
