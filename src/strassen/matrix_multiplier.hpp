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
  };
}

#endif /* MATRIX_MULTIPLIER_HPP_ */
