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

#ifndef PRINTER_HPP_
#define PRINTER_HPP_

#include <string>
#include <sstream>

#include "../strassen/matrix.hpp"

namespace strassen
{
  //  template <typename T>
  std::string
  alg_tostring (strassen::matrix<int> &m)
  {
    size_t i = 0;
    size_t rows = m.rows ();
    size_t cols = m.cols ();
    std::stringstream ss;
    matrix<int>::iterator iter (m);
    
    ss << "| ";
    
    while (iter.ok ())
      {
	ss << iter.val () << " ";
	
	if (++i == cols && (--rows))
	  {
	    i = 0;
	    ss << "|\n| ";
	  }
	
	++iter;
      }

    ss << "|\n";

    return ss.str ();
  }

  std::string
  alg_tostring (strassen::matrix<uint32_t> &m)
  {
    size_t i = 0;
    size_t rows = m.rows ();
    size_t cols = m.cols ();
    std::stringstream ss;
    matrix<uint32_t>::iterator iter (m);
    
    ss << "| ";
    
    while (iter.ok ())
      {
	ss << iter.val () << " ";
	
	if (++i == cols && (--rows))
	  {
	    i = 0;
	    ss << "|\n| ";
	  }
	
	++iter;
      }

    ss << "|\n";

    return ss.str ();
  }
}

#endif /* PRINTER_HPP_ */
