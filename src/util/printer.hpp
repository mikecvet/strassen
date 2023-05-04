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
