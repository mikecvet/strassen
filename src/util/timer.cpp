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

#include "timer.hpp"

using namespace strassen;

timer::timer ()
  : _secs (0),
    _usecs (0)
{
}

timer::~timer ()
{
}

void
timer::start ()
{
  gettimeofday (&_start, NULL);
}

void
timer::stop ()
{
  gettimeofday (&_stop, NULL);

  timersub (&_stop, &_start, &_diff);

  _secs = _diff.tv_sec;
  _usecs = _diff.tv_usec;
}

time_t
timer::secs ()
{
  return _secs;
}

time_t
timer::usecs ()
{
  return _usecs;
}
