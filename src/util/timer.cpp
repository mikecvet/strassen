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
