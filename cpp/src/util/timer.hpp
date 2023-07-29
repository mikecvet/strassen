#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <time.h>
#include <sys/time.h>

namespace strassen
{
  class timer
  {
  private:
    time_t _secs;
    time_t _usecs;

    struct timeval _start;
    struct timeval _stop;
    struct timeval _diff;

  public:
    timer ();
    ~timer ();

    void start ();
    void stop ();
    time_t secs ();
    time_t usecs ();
  };
}

#endif /* TIMER_HPP_ */
