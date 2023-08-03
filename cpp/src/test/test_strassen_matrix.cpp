#include <stdlib.h>
#include <stdio.h>
#include <execinfo.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

#include "../util/printer.hpp"
#include "../util/timer.hpp"

#include "../strassen/matrix.hpp"
#include "../strassen/naive_matrix_multiplier.hpp"
#include "../strassen/transpose_matrix_multiplier.hpp"
#include "../strassen/strassen_matrix_multiplier.hpp"
#include "../strassen/parallel_strassen_matrix_multiplier.hpp"

void
simple ()
{
  strassen::matrix<int> m (8, 8);
  m.random (10);

  strassen::matrix<int> n (8, 8);
  n.random (10);

  strassen::matrix<int> o (8, 8);
  o = m;
  o.mult (n);
  m.add (o);
  m.mult (7);

  strassen::matrix<int> p = o;

  o.add (m);
  p.add (m);

  o.mult (n);
  p.mult (n);

  if (o == p)
    printf ("matrices equal!\n");
  else
    {
      std::string os = alg_tostring (o);
      std::string ps = alg_tostring (p);
      
      printf ("%s\n\n%s\n\n", os.c_str(), ps.c_str());
    }
}

void
test_matrix_multipliers ()
{
  size_t s = 129;

  strassen::matrix<int> m (s, s);
  strassen::matrix<int> n (s, s);
  strassen::matrix<int> m_nmm (s, s, new strassen::naive_matrix_multiplier<int> ());
  strassen::matrix<int> m_tmm (s, s, new strassen::transpose_matrix_multiplier<int> ());
  strassen::matrix<int> m_smm (s, s, new strassen::strassen_matrix_multiplier<int> ());
  strassen::matrix<int> m_psmm (s, s, new strassen::parallel_strassen_matrix_multiplier<int> ());

  m.random (197);
  n.random (213);
  
  m_nmm = m;
  m_tmm = m;
  m_smm = m;
  m_psmm = m;

  if (!(m_nmm == m || m_tmm == m || m_smm == m || m_psmm == m))
    {
      fprintf (stderr, "time_matrix_multipliers: matrix initialization failure\n");      
    }

  m.mult (n);

  m_nmm.mult (n);
  m_tmm.mult (n);
  m_smm.mult (n);
  m_psmm.mult (n);

   if (!(m_nmm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: m_nmm matrix multiplication failure\n");      
    }
  else
    {
      fprintf (stderr, "test_matrix_multipliers: naive multiplier success\n");
    }
  
  if (!(m_tmm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: m_tmm matrix multiplication failure\n");      
    }
  else
    {
      fprintf (stderr, "test_matrix_multipliers: transpose multiplier success\n");
    }

  if (!(m_smm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: m_smm matrix multiplication failure\n");     

      std::string s = alg_tostring (m);
      std::string smm = alg_tostring (m_smm);

      std::cout << s << std::endl;
      std::cout << "====================" << std::endl;
      std::cout << smm << std::endl;
    }
  else
    {
      fprintf (stderr, "test_matrix_multipliers: strassen multiplier success\n");      
    }

  if (!(m_psmm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: m_psmm matrix multiplication failure\n");     

      std::string s = alg_tostring (m);
      std::string smm = alg_tostring (m_smm);

      std::cout << s << std::endl;
      std::cout << "====================" << std::endl;
      std::cout << smm << std::endl;
    }
  else
    {
      fprintf (stderr, "test_matrix_multipliers: parallel strassen multiplier success\n");      
    }
}

void
time_matrix_multipliers (size_t sz)
{
  strassen::timer t;
  strassen::matrix<int> m (sz, sz);
  strassen::matrix<int> n (sz, sz);
  strassen::matrix<int> m_nmm (sz, sz, new strassen::naive_matrix_multiplier<int> ());
  strassen::matrix<int> m_tmm (sz, sz, new strassen::transpose_matrix_multiplier<int> ());
  strassen::matrix<int> m_smm (sz, sz, new strassen::strassen_matrix_multiplier<int> ());
  strassen::matrix<int> m_psmm (sz, sz, new strassen::parallel_strassen_matrix_multiplier<int> ());

  m.random (103);
  n.random (103);
  
  m_nmm = m;
  m_tmm = m;
  m_smm = m;
  m_psmm = m;

  if (!(m_nmm == m || m_tmm == m || m_smm == m || m_psmm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: matrix initialization failure\n");      
    }

  printf ("MM array size: %lu x %lu\n", sz, sz);

  m.mult (n);

  t.start ();
  m_nmm.mult (n);
  t.stop ();

  printf ("MM time [naive]: %lu.%0lu\n", t.secs(), t.usecs());

  t.start ();
  m_tmm.mult (n);
  t.stop ();

  printf ("MM time [transpose]: %lu.%0lu\n", t.secs(), t.usecs());
  
  t.start ();
  m_smm.mult (n);
  t.stop ();

  printf ("MM time [strassen]: %lu.%0lu\n", t.secs(), t.usecs());

  t.start ();
  m_psmm.mult (n);
  t.stop ();

  printf ("MM time [parallel strassen]: %lu.%0lu\n", t.secs(), t.usecs());

  if (!(m_nmm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: m_nmm matrix multiplication failure\n");      
    }
  else
    {
      fprintf (stderr, "test_matrix_multipliers: naive multiplier success\n");
    }
  
  if (!(m_tmm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: m_tmm matrix multiplication failure\n");      
    }
  else
    {
      fprintf (stderr, "test_matrix_multipliers: transpose multiplier success\n");
    }

  if (!(m_smm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: m_smm matrix multiplication failure\n");     
    }
  else
    {
      fprintf (stderr, "test_matrix_multipliers: strassen multiplier success\n");      
    }
  
  if (!(m_psmm == m))
    {
      fprintf (stderr, "test_matrix_multipliers: m_psmm matrix multiplication failure\n");     
    }
  else
    {
      fprintf (stderr, "test_matrix_multipliers: parallel strassen multiplier success\n");      
    }
}

void
time_full (size_t lower, size_t upper, size_t factor, size_t trials)
{
  bool b = false;
  double tmp;
  double avgsum = 0.0;

  strassen::timer t;
  size_t naive_accumulator = 0;
  size_t transpose_accumulator = 0;
  size_t strassen_accumulator = 0;

  printf ("nxn naive transpose strassen\n");

  for (uint32_t i = 1; i < factor + 1; i++) {

    size_t lower_bound = i * lower;
    size_t upper_bound = i * upper;

    int x = (rand () % lower_bound) + upper_bound;
    int y = (rand () % lower_bound) + upper_bound;

    strassen::matrix<int> n (y, x);

    n.random (1000000);

    for (uint32_t j = 0; j < trials; j++)
      {
        strassen::matrix<int> m_nmm (x, y, new strassen::naive_matrix_multiplier<int> ());
        strassen::matrix<int> m_tmm (x, y, new strassen::transpose_matrix_multiplier<int> ());
        strassen::matrix<int> m_smm (x, y, new strassen::strassen_matrix_multiplier<int> ());
        m_nmm.random (1000000);
        m_tmm.random (1000000);
        m_smm.random (1000000);

        t.start ();
        m_nmm.mult (n);
        t.stop ();

        // Time in millis
        naive_accumulator += t.secs() * 1000 + t.usecs() / 1000;

        t.start ();
        m_tmm.mult (n);
        t.stop ();

        // Time in millis
        transpose_accumulator += t.secs() * 1000 + t.usecs() / 1000;

        t.start ();
        m_smm.mult (n);
        t.stop ();

        // Time in millis
        strassen_accumulator += t.secs() * 1000 + t.usecs() / 1000;
      }

    printf ("%d %d %d %lu.0 %lu.0 %lu.0\n", x, y, x * y, naive_accumulator, transpose_accumulator, strassen_accumulator);
  }
}

void
mult_test ()
{
  strassen::matrix<int> m (800, 800);
  strassen::matrix<int> n (800, 800);
  strassen::matrix<int> o (800, 800);

  m.random (231);
  n.random (673);

  o = m;

  m.mult (n);
  o.mult (n);

  if (!(m == o))
    {
      fprintf (stderr, "matrix mult match failure\n");
      
      std::string ms = alg_tostring (m);
      std::string os = alg_tostring (o);
      
      printf ("%s\n\n%s\n\n", ms.c_str(), os.c_str());
    }
}

void
big_test (size_t start)
{
  std::ofstream out;

  out.open ("./matrix_mult.out", std::ios::out | std::ios::app);

  if (out.fail ())
    {
      fprintf (stderr, "error opening output file\n");
      perror ("open");
      exit (1);
    }

  strassen::timer t;

  size_t reps = 4;

  size_t s = start;
  time_t secs[5];
  time_t usecs[5];
  char buf[128];

  while (s < 5000)
    {     
      strassen::matrix<int> m (s, s);
      strassen::matrix<int> n (s, s);
      strassen::matrix<int> m_nmm (s, s, new strassen::naive_matrix_multiplier<int> ());
      strassen::matrix<int> m_tmm (s, s, new strassen::transpose_matrix_multiplier<int> ());
      strassen::matrix<int> m_smm (s, s, new strassen::strassen_matrix_multiplier<int> ());
      strassen::matrix<int> m_psmm (s, s, new strassen::parallel_strassen_matrix_multiplier<int> ());
      
      m.random (197);
      n.random (213);
      
      m_nmm = m;
      m_tmm = m;
      m_smm = m;
      m_psmm = m;
      
      printf ("\nMM array size: %lu x %lu\n", s, s);
      
      secs[0] = 0;
      usecs[0] = 0;
      
      for (uint32_t i = 0; i < reps; i++)
        {
          t.start ();
          m_nmm.mult (n);
          t.stop ();
          
          secs[0] += t.secs ();
          usecs[0] += t.usecs ();
        }           

      secs[0] = secs[0] / reps;
      usecs[0] = usecs[0] / reps;
      m_nmm.clear ();
      
      printf ("MM time [naive]: %lu.%lu\n", secs[0], usecs[0]);
            
      secs[1] = 0;
      usecs[1] = 0;
      
      for (uint32_t i = 0; i < reps; i++)
        {
          t.start ();
          m_tmm.mult (n);
          t.stop ();
          
          secs[1] += t.secs ();
          usecs[1] += t.usecs ();
        }            

      secs[1] = (time_t)(secs[1] / reps);
      usecs[1] = (time_t)(usecs[1] / reps);
      m_tmm.clear ();

      printf ("MM time [transpose]: %lu.%lu\n", secs[1], usecs[1]);
      
      secs[3] = 0;
      usecs[3] = 0;
      
      for (uint32_t i = 0; i < reps; i++)
        {
          t.start ();
          m_smm.mult (n);
          t.stop ();
          
          secs[3] += t.secs ();
          usecs[3] += t.usecs ();
        }      

      secs[3] = secs[3] / reps;
      usecs[3] = usecs[3] / reps;
      m_smm.clear ();      
     
      printf ("MM time [strassen]: %lu.%lu\n", secs[3], usecs[3]);

      secs[4] = 0;
      usecs[4] = 0;
      
      for (uint32_t i = 0; i < reps; i++)
        {
          t.start ();
          m_psmm.mult (n);
          t.stop ();
          
          secs[4] += t.secs ();
          usecs[4] += t.usecs ();
        }      

      secs[4] = secs[4] / reps;
      usecs[4] = usecs[4] / reps;
      m_psmm.clear ();

      printf ("MM time [parallel strassen]: %lu.%lu\n", secs[4], usecs[4]);

      snprintf (buf, 128, "%lu %lu.%lu %lu.%lu %lu.%lu %lu.%lu %lu.%lu\n", 
        s,
        secs[0], usecs[0],
        secs[1], usecs[1],
        secs[2], usecs[2],
        secs[3], usecs[3],
        secs[4], usecs[4]);

      out.write (buf, strlen (buf));
      out.flush ();

      s += 137;
      sleep (1);
    }
}

void handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

int
main ()
{
  signal(SIGSEGV, handler);
  srand (time (NULL));

  //simple ();
  //test_matrix_multipliers ();
  time_full (50, 100, 50, 2);
  //mult_test ();

  //13
  //big_test (40);

  return 0;
}
