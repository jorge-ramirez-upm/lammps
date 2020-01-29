/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(bd/ABP,FixBDABP)

#else

#ifndef LMP_FIX_BD_ABP_H
#define LMP_FIX_BD_ABP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBDABP : public Fix {
 public:
  FixBDABP(class LAMMPS *, int, char **);
  virtual ~FixBDABP() {}
  int setmask();
  virtual void init();
  virtual void final_integrate();
  virtual void reset_dt();

 protected:
  double dtr,dtf;
  int mass_require;

  class RanMars *random;
  double Dt, Dr, Fp; // Translational diffusion constant
  double T; // Temperature
  double dtDt_T, sqrt2Dtdt, sqrt2Drdt;
  int seed;
  int dimension;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
