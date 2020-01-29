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

#ifdef PAIR_CLASS

PairStyle(RABP,PairRABP)

#else

#ifndef LMP_PAIR_RABP_H
#define LMP_PAIR_RABP_H

#include "pair.h"

namespace LAMMPS_NS {

class PairRABP : public Pair {
 public:
  PairRABP(class LAMMPS *);
  virtual ~PairRABP();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);

 protected:
 
  double cut_lj_global,cut_coul_global;
  double **cut_lj,**cut_ljsq;
  //double **cut_coul,**cut_coulsq;
  double **epsilon, **sigma, **epsilon_rep, **theta_max, **theta_tail;
  int **with_the_patch;
  
  double **ljRf, **ljRe; // Precalculated Repulsive LJ prefactors for force and energy
  double **ljAe1, **ljAe2; // Precalculated Attractive LJ prefactors for energy
  double **ljAf1, **ljAf2; // Precalculated Attractive LJ prefactors for force

  double **ljA1, **ljA2, **ljA3, **ljA4; // Attractive LJ


  //double **lj1,**lj2,**lj3,**lj4,**offset;
  //double **offset;
  class AtomVecEllipsoid *avec;
  
  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args in pair_style command

Self-explanatory.

E: Cannot (yet) use 'electron' units with dipoles

This feature is not yet supported.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair dipole/cut requires atom attributes q, mu, torque

The atom style defined does not have these attributes.

*/
