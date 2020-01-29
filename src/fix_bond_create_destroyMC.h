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

FixStyle(bond/create/destroy/MC,FixBondCreateDestroyMC)

#else

#ifndef LMP_FIX_BOND_CREATE_DESTROY_MC_H
#define LMP_FIX_BOND_CREATE_DESTROY_MC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondCreateDestroyMC : public Fix {
 public:
  FixBondCreateDestroyMC(class LAMMPS *, int, char **);
  ~FixBondCreateDestroyMC();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
  void post_integrate();

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  double compute_vector(int);
  double memory_usage();

 private:
  int me;
  int iatomtype,jatomtype;
  int btype,seed;
  double cutoff, cutsq, fraction, Ea, Ee, T, kFENE, RFENE, sigmaLJ, epsilonLJ, kA; //JAVI
  int imaxbond,jmaxbond;
  double cutminsq,pon,poff;
  int overflow;
  tagint lastcheck;

  int *bondcount;
  int createcount,createcounttotal;
  int breakcount, breakcounttotal;
  int nmax;
  tagint *partner,*finalpartner;
  double *distsq,*probability;

  //JAVI: Gillespie arrays
  tagint* Gi, * Gj;  // Partners of a bond that can be broken
  double* Gaccumaij;      // Accumulated propensity
  double dtGillespie;
  //

  int ncreate,maxcreate;
  int nbreak, maxbreak;
  tagint **created;
  tagint **broken;

  tagint *copy;

  class RanMars *random;
  class NeighList *list;

  int countflag,commflag;
  int diffmol; // Only allow bonds between different molecules

  void check_ghosts();
  void update_topology();
  void update_topology_break();
  void rebuild_special_one(int);
  int dedup(int, int, tagint *);

  //JAVI: New Functions
  int PoissonSmall(double lambda);
  int PoissonLarge(double lambda);
  int GetPoisson(double lambda);

  double ULJ(double rsq);
  double UFENE(double rsq);
  double UBondedSticker(double rsq);
  //JAVI: End of new Functions

  // DEBUG

  void print_bb();
  void print_copy(const char *, tagint, int, int, int, int *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix bond/create/destroy command

Self-explanatory.

E: Invalid bond type in fix bond/create/destroy command

Self-explanatory.

E: Cannot use fix bond/create/destroy with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Inconsistent iparam/jparam values in fix bond/create/destroy command

If itype and jtype are the same, then their maxbond and newtype
settings must also be the same.

E: Fix bond/create/destroy cutoff is longer than pairwise cutoff

This is not allowed because bond creation is done using the
pairwise neighbor list.

E: Fix bond/create/destroy angle type is invalid

Self-explanatory.

E: Fix bond/create/destroy dihedral type is invalid

Self-explanatory.

E: Fix bond/create/destroy improper type is invalid

Self-explanatory.

E: Cannot yet use fix bond/create/destroy with this improper style

This is a current restriction in LAMMPS.

E: Fix bond/create/destroy needs ghost atoms from further away

This is because the fix needs to walk bonds to a certain distance to
acquire needed info, The comm_modify cutoff command can be used to
extend the communication range.

E: New bond exceeded bonds per atom in fix bond/create/destroy

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

E: New bond exceeded special list size in fix bond/create/destroy

See the special_bonds extra command for info on how to leave space in
the special bonds list to allow for additional bonds to be formed.

E: Fix bond/create/destroy induced too many angles/dihedrals/impropers per atom

See the read_data command for info on setting the "extra angle per
atom", etc header values to allow for additional angles, etc to be
formed.

E: Special list size exceeded in fix bond/create/destroy

See the read_data command for info on setting the "extra special per
atom" header value to allow for additional special values to be
stored.

*/
