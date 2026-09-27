/* -*- c++ -*- ---------------------------------------------------------- */
#ifdef FIX_CLASS
// clang-format off
FixStyle(associating/kinetics,FixAssociatingKinetics);
// clang-format on
#else
#ifndef LMP_FIX_ASSOCIATING_KINETICS_H
#define LMP_FIX_ASSOCIATING_KINETICS_H
#include "fix.h"
namespace LAMMPS_NS {
class FixAssociatingKinetics : public Fix {
 public:
  FixAssociatingKinetics(class LAMMPS *, int, char **);
  ~FixAssociatingKinetics() override;
  int setmask() override;
  void init() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  int pack_border(int, int *, double *) override;
  int unpack_border(int, int, double *) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int maxsize_restart() override;
  int size_restart(int) override;
  double memory_usage() override;
  tagint *partners() const { return partner; }
 private:
  tagint *partner;
  tagint first, second;
  int nmax_old;
  void initialize_debug_pair();
};
}
#endif
#endif
