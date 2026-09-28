/* -*- c++ -*- ---------------------------------------------------------- */
#ifdef FIX_CLASS
// clang-format off
FixStyle(associating/kinetics,FixAssociatingKinetics);
// clang-format on
#else
#ifndef LMP_FIX_ASSOCIATING_KINETICS_H
#define LMP_FIX_ASSOCIATING_KINETICS_H
#include "fix.h"
#include <vector>
namespace LAMMPS_NS {
class FixAssociatingKinetics : public Fix {
 public:
  struct Event { tagint first, second, molecule_first, molecule_second; int creation; };
  FixAssociatingKinetics(class LAMMPS *, int, char **);
  ~FixAssociatingKinetics() override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void end_of_step() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  int pack_border(int, int *, double *) override;
  int unpack_border(int, int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int maxsize_restart() override;
  int size_restart(int) override;
  double memory_usage() override;
  double compute_vector(int) override;
  tagint *partners() const { return partner; }
  const std::vector<Event> &events() const { return accepted_events; }
 private:
  tagint *partner;
  tagint first, second;
  int debug_pair;
  int seed;
  int kinetics;
  double nu0, ea, temperature, r_assoc;
  bigint created, broken;
  std::vector<Event> accepted_events;
  class NeighList *list;
  class PairAssociating *pair;
  int nmax_old;
  void initialize_debug_pair();
};
}
#endif
#endif
