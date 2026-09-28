/* -*- c++ -*- ---------------------------------------------------------- */
#ifdef FIX_CLASS
// clang-format off
FixStyle(associating/kinetics,FixAssociatingKinetics);
// clang-format on
#else
#ifndef LMP_FIX_ASSOCIATING_KINETICS_H
#define LMP_FIX_ASSOCIATING_KINETICS_H
#include "fix.h"
#include <cstdint>
#include <unordered_map>
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
  void post_run() override;
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
  void write_restart(FILE *) override;
  void restart(char *) override;
  double memory_usage() override;
  double compute_vector(int) override;
  tagint *partners() const { return partner; }
  const std::vector<Event> &events() const { return accepted_events; }
 private:
  struct StickerState { tagint partner, molecule; };
  struct StickerEdge { tagint first, second; double r; };
  using StickerStates = std::unordered_map<tagint, StickerState>;
  tagint *partner;
  tagint first, second;
  int debug_pair;
  int seed;
  int kinetics;
  int timing;
  double nu0, ea, temperature, r_assoc;
  bigint created, broken;
  bigint timing_sweeps, timing_stickers, timing_edges_sum, timing_edges_min, timing_edges_max;
  bigint timing_active_sum, timing_active_min, timing_active_max;
  bigint timing_created0, timing_broken0;
  double timing_stage[5];
  std::vector<Event> accepted_events;
  class NeighList *list;
  class PairAssociating *pair;
  int nmax_old;
  void initialize_debug_pair();
  static uint64_t random_value(uint64_t, bigint, tagint, tagint, uint64_t);
  void process_sweep(StickerStates &, std::vector<StickerEdge> &, bigint);
};
}
#endif
#endif
