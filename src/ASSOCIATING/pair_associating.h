/* -*- c++ -*- ---------------------------------------------------------- */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(associating,PairAssociating);
// clang-format on
#else
#ifndef LMP_PAIR_ASSOCIATING_H
#define LMP_PAIR_ASSOCIATING_H
#include "pair.h"
namespace LAMMPS_NS {
class PairAssociating : public Pair {
 public:
  PairAssociating(class LAMMPS *);
  ~PairAssociating() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
 private:
  double k, r0, ee, rstar, shift;
  int coeff_set;
  class FixAssociatingKinetics *fix;
  double fene(double) const;
  double kg_derivative(double) const;
  double find_rstar() const;
  void allocate();
};
}
#endif
#endif
