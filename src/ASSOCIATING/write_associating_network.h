/* -*- c++ -*- ---------------------------------------------------------- */
#ifdef COMMAND_CLASS
// clang-format off
CommandStyle(write_associating_network,WriteAssociatingNetwork);
// clang-format on
#else
#ifndef LMP_WRITE_ASSOCIATING_NETWORK_H
#define LMP_WRITE_ASSOCIATING_NETWORK_H
#include "command.h"
namespace LAMMPS_NS {
class WriteAssociatingNetwork : public Command {
 public:
  WriteAssociatingNetwork(class LAMMPS *lmp) : Command(lmp) {}
  void command(int, char **) override;
};
}
#endif
#endif
