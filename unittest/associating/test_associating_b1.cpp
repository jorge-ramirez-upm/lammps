#include "atom.h"
#include "fix_associating_kinetics.h"
#include "input.h"
#include "lammps.h"
#include "modify.h"
#include "gtest/gtest.h"
using namespace LAMMPS_NS;

class AssociatingB1 : public ::testing::Test {
 protected:
  LAMMPS *lmp;
  void SetUp() override { LAMMPS::argv a={"b1","-log","none","-echo","none"}; lmp=new LAMMPS(a,MPI_COMM_WORLD); }
  void TearDown() override { delete lmp; }
  void cmd(const char *s) { lmp->input->one(s); }
  FixAssociatingKinetics *setup(const char *atoms, const char *fix) {
    cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
    cmd(atoms); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 100"); cmd(fix); cmd("fix hold all move linear 0 0 0");
    return dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k"));
  }
};

TEST_F(AssociatingB1, ConflictReevaluation)
{
  auto *k=setup("create_atoms 1 single 4 5 5\ncreate_atoms 1 single 5 5 5\ncreate_atoms 1 single 4.5 5.8 5", "fix k all associating/kinetics 1 19 1e9 0 1 1.2");
  cmd("run 1"); auto *p=k->partners(); int bound=0; for(int i=0;i<lmp->atom->nlocal;i++) if(p[i]) ++bound;
  EXPECT_EQ(bound,2); EXPECT_EQ(k->events().size(),1u);
}
