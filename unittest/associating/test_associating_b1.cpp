#include "atom.h"
#include "fix_associating_kinetics.h"
#include "pair_associating.h"
#include "force.h"
#include "input.h"
#include "lammps.h"
#include "modify.h"
#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <cstdio>
using namespace LAMMPS_NS;

class AssociatingB1 : public ::testing::Test {
 protected:
  struct EquilibriumSample { std::array<double,4> mean, se, exact; bigint created, broken; };
  LAMMPS *lmp;
  void SetUp() override { LAMMPS::argv a={"b1","-log","none","-screen","none","-echo","none"}; lmp=new LAMMPS(a,MPI_COMM_WORLD); }
  void TearDown() override { delete lmp; }
  void cmd(const char *s) { lmp->input->one(s); }
  FixAssociatingKinetics *setup(const char *fix) {
    cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
    cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 100"); cmd(fix); cmd("fix hold all move linear 0 0 0");
    return dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k"));
  }
  FixAssociatingKinetics *setup_equilibrium(double ea, double ee = 1.0, double temperature = 1.0) {
    cmd("clear");
    cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
    cmd("create_atoms 1 single 4 5 5"); cmd("create_atoms 1 single 5 5 5"); cmd("create_atoms 1 single 4.4 5.7 5"); cmd("mass 1 1");
    cmd("pair_style associating");
    char coeff[80], fix[160];
    std::snprintf(coeff, sizeof(coeff), "pair_coeff * * 30 1.5 %.17g", ee);
    std::snprintf(fix, sizeof(fix), "fix k all associating/kinetics 1 73 100 %.17g %.17g 1.2", ea, temperature);
    cmd(coeff); cmd(fix); cmd("fix hold all move linear 0 0 0");
    return dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k"));
  }
  EquilibriumSample sample(FixAssociatingKinetics *k, PairAssociating *pair, double temperature = 1.0) {
    auto *p=k->partners(); for(int i=0;i<1000;i++) cmd("run 1"); std::array<double,4> sum{}, sumsq{}; const int batches=20,width=500; bigint c0=k->compute_vector(1),b0=k->compute_vector(2);
    for(int b=0;b<batches;b++) { std::array<double,4> local{}; for(int n=0;n<width;n++) { cmd("run 1"); int s=-1; if(!p[0]&&!p[1]&&!p[2]) s=0; else if(p[0]==2&&p[1]==1&&!p[2]) s=1; else if(p[0]==3&&!p[1]&&p[2]==1) s=2; else if(!p[0]&&p[1]==3&&p[2]==2) s=3; EXPECT_GE(s,0); local[s]+=1; } for(int s=0;s<4;s++) { double x=local[s]/width; sum[s]+=x; sumsq[s]+=x*x; } }
    double d12=1.,d13=std::sqrt(.65),d23=std::sqrt(.85); std::array<double,4>w={1.,std::exp(-pair->delta_u(d12)/temperature),std::exp(-pair->delta_u(d13)/temperature),std::exp(-pair->delta_u(d23)/temperature)}; double z=w[0]+w[1]+w[2]+w[3]; EquilibriumSample out{};
    for(int s=0;s<4;s++) { out.mean[s]=sum[s]/batches; out.se[s]=std::sqrt((sumsq[s]-batches*out.mean[s]*out.mean[s])/(batches*(batches-1))); out.exact[s]=w[s]/z; } out.created=static_cast<bigint>(k->compute_vector(1))-c0; out.broken=static_cast<bigint>(k->compute_vector(2))-b0; return out;
  }
};

TEST_F(AssociatingB1, ConflictReevaluation)
{
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
  cmd("create_atoms 1 single 4 5 5"); cmd("create_atoms 1 single 5 5 5"); cmd("create_atoms 1 single 4.5 5.8 5");
  cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 100"); cmd("fix k all associating/kinetics 1 19 1e9 0 1 1.2"); cmd("fix hold all move linear 0 0 0");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k"));
  cmd("run 1"); auto *p=k->partners(); int bound=0; for(int i=0;i<lmp->atom->nlocal;i++) if(p[i]) ++bound;
  EXPECT_EQ(bound,2); EXPECT_EQ(k->events().size(),1u);
}

TEST_F(AssociatingB1, TwoStateFiniteStepProbabilities)
{
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
  cmd("create_atoms 1 single 4 5 5"); cmd("create_atoms 1 single 5 5 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 1"); cmd("fix k all associating/kinetics 1 19 10 0 1 1.2"); cmd("fix hold all move linear 0 0 0");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k")); auto *p=k->partners(); const int n=2000; int make=0, cut=0;
  for(int t=0;t<n;t++) { p[0]=p[1]=0; cmd("run 1"); if(p[0]) ++make; }
  for(int t=0;t<n;t++) { p[0]=2; p[1]=1; cmd("run 1"); if(!p[0]) ++cut; }
  auto *pair=dynamic_cast<PairAssociating *>(lmp->force->pair); double q=1-std::exp(-10*.005), du=pair->delta_u(1.0);
  double pc=q*std::min(1.0,std::exp(-du)), pb=q*std::min(1.0,std::exp(du));
  double sem=std::sqrt(pc*(1-pc)/n), seb=std::sqrt(pb*(1-pb)/n);
  EXPECT_LT(std::abs(make/double(n)-pc)/sem,5.0); EXPECT_LT(std::abs(cut/double(n)-pb)/seb,5.0);
}

TEST_F(AssociatingB1, UnequalDistanceEquilibrium)
{
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
  cmd("create_atoms 1 single 4 5 5"); cmd("create_atoms 1 single 5 5 5"); cmd("create_atoms 1 single 4.4 5.7 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 1"); cmd("fix k all associating/kinetics 1 73 100 0 1 1.2"); cmd("fix hold all move linear 0 0 0");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k")); auto *pair=dynamic_cast<PairAssociating *>(lmp->force->pair); auto result=sample(k,pair);
  for(int s=0;s<4;s++) EXPECT_LT(std::abs(result.mean[s]-result.exact[s]),5*result.se[s]+.002);
}

TEST_F(AssociatingB1, EaChangesRateNotEquilibrium)
{
  auto *slow_k=setup_equilibrium(10.0,1.0,5.0);
  auto *slow_pair=dynamic_cast<PairAssociating *>(lmp->force->pair);
  auto slow=sample(slow_k,slow_pair,5.0);
  auto *fast_k=setup_equilibrium(0.0,1.0,5.0);
  auto *fast_pair=dynamic_cast<PairAssociating *>(lmp->force->pair);
  auto fast=sample(fast_k,fast_pair,5.0);
  const double slow_rate=(slow.created+slow.broken)/10000.0;
  const double fast_rate=(fast.created+fast.broken)/10000.0;
  for(int s=0;s<4;s++) {
    EXPECT_LT(std::abs(slow.mean[s]-slow.exact[s]),5*slow.se[s]+.002);
    EXPECT_LT(std::abs(fast.mean[s]-fast.exact[s]),5*fast.se[s]+.002);
    EXPECT_LT(std::abs(slow.mean[s]-fast.mean[s]),5*std::sqrt(slow.se[s]*slow.se[s]+fast.se[s]*fast.se[s])+.004);
  }
  EXPECT_LT(slow_rate, fast_rate/3.0);
}

TEST_F(AssociatingB1, EeChangesEquilibrium)
{
  auto *weak_k=setup_equilibrium(0.0,0.0,5.0);
  auto *weak_pair=dynamic_cast<PairAssociating *>(lmp->force->pair);
  auto weak=sample(weak_k,weak_pair,5.0);
  auto *strong_k=setup_equilibrium(0.0,5.0,5.0);
  auto *strong_pair=dynamic_cast<PairAssociating *>(lmp->force->pair);
  auto strong=sample(strong_k,strong_pair,5.0);
  for(int s=0;s<4;s++) {
    EXPECT_LT(std::abs(weak.mean[s]-weak.exact[s]),5*weak.se[s]+.002);
    EXPECT_LT(std::abs(strong.mean[s]-strong.exact[s]),5*strong.se[s]+.002);
  }
  EXPECT_LT(strong.mean[0]+5*strong.se[0],weak.mean[0]-5*weak.se[0]);
}

TEST_F(AssociatingB1, EventMoleculeIDs)
{
  for (int same : {0,1}) {
    cmd("clear");
    cmd("units lj"); cmd("atom_style molecular"); cmd("atom_modify map yes"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
    cmd("create_atoms 1 single 4 5 5"); cmd("create_atoms 1 single 5 5 5"); cmd("set atom 1 mol 7"); cmd(same ? "set atom 2 mol 7" : "set atom 2 mol 9");
    cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 100"); cmd("fix k all associating/kinetics 1 19 1e9 0 1 1.2"); cmd("fix hold all move linear 0 0 0"); cmd("run 1");
    auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k"));
    ASSERT_EQ(k->events().size(),1u);
    const auto &event=k->events()[0];
    EXPECT_EQ(event.first,1); EXPECT_EQ(event.second,2); EXPECT_EQ(event.molecule_first,7); EXPECT_EQ(event.molecule_second,same ? 7 : 9); EXPECT_EQ(event.creation,1);
  }
}

TEST_F(AssociatingB1, StretchedBondIsChemicallyFrozen)
{
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
  cmd("create_atoms 1 single 4 5 5"); cmd("create_atoms 1 single 5.3 5 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 1"); cmd("fix k all associating/kinetics 1 19 1e9 0 1 1.2 debug_pair 1 2"); cmd("fix hold all move linear 0 0 0");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k"));
  cmd("run 10");
  auto *p=k->partners();
  EXPECT_EQ(p[0],2); EXPECT_EQ(p[1],1); EXPECT_TRUE(k->events().empty());
  EXPECT_NEAR(lmp->atom->f[0][0],30.0*1.3/(1.0-1.3*1.3/(1.5*1.5)),1e-10);
}
