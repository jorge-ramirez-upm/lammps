#include "atom.h"
#include "comm.h"
#include "fix_associating_kinetics.h"
#include "pair_associating.h"
#include "force.h"
#include "input.h"
#include "lammps.h"
#include "modify.h"
#include "gtest/gtest.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <array>
#include <cstdio>
#include <utility>
#include <vector>
using namespace LAMMPS_NS;

class AssociatingB1 : public ::testing::Test {
 protected:
  struct EquilibriumSample { std::array<double,4> mean, se, exact; bigint created, broken; };
  struct SweepTrace {
    std::vector<std::pair<tagint,tagint>> state;
    std::vector<std::array<tagint,3>> events;
    bigint created, broken, active;
  };
  LAMMPS *lmp;
  void SetUp() override { LAMMPS::argv a={"b1","-log","none","-screen","none","-echo","none"}; lmp=new LAMMPS(a,MPI_COMM_WORLD); }
  void TearDown() override { delete lmp; }
  void cmd(const char *s) { lmp->input->one(s); }
  static void setup_trace_system(LAMMPS *target, bool parallel, bool three) {
    auto one=[&](const char *s) { target->input->one(s); };
    one("units lj"); one("atom_style atomic"); one("atom_modify map yes"); one(parallel ? "processors 2 1 1" : "processors 1 1 1"); one("region b block 0 10 0 10 0 10"); one("create_box 1 b");
    one("create_atoms 1 single 4.5 5 5"); one("create_atoms 1 single 5.5 5 5");
    if (three) one("create_atoms 1 single 4.9 5.7 5");
    one("mass 1 1"); one("pair_style associating"); one("pair_coeff * * 30 1.5 100"); one("fix k all associating/kinetics 1 73 1e9 0 1 1.2"); one("fix hold all move linear 0 0 0");
  }
  static SweepTrace trace(LAMMPS *target) {
    auto *k=dynamic_cast<FixAssociatingKinetics *>(target->modify->get_fix_by_id("k"));
    std::vector<tagint> local;
    for (int i=0;i<target->atom->nlocal;++i) { local.push_back(target->atom->tag[i]); local.push_back(k->partners()[i]); }
    int nlocal=local.size()/2,nprocs;
    MPI_Comm_size(target->world,&nprocs);
    std::vector<int> counts(nprocs),offsets(nprocs),counts2(nprocs),offsets2(nprocs);
    MPI_Allgather(&nlocal,1,MPI_INT,counts.data(),1,MPI_INT,target->world);
    int total=0; for (int i=0;i<nprocs;++i) { offsets[i]=total; total+=counts[i]; counts2[i]=2*counts[i]; offsets2[i]=2*offsets[i]; }
    std::vector<tagint> global(2*total);
    MPI_Allgatherv(local.data(),2*nlocal,MPI_LMP_TAGINT,global.data(),counts2.data(),offsets2.data(),MPI_LMP_TAGINT,target->world);
    SweepTrace out{};
    for (int i=0;i<total;++i) out.state.push_back({global[2*i],global[2*i+1]});
    std::sort(out.state.begin(),out.state.end());
    for (const auto &event : k->events()) out.events.push_back({event.first,event.second,static_cast<tagint>(event.creation)});
    out.created=static_cast<bigint>(k->compute_vector(1)); out.broken=static_cast<bigint>(k->compute_vector(2)); out.active=static_cast<bigint>(k->compute_vector(0));
    return out;
  }
  static void assert_diagnostics(FixAssociatingKinetics *k) {
    bigint values[3]={static_cast<bigint>(k->compute_vector(0)),static_cast<bigint>(k->compute_vector(1)),static_cast<bigint>(k->compute_vector(2))},lo[3],hi[3];
    MPI_Allreduce(values,lo,3,MPI_LMP_BIGINT,MPI_MIN,MPI_COMM_WORLD); MPI_Allreduce(values,hi,3,MPI_LMP_BIGINT,MPI_MAX,MPI_COMM_WORLD);
    for (int i=0;i<3;++i) EXPECT_EQ(lo[i],hi[i]);
    int n=k->events().size(),nlo,nhi; MPI_Allreduce(&n,&nlo,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD); MPI_Allreduce(&n,&nhi,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); ASSERT_EQ(nlo,nhi);
    for (int i=0;i<n;++i) { tagint event[3]={k->events()[i].first,k->events()[i].second,static_cast<tagint>(k->events()[i].creation)},min[3],max[3]; MPI_Allreduce(event,min,3,MPI_LMP_TAGINT,MPI_MIN,MPI_COMM_WORLD); MPI_Allreduce(event,max,3,MPI_LMP_TAGINT,MPI_MAX,MPI_COMM_WORLD); for(int j=0;j<3;++j) EXPECT_EQ(min[j],max[j]); }
  }
  void decomposition_equivalence(bool three) {
    int nprocs; MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    if (nprocs != 2) GTEST_SKIP() << "requires two MPI ranks";
    std::vector<SweepTrace> serial;
    { LAMMPS::argv a={"serial","-log","none","-screen","none","-echo","none"}; LAMMPS reference(a,MPI_COMM_SELF); setup_trace_system(&reference,false,three);
      for (int step=0;step<100;++step) { reference.input->one("run 1"); serial.push_back(trace(&reference)); } }
    if (!three) ASSERT_EQ(serial[0].events.size(),1u);
    setup_trace_system(lmp,true,three);
    for (int step=0;step<100;++step) {
      cmd("run 1"); auto parallel=trace(lmp);
      EXPECT_EQ(parallel.state,serial[step].state) << "step " << step;
      EXPECT_EQ(parallel.events,serial[step].events) << "step " << step;
      EXPECT_EQ(parallel.created,serial[step].created) << "step " << step;
      EXPECT_EQ(parallel.broken,serial[step].broken) << "step " << step;
      EXPECT_EQ(parallel.active,serial[step].active) << "step " << step;
    }
  }
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

TEST_F(AssociatingB1, ReplicatedMPISmoke)
{
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  if (nprocs != 2) GTEST_SKIP() << "requires two MPI ranks";
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("processors 2 1 1"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
  cmd("create_atoms 1 single 4.5 5 5"); cmd("create_atoms 1 single 5.5 5 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 100"); cmd("fix k all associating/kinetics 1 19 1e9 0 1 1.2"); cmd("fix hold all move linear 0 0 0");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k"));
  cmd("run 1");
  ASSERT_EQ(k->events().size(),1u); EXPECT_EQ(k->events()[0].creation,1);
  auto *p=k->partners(); int local_reciprocal=0;
  for (int i=0;i<lmp->atom->nlocal;++i) if (p[i] && p[i]==3-lmp->atom->tag[i]) ++local_reciprocal;
  int reciprocal=0; MPI_Allreduce(&local_reciprocal,&reciprocal,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  EXPECT_EQ(reciprocal,2); EXPECT_EQ(k->compute_vector(0),1);
  cmd("run 1");
  double local_force=0.0; for (int i=0;i<lmp->atom->nlocal;++i) local_force=std::max(local_force,std::abs(lmp->atom->f[i][0]));
  double force=0.0; MPI_Allreduce(&local_force,&force,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  EXPECT_NEAR(force,30.0/(1.0-1.0/(1.5*1.5)),1e-10);
}

TEST_F(AssociatingB1, DecompositionEquivalenceTwoStickers)
{
  decomposition_equivalence(false);
}

TEST_F(AssociatingB1, DecompositionEquivalenceThreeStickers)
{
  decomposition_equivalence(true);
}

TEST_F(AssociatingB1, CrossDomainTwoStateProbabilities)
{
  int nprocs; MPI_Comm_size(MPI_COMM_WORLD,&nprocs); if (nprocs != 2) GTEST_SKIP() << "requires two MPI ranks";
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("processors 2 1 1"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
  cmd("create_atoms 1 single 4.5 5 5"); cmd("create_atoms 1 single 5.5 5 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 1"); cmd("fix k all associating/kinetics 1 19 10 0 1 1.2"); cmd("fix hold all move linear 0 0 0");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k")); auto *pair=dynamic_cast<PairAssociating *>(lmp->force->pair); const int n=2000; int make=0,cut=0;
  for (int trial=0;trial<n;++trial) { auto *p=k->partners(); for(int i=0;i<lmp->atom->nlocal;++i) p[i]=0; lmp->comm->forward_comm(k); cmd("run 1"); if(k->compute_vector(0)==1) ++make; assert_diagnostics(k); }
  for (int trial=0;trial<n;++trial) { auto *p=k->partners(); for(int i=0;i<lmp->atom->nlocal;++i) p[i]=lmp->atom->tag[i]==1 ? 2 : 1; lmp->comm->forward_comm(k); cmd("run 1"); if(k->compute_vector(0)==0) ++cut; assert_diagnostics(k); }
  double q=1-std::exp(-10*.005),du=pair->delta_u(1.0),pc=q*std::min(1.0,std::exp(-du)),pb=q*std::min(1.0,std::exp(du));
  EXPECT_LT(std::abs(make/double(n)-pc)/std::sqrt(pc*(1-pc)/n),5.0);
  EXPECT_LT(std::abs(cut/double(n)-pb)/std::sqrt(pb*(1-pb)/n),5.0);
}

TEST_F(AssociatingB1, CrossDomainThreeStickerEquilibrium)
{
  int nprocs; MPI_Comm_size(MPI_COMM_WORLD,&nprocs); if (nprocs != 2) GTEST_SKIP() << "requires two MPI ranks";
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("processors 2 1 1"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b");
  cmd("create_atoms 1 single 4.5 5 5"); cmd("create_atoms 1 single 5.5 5 5"); cmd("create_atoms 1 single 4.9 5.7 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 1"); cmd("fix k all associating/kinetics 1 73 100 0 5 1.2"); cmd("fix hold all move linear 0 0 0");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k")); auto *pair=dynamic_cast<PairAssociating *>(lmp->force->pair);
  for(int i=0;i<1000;++i) { cmd("run 1"); assert_diagnostics(k); }
  std::array<double,4> sum{},sumsq{}; const int batches=20,width=500;
  for(int b=0;b<batches;++b) { std::array<double,4> local{}; for(int n=0;n<width;++n) { cmd("run 1"); auto state=trace(lmp).state; int s=-1; if(!state[0].second&&!state[1].second&&!state[2].second) s=0; else if(state[0].second==2&&state[1].second==1&&!state[2].second) s=1; else if(state[0].second==3&&!state[1].second&&state[2].second==1) s=2; else if(!state[0].second&&state[1].second==3&&state[2].second==2) s=3; EXPECT_GE(s,0); local[s]+=1; assert_diagnostics(k); } for(int s=0;s<4;++s) { double x=local[s]/width; sum[s]+=x; sumsq[s]+=x*x; } }
  std::array<double,4> w={1.,std::exp(-pair->delta_u(1.0)/5),std::exp(-pair->delta_u(std::sqrt(.65))/5),std::exp(-pair->delta_u(std::sqrt(.85))/5)}; double z=w[0]+w[1]+w[2]+w[3];
  for(int s=0;s<4;++s) { double mean=sum[s]/batches,se=std::sqrt((sumsq[s]-batches*mean*mean)/(batches*(batches-1))); EXPECT_LT(std::abs(mean-w[s]/z),5*se+.002); }
}

TEST_F(AssociatingB1, ActiveKineticsMigration)
{
  int nprocs,me; MPI_Comm_size(MPI_COMM_WORLD,&nprocs); MPI_Comm_rank(MPI_COMM_WORLD,&me); if(nprocs!=2) GTEST_SKIP() << "requires two MPI ranks";
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("processors 2 1 1"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b"); cmd("create_atoms 1 single 4.6 5 5"); cmd("create_atoms 1 single 5.6 5 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 100"); cmd("fix k all associating/kinetics 1 19 1e9 0 1 1.2"); cmd("fix hold all move linear .2 0 0"); cmd("timestep .5");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k")); cmd("run 6"); assert_diagnostics(k);
  int local_owner=-1,owner; for(int i=0;i<lmp->atom->nlocal;++i) if(lmp->atom->tag[i]==1) local_owner=me; MPI_Allreduce(&local_owner,&owner,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); EXPECT_EQ(owner,1);
  auto state=trace(lmp).state; EXPECT_EQ(state[0].second,2); EXPECT_EQ(state[1].second,1); EXPECT_EQ(k->compute_vector(0),1); EXPECT_EQ(k->compute_vector(1),1); EXPECT_EQ(k->compute_vector(2),0);
  double local_force=0,force; for(int i=0;i<lmp->atom->nlocal;++i) local_force=std::max(local_force,std::abs(lmp->atom->f[i][0])); MPI_Allreduce(&local_force,&force,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); EXPECT_NEAR(force,30.0/(1.0-1.0/(1.5*1.5)),1e-10);
}

TEST_F(AssociatingB1, FrozenStretchedBondBlocksFormationAfterMigration)
{
  int nprocs,me; MPI_Comm_size(MPI_COMM_WORLD,&nprocs); MPI_Comm_rank(MPI_COMM_WORLD,&me); if(nprocs!=2) GTEST_SKIP() << "requires two MPI ranks";
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("processors 2 1 1"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b"); cmd("create_atoms 1 single 4.6 5 5"); cmd("create_atoms 1 single 5.9 5 5"); cmd("create_atoms 1 single 5.6 5 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 1"); cmd("fix k all associating/kinetics 1 19 1e9 0 1 1.2 debug_pair 1 2"); cmd("fix hold all move linear .2 0 0"); cmd("timestep .5");
  auto *k=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k")); cmd("run 6"); assert_diagnostics(k);
  int local_owner=-1,owner; for(int i=0;i<lmp->atom->nlocal;++i) if(lmp->atom->tag[i]==1) local_owner=me; MPI_Allreduce(&local_owner,&owner,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); EXPECT_EQ(owner,1);
  auto state=trace(lmp).state; EXPECT_EQ(state[0].second,2); EXPECT_EQ(state[1].second,1); EXPECT_EQ(state[2].second,0); EXPECT_EQ(k->compute_vector(0),1); EXPECT_EQ(k->compute_vector(1),0); EXPECT_EQ(k->compute_vector(2),0); EXPECT_TRUE(k->events().empty());
  double local_force=0,force; for(int i=0;i<lmp->atom->nlocal;++i) local_force=std::max(local_force,std::abs(lmp->atom->f[i][0])); MPI_Allreduce(&local_force,&force,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); EXPECT_NEAR(force,30.0*1.3/(1.0-1.3*1.3/(1.5*1.5)),1e-10);
}

TEST_F(AssociatingB1, KineticMPIRestartPreservesDiagnosticsAndForce)
{
  int nprocs,me; MPI_Comm_size(MPI_COMM_WORLD,&nprocs); MPI_Comm_rank(MPI_COMM_WORLD,&me); if(nprocs!=2) GTEST_SKIP() << "requires two MPI ranks";
  const char *path="/tmp/associating-b2-restart"; if(me==0) std::remove(path); MPI_Barrier(MPI_COMM_WORLD);
  cmd("units lj"); cmd("atom_style atomic"); cmd("atom_modify map yes"); cmd("processors 2 1 1"); cmd("region b block 0 10 0 10 0 10"); cmd("create_box 1 b"); cmd("create_atoms 1 single 4.5 5 5"); cmd("create_atoms 1 single 5.5 5 5"); cmd("mass 1 1"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 100"); cmd("fix k all associating/kinetics 1 19 1e9 0 1 1.2"); cmd("fix hold all move linear 0 0 0");
  auto *before=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k")); cmd("run 2"); auto saved=trace(lmp); assert_diagnostics(before); cmd("write_restart /tmp/associating-b2-restart"); cmd("clear"); cmd("read_restart /tmp/associating-b2-restart"); cmd("pair_style associating"); cmd("pair_coeff * * 30 1.5 100"); cmd("fix k all associating/kinetics 1 19 1e9 0 1 1.2"); cmd("run 0");
  auto *after=dynamic_cast<FixAssociatingKinetics *>(lmp->modify->get_fix_by_id("k")); auto restored=trace(lmp); assert_diagnostics(after); EXPECT_EQ(restored.state,saved.state); EXPECT_EQ(restored.events,saved.events); EXPECT_EQ(restored.created,saved.created); EXPECT_EQ(restored.broken,saved.broken); EXPECT_EQ(restored.active,saved.active);
  double local_force=0,force; for(int i=0;i<lmp->atom->nlocal;++i) local_force=std::max(local_force,std::abs(lmp->atom->f[i][0])); MPI_Allreduce(&local_force,&force,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD); EXPECT_NEAR(force,30.0/(1.0-1.0/(1.5*1.5)),1e-10); MPI_Barrier(MPI_COMM_WORLD); if(me==0) std::remove(path);
}
