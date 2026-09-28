#include "pair_associating.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_associating_kinetics.h"
#include "force.h"
#include "modify.h"
#include "memory.h"
#include "neighbor.h"
#include "utils.h"
#include <cmath>
using namespace LAMMPS_NS;

PairAssociating::PairAssociating(LAMMPS *lmp) : Pair(lmp), k(0), r0(0), ee(0), rstar(0), shift(0), coeff_set(0), fix(nullptr)
{ restartinfo=0; single_enable=0; }
PairAssociating::~PairAssociating()
{
  if (allocated) { memory->destroy(setflag); memory->destroy(cutsq); }
}
void PairAssociating::settings(int narg,char **)
{ if (narg) error->all(FLERR,"Illegal pair_style associating command"); }
void PairAssociating::coeff(int narg,char **arg)
{
  if (narg != 5) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();
  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
  k=utils::numeric(FLERR,arg[2],false,lmp); r0=utils::numeric(FLERR,arg[3],false,lmp); ee=utils::numeric(FLERR,arg[4],false,lmp);
  if (k<=0 || r0<=0) error->all(FLERR,"Invalid associating FENE parameters");
  rstar=find_rstar(); shift=fene(rstar); coeff_set=1;
  for (int i=ilo;i<=ihi;++i)
    for (int j=MAX(jlo,i);j<=jhi;++j) setflag[i][j]=1;
}
void PairAssociating::allocate()
{
  allocated=1;
  int n=atom->ntypes+1;
  memory->create(setflag,n,n,"associating:setflag");
  memory->create(cutsq,n,n,"associating:cutsq");
  for (int i=1;i<n;++i) for (int j=i;j<n;++j) setflag[i][j]=0;
}
void PairAssociating::init_style()
{
  if (!coeff_set) error->all(FLERR,"All pair coefficients are not set");
  fix=nullptr;
  for (int i=0;i<modify->nfix;++i) {
    auto *candidate=dynamic_cast<FixAssociatingKinetics *>(modify->fix[i]);
    if (candidate) { if (fix) error->all(FLERR,"Only one fix associating/kinetics is allowed"); fix=candidate; }
  }
  if (!fix) error->all(FLERR,"Pair associating requires fix associating/kinetics");
  if (atom->map_style == Atom::MAP_NONE) error->all(FLERR,"Pair associating requires an atom map");
  neighbor->add_request(this);
}
double PairAssociating::init_one(int,int) { return r0; }
double PairAssociating::fene(double r) const { return -0.5*k*r0*r0*std::log(1.0-r*r/(r0*r0)); }
double PairAssociating::kg_derivative(double r) const
{ return -48.0/std::pow(r,13)+24.0/std::pow(r,7)+k*r/(1.0-r*r/(r0*r0)); }
double PairAssociating::find_rstar() const
{
  double lo=0.5, hi=std::pow(2.0,1.0/6.0);
  for (int n=0;n<100;++n) { double mid=0.5*(lo+hi); if (kg_derivative(mid)<0) lo=mid; else hi=mid; }
  return 0.5*(lo+hi);
}
void PairAssociating::compute(int eflag,int vflag)
{
  ev_init(eflag,vflag); comm->forward_comm(fix);
  tagint *partner=fix->partners(); double **x=atom->x, **f=atom->f; tagint *tag=atom->tag;
  for(int i=0;i<atom->nlocal;++i) {
    if (!partner[i]) continue;
    int j=atom->map(partner[i]);
    if (j < 0) error->one(FLERR,"Associating partner is outside communication range");
    if (partner[j] != tag[i]) error->one(FLERR,"Associating partner state is not reciprocal");
    double dx=x[i][0]-x[j][0], dy=x[i][1]-x[j][1], dz=x[i][2]-x[j][2]; domain->minimum_image(FLERR,dx,dy,dz);
    double rsq=dx*dx+dy*dy+dz*dz, r=std::sqrt(rsq), arg=1.0-rsq/(r0*r0);
    if (arg <= 0.0) error->one(FLERR,"Associating FENE bond exceeded R0");
    double fbond=-k/arg; f[i][0]+=dx*fbond; f[i][1]+=dy*fbond; f[i][2]+=dz*fbond;
    if (evflag) ev_tally_full(i,fene(r)-shift-ee,0.0,fbond,dx,dy,dz);
  }
  if (vflag_fdotr) virial_fdotr_compute();
}
