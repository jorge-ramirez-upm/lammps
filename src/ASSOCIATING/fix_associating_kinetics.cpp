#include "fix_associating_kinetics.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "force.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair_associating.h"
#include "update.h"
#include "utils.h"
#include <cstring>
#include <algorithm>
#include <vector>
using namespace LAMMPS_NS;
using namespace FixConst;

FixAssociatingKinetics::FixAssociatingKinetics(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp,narg,arg), partner(nullptr), first(0), second(0), debug_pair(0), kinetics(0), nu0(0), ea(0), temperature(0), r_assoc(0), created(0), broken(0), list(nullptr), nmax_old(0)
{
  if (narg != 9 && (narg != 3 && (narg != 6 || strcmp(arg[3],"debug_pair") != 0)))
    error->all(FLERR,"Illegal fix associating/kinetics command");
  if (narg == 6) {
    first = utils::tnumeric(FLERR,arg[4],false,lmp);
    second = utils::tnumeric(FLERR,arg[5],false,lmp);
    if (first <= 0 || second <= 0 || first == second)
      error->all(FLERR,"Invalid debug association pair");
    debug_pair = 1;
  } else if (narg == 9) {
    nevery=utils::inumeric(FLERR,arg[3],false,lmp); int seed=utils::inumeric(FLERR,arg[4],false,lmp);
    nu0=utils::numeric(FLERR,arg[5],false,lmp); ea=utils::numeric(FLERR,arg[6],false,lmp);
    temperature=utils::numeric(FLERR,arg[7],false,lmp); r_assoc=utils::numeric(FLERR,arg[8],false,lmp);
    if (nevery<=0 || seed<=0 || nu0<0 || temperature<=0 || r_assoc<=0) error->all(FLERR,"Illegal associating kinetics parameters");
    first=seed; kinetics=1;
  }
  peratom_flag = 1;
  size_peratom_cols = 0;
  peratom_freq = 1;
  restart_peratom = 1;
  vector_flag=1; size_vector=3; global_freq=1;
  comm_border = 1;
  comm_forward = 1;
  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);
  atom->add_callback(Atom::BORDER);
}
FixAssociatingKinetics::~FixAssociatingKinetics()
{
  atom->delete_callback(id,Atom::GROW);
  atom->delete_callback(id,Atom::RESTART);
  atom->delete_callback(id,Atom::BORDER);
  memory->destroy(partner);
}
int FixAssociatingKinetics::setmask() { return kinetics ? END_OF_STEP : 0; }
void FixAssociatingKinetics::init()
{
  if (kinetics && comm->nprocs != 1) error->all(FLERR,"Fix associating/kinetics B1 supports one MPI rank only");
  if (kinetics) { auto *req=neighbor->add_request(this,NeighConst::REQ_FULL|NeighConst::REQ_OCCASIONAL); req->set_cutoff_fixed(r_assoc); }
  if (debug_pair) initialize_debug_pair();
  comm->forward_comm(this);
}
void FixAssociatingKinetics::init_list(int,NeighList *ptr) { list=ptr; }
static unsigned long long ahash(unsigned long long x) { x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
void FixAssociatingKinetics::end_of_step()
{
  if (update->ntimestep % nevery || !list) return;
  struct Edge { int i,j; unsigned long long p; }; std::vector<Edge> edges;
  for(int ii=0;ii<list->inum;++ii) { int i=list->ilist[ii]; if(!(atom->mask[i]&groupbit)) continue; int *n=list->firstneigh[i];
    for(int jj=0;jj<list->numneigh[i];++jj) { int j=n[jj]&NEIGHMASK; if(!(atom->mask[j]&groupbit)||((n[jj]>>SBBITS)&3)==1) continue; if(atom->tag[i]>=atom->tag[j]) continue;
      double dx=atom->x[i][0]-atom->x[j][0],dy=atom->x[i][1]-atom->x[j][1],dz=atom->x[i][2]-atom->x[j][2]; domain->minimum_image(FLERR,dx,dy,dz); if(dx*dx+dy*dy+dz*dz<r_assoc*r_assoc) edges.push_back({i,j,ahash(first^update->ntimestep^atom->tag[i]^(atom->tag[j]<<1))}); }}
  std::sort(edges.begin(),edges.end(),[](const Edge&a,const Edge&b){return a.p<b.p;});
  auto *pair=dynamic_cast<PairAssociating *>(force->pair); double q=1-std::exp(-nu0*std::exp(-ea/temperature)*nevery*update->dt);
  for(auto &e:edges) { tagint pi=partner[e.i],pj=partner[e.j]; bool make=!pi&&!pj, cut=pi==atom->tag[e.j]&&pj==atom->tag[e.i]; if(!make&&!cut) continue;
    double dx=atom->x[e.i][0]-atom->x[e.j][0],dy=atom->x[e.i][1]-atom->x[e.j][1],dz=atom->x[e.i][2]-atom->x[e.j][2]; domain->minimum_image(FLERR,dx,dy,dz); double du=pair->delta_u(std::sqrt(dx*dx+dy*dy+dz*dz)); double a=make?std::min(1.0,std::exp(-du/temperature)):std::min(1.0,std::exp(du/temperature));
    if((ahash(e.p+0x517cc1b727220a95ULL)>>11)*0x1.0p-53 < q*a) {
      tagint ti=atom->tag[e.i], tj=atom->tag[e.j], mi=atom->molecule ? atom->molecule[e.i] : 0, mj=atom->molecule ? atom->molecule[e.j] : 0;
      if(tj<ti) { std::swap(ti,tj); std::swap(mi,mj); }
      accepted_events.push_back({ti,tj,mi,mj,make});
      partner[e.i]=make?atom->tag[e.j]:0; partner[e.j]=make?atom->tag[e.i]:0; if(make)++created;else++broken; }}
  comm->forward_comm(this);
}
double FixAssociatingKinetics::compute_vector(int n) {
  if(n==1) return created; if(n==2) return broken;
  bigint count=0; for(int i=0;i<atom->nlocal;++i) if(partner[i] && atom->tag[i]<partner[i]) ++count; return count;
}
void FixAssociatingKinetics::initialize_debug_pair()
{
  int found = 0;
  for (int i=0; i<atom->nlocal; ++i) {
    if (atom->tag[i] == first) {
      if (!(atom->mask[i] & groupbit)) error->all(FLERR,"Debug association atom is outside fix group");
      if (partner[i] && partner[i] != second) error->all(FLERR,"Conflicting associating partner in restart");
      partner[i] = second; ++found;
    } else if (atom->tag[i] == second) {
      if (!(atom->mask[i] & groupbit)) error->all(FLERR,"Debug association atom is outside fix group");
      if (partner[i] && partner[i] != first) error->all(FLERR,"Conflicting associating partner in restart");
      partner[i] = first; ++found;
    }
  }
  int total; MPI_Allreduce(&found,&total,1,MPI_INT,MPI_SUM,world);
  if (total != 2) error->all(FLERR,"Debug association atom ID does not exist");
}
void FixAssociatingKinetics::grow_arrays(int nmax)
{
  memory->grow(partner,nmax,"associating:partner");
  if (nmax > nmax_old) std::memset(&partner[nmax_old],0,(nmax-nmax_old)*sizeof(tagint));
  nmax_old=nmax;
}
void FixAssociatingKinetics::copy_arrays(int i,int j,int) { partner[j]=partner[i]; }
void FixAssociatingKinetics::set_arrays(int i) { partner[i]=0; }
int FixAssociatingKinetics::pack_border(int n,int *list,double *buf)
{ int m=0; for(int i=0;i<n;++i) buf[m++]=ubuf(partner[list[i]]).d; return m; }
int FixAssociatingKinetics::unpack_border(int n,int firsti,double *buf)
{ int m=0; for(int i=firsti;i<firsti+n;++i) partner[i]=(tagint)ubuf(buf[m++]).i; return m; }
int FixAssociatingKinetics::pack_forward_comm(int n,int *list,double *buf,int,int *)
{ int m=0; for(int i=0;i<n;++i) buf[m++]=ubuf(partner[list[i]]).d; return m; }
void FixAssociatingKinetics::unpack_forward_comm(int n,int firsti,double *buf)
{ int m=0; for(int i=firsti;i<firsti+n;++i) partner[i]=(tagint)ubuf(buf[m++]).i; }
int FixAssociatingKinetics::pack_exchange(int i,double *buf) { buf[0]=ubuf(partner[i]).d; return 1; }
int FixAssociatingKinetics::unpack_exchange(int i,double *buf) { partner[i]=(tagint)ubuf(buf[0]).i; return 1; }
int FixAssociatingKinetics::pack_restart(int i,double *buf) { buf[0]=2; buf[1]=ubuf(partner[i]).d; return 2; }
void FixAssociatingKinetics::unpack_restart(int i,int nth)
{ int m=0; for(int k=0;k<nth;++k) m += static_cast<int>(atom->extra[i][m]); partner[i]=(tagint)ubuf(atom->extra[i][m+1]).i; }
int FixAssociatingKinetics::maxsize_restart() { return 2; }
int FixAssociatingKinetics::size_restart(int) { return 2; }
double FixAssociatingKinetics::memory_usage() { return atom->nmax*sizeof(tagint); }
