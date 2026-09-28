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
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
using namespace LAMMPS_NS;
using namespace FixConst;

FixAssociatingKinetics::FixAssociatingKinetics(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp,narg,arg), partner(nullptr), first(0), second(0), debug_pair(0), seed(0), kinetics(0), nu0(0), ea(0), temperature(0), r_assoc(0), created(0), broken(0), list(nullptr), pair(nullptr), nmax_old(0)
{
  if (narg != 12 && narg != 9 && (narg != 3 && (narg != 6 || strcmp(arg[3],"debug_pair") != 0)))
    error->all(FLERR,"Illegal fix associating/kinetics command");
  if (narg == 6) {
    first = utils::tnumeric(FLERR,arg[4],false,lmp);
    second = utils::tnumeric(FLERR,arg[5],false,lmp);
    if (first <= 0 || second <= 0 || first == second)
      error->all(FLERR,"Invalid debug association pair");
    debug_pair = 1;
  } else if (narg == 9 || narg == 12) {
    nevery=utils::inumeric(FLERR,arg[3],false,lmp); seed=utils::inumeric(FLERR,arg[4],false,lmp);
    nu0=utils::numeric(FLERR,arg[5],false,lmp); ea=utils::numeric(FLERR,arg[6],false,lmp);
    temperature=utils::numeric(FLERR,arg[7],false,lmp); r_assoc=utils::numeric(FLERR,arg[8],false,lmp);
    if (nevery<=0 || seed<=0 || nu0<0 || ea<0 || temperature<=0 || r_assoc<=0) error->all(FLERR,"Illegal associating kinetics parameters");
    kinetics=1;
    if (narg == 12) {
      if (strcmp(arg[9],"debug_pair") != 0) error->all(FLERR,"Illegal fix associating/kinetics command");
      first=utils::tnumeric(FLERR,arg[10],false,lmp); second=utils::tnumeric(FLERR,arg[11],false,lmp);
      if(first<=0 || second<=0 || first==second) error->all(FLERR,"Invalid debug association pair"); debug_pair=1;
    }
  }
  peratom_flag = 1;
  size_peratom_cols = 0;
  peratom_freq = 1;
  restart_peratom = 1;
  vector_flag=1; size_vector=3; global_freq=1; extvector=0;
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
  pair=dynamic_cast<PairAssociating *>(force->pair_match("associating",1));
  if (!pair) error->all(FLERR,"Fix associating/kinetics requires pair associating");
  if (kinetics && r_assoc >= pair->r0_value())
    error->all(FLERR,"Fix associating/kinetics r_assoc must be smaller than associating R0");
  if (kinetics) { auto *req=neighbor->add_request(this,NeighConst::REQ_FULL|NeighConst::REQ_OCCASIONAL); req->set_cutoff_fixed(r_assoc); }
  if (debug_pair) initialize_debug_pair();
  comm->forward_comm(this);
}
void FixAssociatingKinetics::init_list(int,NeighList *ptr) { list=ptr; }
static uint64_t ahash(uint64_t x) { x+=UINT64_C(0x9e3779b97f4a7c15); x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9); x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb); return x^(x>>31); }
uint64_t FixAssociatingKinetics::random_value(uint64_t seed, bigint timestep, tagint first, tagint second, uint64_t stream)
{
  uint64_t key=seed^static_cast<uint64_t>(timestep)^(static_cast<uint64_t>(first)<<1)^(static_cast<uint64_t>(second)<<17);
  return ahash(key^stream);
}
void FixAssociatingKinetics::process_sweep(StickerStates &states, std::vector<StickerEdge> &edges, bigint timestep)
{
  const uint64_t order=UINT64_C(0x4f52444552), accept=UINT64_C(0x414343455054);
  std::sort(edges.begin(),edges.end(),[&](const StickerEdge &a,const StickerEdge &b) {
    uint64_t pa=random_value(seed,timestep,a.first,a.second,order), pb=random_value(seed,timestep,b.first,b.second,order);
    return pa!=pb ? pa<pb : (a.first!=b.first ? a.first<b.first : a.second<b.second);
  });
  double q=1-std::exp(-nu0*std::exp(-ea/temperature)*nevery*update->dt);
  for (const auto &edge : edges) {
    auto i=states.find(edge.first), j=states.find(edge.second);
    if (i==states.end() || j==states.end()) continue;
    bool make=!i->second.partner&&!j->second.partner;
    bool cut=i->second.partner==edge.second&&j->second.partner==edge.first;
    if (!make&&!cut) continue;
    double du=pair->delta_u(edge.r);
    double a=make ? std::min(1.0,std::exp(-du/temperature)) : std::min(1.0,std::exp(du/temperature));
    if ((random_value(seed,timestep,edge.first,edge.second,accept)>>11)*0x1.0p-53 >= q*a) continue;
    accepted_events.push_back({edge.first,edge.second,i->second.molecule,j->second.molecule,make});
    i->second.partner=make ? edge.second : 0;
    j->second.partner=make ? edge.first : 0;
    if (make) ++created; else ++broken;
  }
}
void FixAssociatingKinetics::end_of_step()
{
  if (update->ntimestep % nevery || !list) return;
  accepted_events.clear();
  neighbor->build_one(list);
  StickerStates states;
  for (int i=0;i<atom->nlocal;++i)
    if (atom->mask[i]&groupbit) states.emplace(atom->tag[i],StickerState{partner[i],atom->molecule ? atom->molecule[i] : 0});
  std::vector<StickerEdge> edges;
  for(int ii=0;ii<list->inum;++ii) { int i=list->ilist[ii]; if(!(atom->mask[i]&groupbit)) continue; int *n=list->firstneigh[i];
    for(int jj=0;jj<list->numneigh[i];++jj) { int j=n[jj]&NEIGHMASK; if(!(atom->mask[j]&groupbit)||((n[jj]>>SBBITS)&3)==1) continue; if(atom->tag[i]>=atom->tag[j]) continue;
      double dx=atom->x[i][0]-atom->x[j][0],dy=atom->x[i][1]-atom->x[j][1],dz=atom->x[i][2]-atom->x[j][2]; domain->minimum_image(FLERR,dx,dy,dz); double rsq=dx*dx+dy*dy+dz*dz; if(rsq<r_assoc*r_assoc) edges.push_back({atom->tag[i],atom->tag[j],std::sqrt(rsq)}); }}
  process_sweep(states,edges,update->ntimestep);
  for (int i=0;i<atom->nlocal;++i) {
    auto state=states.find(atom->tag[i]);
    if (state!=states.end()) partner[i]=state->second.partner;
  }
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
