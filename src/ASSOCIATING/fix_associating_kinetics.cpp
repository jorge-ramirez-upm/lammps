#include "fix_associating_kinetics.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "utils.h"
#include <cstring>
using namespace LAMMPS_NS;

FixAssociatingKinetics::FixAssociatingKinetics(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp,narg,arg), partner(nullptr), first(0), second(0), nmax_old(0)
{
  if (narg != 6 || strcmp(arg[3],"pair") != 0)
    error->all(FLERR,"Illegal fix associating/kinetics command");
  first = utils::tnumeric(FLERR,arg[4],false,lmp);
  second = utils::tnumeric(FLERR,arg[5],false,lmp);
  if (first <= 0 || second <= 0 || first == second)
    error->all(FLERR,"Invalid debug association pair");
  peratom_flag = 1;
  size_peratom_cols = 0;
  peratom_freq = 1;
  restart_peratom = 1;
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
int FixAssociatingKinetics::setmask() { return 0; }
void FixAssociatingKinetics::init()
{
  initialize_debug_pair();
  comm->forward_comm(this);
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
