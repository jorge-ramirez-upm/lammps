#include "write_associating_network.h"
#include "comm.h"
#include "error.h"
#include "fix_associating_kinetics.h"
#include "modify.h"
#include "update.h"
#include "utils.h"
#include <algorithm>
#include <cstdio>
#include <vector>
using namespace LAMMPS_NS;

void WriteAssociatingNetwork::command(int narg, char **arg)
{
  if (narg != 3 || strcmp(arg[1],"fix") != 0)
    error->all(FLERR,"Illegal write_associating_network command");
  auto *fix=dynamic_cast<FixAssociatingKinetics *>(modify->get_fix_by_id(arg[2]));
  if (!fix) error->all(FLERR,"write_associating_network requires fix associating/kinetics");
  auto local=fix->active_network();
  int nlocal=static_cast<int>(local.size());
  std::vector<int> counts(comm->me == 0 ? comm->nprocs : 0);
  MPI_Gather(&nlocal,1,MPI_INT,counts.data(),1,MPI_INT,0,world);
  std::vector<tagint> send(4*nlocal);
  for (int i=0;i<nlocal;++i) {
    send[4*i]=local[i].first; send[4*i+1]=local[i].second;
    send[4*i+2]=local[i].molecule_first; send[4*i+3]=local[i].molecule_second;
  }
  std::vector<int> counts4, offsets4;
  int total=0;
  if (comm->me == 0) {
    counts4.resize(comm->nprocs); offsets4.resize(comm->nprocs);
    for (int i=0;i<comm->nprocs;++i) { offsets4[i]=4*total; total+=counts[i]; counts4[i]=4*counts[i]; }
  }
  std::vector<tagint> received(comm->me == 0 ? 4*total : 0);
  MPI_Gatherv(send.data(),4*nlocal,MPI_LMP_TAGINT,received.data(),counts4.data(),offsets4.data(),MPI_LMP_TAGINT,0,world);
  if (comm->me != 0) return;
  std::vector<FixAssociatingKinetics::NetworkEdge> edges(total);
  for (int i=0;i<total;++i) edges[i]={received[4*i],received[4*i+1],received[4*i+2],received[4*i+3]};
  std::sort(edges.begin(),edges.end(),[](const auto &a,const auto &b) { return a.first != b.first ? a.first < b.first : a.second < b.second; });
  for (int i=0;i<total;++i) {
    if (edges[i].first >= edges[i].second || (i && edges[i-1].first == edges[i].first && edges[i-1].second == edges[i].second))
      error->all(FLERR,"write_associating_network found a non-canonical or duplicate association");
  }
  FILE *fp=fopen(arg[0],"w");
  if (!fp) error->one(FLERR,"Cannot open associating network file for writing: {}",utils::getsyserror());
  utils::print(fp,"# timestep {} bonds {}\n# tag_i tag_j molecule_i molecule_j\n",update->ntimestep,total);
  for (const auto &edge : edges) utils::print(fp,"{} {} {} {}\n",edge.first,edge.second,edge.molecule_first,edge.molecule_second);
  fclose(fp);
}
