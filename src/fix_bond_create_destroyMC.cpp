/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#define _USE_MATH_DEFINES
#include "fix_bond_create_destroyMC.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "mpi.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"
#include <cmath>
#include <stdlib.h>
#include <string.h>

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20
#define DELTA 16

/* ---------------------------------------------------------------------- */

FixBondCreateDestroyMC::FixBondCreateDestroyMC(LAMMPS *lmp, int narg,
                                               char **arg)
    : Fix(lmp, narg, arg) {
  if (narg < 6)
    error->all(FLERR, "Illegal fix bond/create/destroy/MC command");

  MPI_Comm_rank(world, &me);

  nevery = force->inumeric(FLERR, arg[3]);
  if (nevery <= 0)
    error->all(FLERR, "Illegal fix bond/create/destroy/MC command");

  // Gillespie dt
  dtGillespie = update->dt * nevery;

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 5;
  global_freq = 1;
  extvector = 0;

  iatomtype = force->inumeric(FLERR, arg[4]);
  jatomtype = force->inumeric(FLERR, arg[5]);
  double cutoff = force->numeric(FLERR, arg[6]);
  btype = force->inumeric(FLERR, arg[7]);

  if (iatomtype < 1 || iatomtype > atom->ntypes || jatomtype < 1 ||
      jatomtype > atom->ntypes)
    error->all(FLERR,
               "Invalid atom type in fix bond/create/destroy/MC command");
  if (cutoff < 0.0)
    error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,
               "Invalid bond type in fix bond/create/destroy/MC command");

  cutsq = cutoff * cutoff;

  // optional keywords

  imaxbond = 0;
  jmaxbond = 0;
  pon = 1.0;
  poff = 1.0;
  int seed = 12345;
  cutminsq = 0.0;
  diffmol = 0;
  maxG = 2.0;

  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "imax") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      imaxbond = force->inumeric(FLERR, arg[iarg + 1]);
      if (imaxbond < 0)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "jmax") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      jmaxbond = force->inumeric(FLERR, arg[iarg + 1]);
      if (jmaxbond < 0)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "energies") == 0) {
      if (iarg + 5 > narg)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      Ea = force->numeric(FLERR, arg[iarg + 1]);
      Ee = force->numeric(FLERR,
                          arg[iarg + 2]); // E- lo he cambiado a Ee para saber
                                          // que es la energia de enlace
      T = force->numeric(FLERR, arg[iarg + 3]);
      seed = force->inumeric(FLERR, arg[iarg + 4]);
      if (Ea < 0.0 || Ee < 0.0 || T < 0.0)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      if (seed <= 0)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      iarg += 5;
      kA = exp(-Ea / T); // el antiguo pon, ahora no depende de nevery
    }
    // GROUP FOR FENE POTENTIAL
    else if (strcmp(arg[iarg], "FENE") == 0) {
      if (iarg + 3 > narg)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      kFENE = force->numeric(FLERR, arg[iarg + 1]);
      RFENE = force->numeric(FLERR, arg[iarg + 2]);
      if (kFENE < 0.0 || RFENE < 0.0)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      iarg += 3;
    }
    // GROUP FOR LJ PARAMETERS
    else if (strcmp(arg[iarg], "LJ") == 0) {
      if (iarg + 3 > narg)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      sigmaLJ = force->numeric(FLERR, arg[iarg + 1]);
      epsilonLJ = force->numeric(FLERR, arg[iarg + 2]);
      if (sigmaLJ < 0.0 || epsilonLJ < 0.0)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      iarg += 3;
    } else if (strcmp(arg[iarg], "Rmin") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      double Rmin = force->numeric(FLERR, arg[iarg + 1]);
      if (Rmin < 0.0)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      cutminsq = Rmin * Rmin;
      iarg += 2;
    } else if (strcmp(arg[iarg], "maxG") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      maxG = force->numeric(FLERR, arg[iarg + 1]);
      if (maxG < 0.0)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "diffmol") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
      diffmol = force->inumeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else
      error->all(FLERR, "Illegal fix bond/create/destroy/MC command");
  }

  // pon and poff are expected to be probability per time step
  // They need to be affected by the frequency of the fix
  pon *= nevery;
  poff *= nevery;

  // error check

  if (atom->molecular != 1)
    error->all(
        FLERR,
        "Cannot use fix bond/create/destroy/MC with non-molecular systems");
  if (iatomtype == jatomtype && ((imaxbond != jmaxbond)))
    error->all(
        FLERR,
        "Inconsistent imax/jmax values in fix bond/create/destroy/MC command");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp, seed + me);

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bondcount values will be initialized in setup()

  bondcount = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

  comm_forward = MAX(2, 2 + atom->maxspecial);
  comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;
  partner = finalpartner = NULL;
  distsq = NULL;
  probability = NULL; //JAVI: Not sure if needed

  // Gillespie variables
  Gi = NULL;
  Gj = NULL;
  Gtagi = NULL;
  Gtagj = NULL;
  Gaccumaij = NULL;
  pairDist = NULL;

  maxcreate = 0;
  maxbreak = 0;
  created = NULL;
  broken = NULL;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special_one() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial * maxspecial + maxspecial];

  // zero out stats

  createcount = 0;
  createcounttotal = 0;
  breakcount = 0;
  breakcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondCreateDestroyMC::~FixBondCreateDestroyMC() {
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id, 0);

  delete random;

  // delete locally stored arrays

  memory->destroy(bondcount);
  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(distsq);
  memory->destroy(created);
  memory->destroy(broken);
  // JAVI
  memory->destroy(Gi);
  memory->destroy(Gj);
  memory->destroy(Gtagi);
  memory->destroy(Gtagj);
  memory->destroy(Gaccumaij);
  memory->destroy(pairDist);
  //
  delete[] copy;
}

/* ---------------------------------------------------------------------- */

int FixBondCreateDestroyMC::setmask() {
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateDestroyMC::init() {

  // check cutoff for iatomtype,jatomtype
  if (force->pair == NULL || cutsq > force->pair->cutsq[iatomtype][jatomtype])
    error->all(
        FLERR,
        "Fix bond/create/destroy/MC cutoff is longer than pairwise cutoff");

  // need a half neighbor list, built every Nevery steps
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateDestroyMC::init_list(int id, NeighList *ptr) { list = ptr; }

/* ---------------------------------------------------------------------- */

void FixBondCreateDestroyMC::setup(int vflag) {
  int i, j, m;

  // compute initial bondcount if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag)
    return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++)
    bondcount[i] = 0;

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0)
            error->one(FLERR, "Fix bond/create/destroy/MC needs ghost atoms "
                              "from further away");
          bondcount[m]++;
        }
      }
    }

  // if newton_bond is set, need to sum bondcount

  commflag = 1;
  stageflag = 0;
  if (newton_bond)
    comm->reverse_comm_fix(this, 1);
}

/* ---------------------------------------------------------------------- */
double FixBondCreateDestroyMC::ULJ(double rsq) {

  double rc = pow(2.0, 1.0 / 6.0) * sigmaLJ;
  if (rsq > rc * rc)
    return 0.0;
  double invrcsq = 1.0 / rc / rc;
  double invrc6 = invrcsq * invrcsq * invrcsq;
  double invrsq = sigmaLJ * sigmaLJ / rsq;
  double invr6 = invrsq * invrsq * invrsq;
  return 4.0 * epsilonLJ * (invr6 * (invr6 - 1.0) - invrc6 * (invrc6 - 1));
}

double FixBondCreateDestroyMC::UFENE(double rsq) {
  return -0.5 * kFENE * pow(RFENE, 2.0) * log(1.0 - rsq / RFENE / RFENE);
  // E- he cambiado R0FENE por RFENE que es como estaba inicializado en el .h
}

double FixBondCreateDestroyMC::UBondedSticker(double rsq) {
  // MUY CHAPUZA, HAY QUE METER 0.9678598275 A MANO, cuyo CUADRADO ES
  // 0.9367526465
  return UFENE(rsq) + ULJ(rsq) - UFENE(0.9367526465) - ULJ(0.9367526465) - Ee;
  // E- aqu� ponia epsilon pero lo he cambiado por Ee que es como hemos llamado
  // a la energia de enlace
}
// UBondedSticker es la energ�a de enlace para MC

int FixBondCreateDestroyMC::PoissonSmall(double lambda) // JAVI
{
  // Algorithm due to Donald Knuth, 1969.
  double p = 1.0, L = exp(-lambda);
  if (lambda <= 0)
    return 0;
  int k = 0;
  do {
    k++;
    p *=
        random
            ->uniform(); // JORGE, JAVI: CAMBIAR POR EL GENERADOR DE N�s ALEATORIOS DE LAMMPS
  } while (p > L);
  return k - 1;
}

int FixBondCreateDestroyMC::PoissonLarge(double lambda) {
  // "Rejection method PA" from "The Computer Generation of Poisson Random
  // Variables" by A. C. Atkinson Journal of the Royal Statistical Society
  // Series C (Applied Statistics) Vol. 28, No. 1. (1979) The article is on
  // pages 29-35. The algorithm given here is on page 32.

  double c = 0.767 - 3.36 / lambda;
  double beta = M_PI / sqrt(3.0 * lambda);
  double alpha = beta * lambda;
  double k = log(c) - lambda - log(beta);

  for (;;) {
    double u = random->uniform();
    double x = (alpha - log((1.0 - u) / u)) / beta;
    int n = (int)floor(x + 0.5);
    if (n < 0)
      continue;
    double v = random->uniform();
    double y = alpha - beta * x;
    double temp = 1.0 + exp(y);
    double lhs = y + log(v / (temp * temp));
    double rhs = k + n * log(lambda) - lgamma(n + 1);
    if (lhs <= rhs)
      return n;
  }
}

int FixBondCreateDestroyMC::GetPoisson(double lambda) {
  return (lambda < 30.0) ? PoissonSmall(lambda) : PoissonLarge(lambda);
}

/* ---------------------------------------------------------------------- */

void FixBondCreateDestroyMC::post_integrate() {
  int i, j, k, m, n, ii, jj, inum, jnum, itype, jtype, n1, n2, n3, possible, i1,
      i2;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int npairs, maxbonds;
  tagint *slist;

  if (update->ntimestep % nevery)
    return;

  // WE START CREATING BONDS
  stageflag = 0;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check
  // needs to be <= test b/c neighbor list could have been re-built in
  //   same timestep as last post_integrate() call, but afterwards
  // NOTE: no longer think is needed, due to error tests on atom->map()
  // NOTE: if delete, can also delete lastcheck and check_ghosts()

  // JORGE: THE NExt line IS NOT COMMENTED IN fix_bond_break.cpp
  // if (lastcheck <= neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // forward comm of bondcount, so ghosts have it

  commflag = 1;
  comm->forward_comm_fix(this, 1);

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(distsq);
    // Gillespie variables:
    memory->destroy(Gi);
    memory->destroy(Gj);
    memory->destroy(Gtagi);
    memory->destroy(Gtagj);
    memory->destroy(Gaccumaij);
	  memory->destroy(pairDist);
    //
    nmax = atom->nmax;
    memory->create(partner, nmax, "bond/create/destroy/MC:partner");
    memory->create(finalpartner, nmax, "bond/create/destroy/MC:finalpartner");
    memory->create(distsq, nmax, "bond/create/destroy/MC:distsq");
    probability = distsq;
    // Gillespie variables:
    memory->create(Gi, nmax, "bond/create/destroy/MC:Gi");
    memory->create(Gj, nmax, "bond/create/destroy/MC:Gj");
    memory->create(Gtagi, nmax, "bond/create/destroy/MC:Gtagi");
    memory->create(Gtagj, nmax, "bond/create/destroy/MC:Gtagj");
    memory->create(Gaccumaij, nmax, "bond/create/destroy/MC:Gaccumaij");
	  memory->create(pairDist, nmax, "bond/create/destroy/MC:pairDist");
    //
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    distsq[i] = BIG;
  }

  // loop over neighbors of my atoms
  // each atom sets one closest eligible partner atom ID to bond with

  double **x = atom->x;
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int *mask = atom->mask;
  int *type = atom->type;

  neighbor->build_one(list, 1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  npairs = 0; // JAVI: New variable, number of pairs
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit))
      continue;
    itype = type[i];
    tagint imol = molecule[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit))
        continue;
      jtype = type[j];
      tagint jmol = molecule[j];

      possible = 0;
      if (imol == jmol && diffmol)
        continue;
      if (itype == iatomtype && jtype == jatomtype) {
        if ((imaxbond == 0 || bondcount[i] < imaxbond) &&
            (jmaxbond == 0 || bondcount[j] < jmaxbond))
          possible = 1;
      } else if (itype == jatomtype && jtype == iatomtype) {
        if ((jmaxbond == 0 || bondcount[i] < jmaxbond) &&
            (imaxbond == 0 || bondcount[j] < imaxbond))
          possible = 1;
      }
      if (!possible)
        continue;

      for (k = 0; k < nspecial[i][0]; k++)
        if (special[i][k] == tag[j])
          possible = 0;
      if (!possible)
        continue;

	  // Attempt to eliminate atoms that should be bonded in another CPU
	  //if (tag[j] < tag[i] && j!=atom->map(tag[j]))
	  //   continue;
	  //if (tag[j] < tag[i] && j >= inum)
		//  continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq)
        continue;

	  for (k = 0; k < npairs; k++) {
		  if (tag[i] == Gtagi[k] && tag[j] == Gtagj[k])
			  continue;
		  if (tag[j] == Gtagi[k] && tag[i] == Gtagj[k])
			  continue;
	  }

      Gi[npairs] = i;
      Gtagi[npairs] = tag[i];
      
	  //Gj[npairs] = atom->map(tag[j]); //This could also work if needed.
      Gj[npairs] = j;
      Gtagj[npairs] = tag[j];

	  pairDist[npairs] = rsq;

	  if (npairs == 0)
		  Gaccumaij[npairs] = kA;
	  else
		  Gaccumaij[npairs] = Gaccumaij[npairs - 1] + kA;

      npairs++;
    }
  }

  ncreate =
      GetPoisson(dtGillespie *
                 Gaccumaij[npairs - 1]); // JAVI: number of bonds to create as
                                         // function of npairs and Gillespie

  // Loop to define "final" partners
  for (i = 0; i < ncreate; i++) {
    int done = 0;
    while (!done) {
      double aux = random->uniform() * Gaccumaij[npairs - 1];
      for (j = 0; j < npairs; j++) {
        if (Gaccumaij[j] > aux)
          break;
      }
      // ¿¿¿UTILIZAR OTRO ARRAY (tmppartner) QUE VAYA DESDE 0 HASTA NATOMS-1 QUE SE INICIALICE A 0 Y SE VAYA RELLENANDO A MEDIDA QUE SE CREAN ENLACES???
      // EN LUGAR DE !partner[Gi[j]], miramos !tmppartner[tag[Gi[j]]-1], y TAMBIEN !partner[Gj[j]], miramos !tmppartner[tag[Gj[j]]-1]
      if (!partner[Gi[j]] &&
          !partner[Gj[j]]) { // We use Partner as if it was Finalpartner
        partner[Gi[j]] =
            Gtagj[j]; // Gi is the id of the atom (local or ghost)
		distsq[Gi[j]] = pairDist[j];
		probability[Gi[j]] = 1.0;
        partner[Gj[j]] = Gtagi[j];
		distsq[Gj[j]] = pairDist[j];
		probability[Gj[j]] = 1.0;

        done = 1;

        /////////////////////////////////
        maxbonds = i + 1;
        for (k = 0; k < npairs; k++) {
          if (!partner[Gi[k]] && !partner[Gj[k]]) {
            maxbonds++;
          }
        }
        if (ncreate > maxbonds) {
          ncreate = maxbonds;
        }
        /////////////////////////////////
      }
    }
  }

  // reverse comm of distsq and partner
  // not needed if newton_pair off since I,J pair was seen by both procs

  commflag = 2;
  if (force->newton_pair)
    comm->reverse_comm_fix(this);

  // each atom now knows its winning partner
  // for prob check, generate random value for each atom with a bond partner
  // forward comm of partner and random value, so ghosts have it

  commflag = 2;
  comm->forward_comm_fix(this, 2);

  // create bonds for atoms I own
  // only if both atoms list each other as winning bond partner
  //   and probability constraint is satisfied
  // if other atom is owned by another proc, it should do same thing

  int **bond_type = atom->bond_type;
  int newton_bond = force->newton_bond;

  ncreate = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0)
      continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i])
      continue;

    // apply probability constraint using RN for atom with smallest ID

    // if newton_bond is set, only store with I or J
    // if not newton_bond, store bond with both I and J
    // atom J will also do this consistently, whatever proc it is on

    if (!newton_bond || tag[i] < tag[j]) {
      if (num_bond[i] == atom->bond_per_atom)
        error->one(
            FLERR,
            "New bond exceeded bonds per atom in fix bond/create/destroy/MC");
      bond_type[i][num_bond[i]] = btype;
      bond_atom[i][num_bond[i]] = tag[j];
      num_bond[i]++;
    }

    // add a 1-2 neighbor to special bond list for atom I
    // atom J will also do this, whatever proc it is on
    // need to first remove tag[j] from later in list if it appears
    // prevents list from overflowing, will be rebuilt in rebuild_special_one()

    slist = special[i];
    n1 = nspecial[i][0];
    n2 = nspecial[i][1];
    n3 = nspecial[i][2];
    for (m = n1; m < n3; m++)
      if (slist[m] == tag[j])
        break;
    if (m < n3) {
      for (n = m; n < n3 - 1; n++)
        slist[n] = slist[n + 1];
      n3--;
      if (m < n2)
        n2--;
    }
    if (n3 == atom->maxspecial)
      error->one(
          FLERR,
          "New bond exceeded special list size in fix bond/create/destroy/MC");
    for (m = n3; m > n1; m--)
      slist[m] = slist[m - 1];
    slist[n1] = tag[j];
    nspecial[i][0] = n1 + 1;
    nspecial[i][1] = n2 + 1;
    nspecial[i][2] = n3 + 1;

    // increment bondcount
    // atom J will also do this, whatever proc it is on

    bondcount[i]++;

    // store final created bond partners and count the created bond once

    finalpartner[i] = tag[j];
    finalpartner[j] = tag[i];
    if (tag[i] < tag[j])
      ncreate++;
  }

  // tally stats

  MPI_Allreduce(&ncreate, &createcount, 1, MPI_INT, MPI_SUM, world);
  createcounttotal += createcount;
  atom->nbonds += createcount;

  // trigger reneighboring if any bonds were formed
  // this insures neigh lists will immediately reflect the topology changes
  // done if any bonds created

  if (createcount)
    next_reneighbor = update->ntimestep;
  // if (!createcount) return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect created bonds

  commflag = 3;
  comm->forward_comm_fix(this);

  // create list of broken bonds that influence my owned atoms
  //   even if between owned-ghost or ghost-ghost atoms
  // finalpartner is now set for owned and ghost atoms so loop over nall
  // OK if duplicates in broken list due to ghosts duplicating owned atoms
  // check J < 0 to insure a broken bond to unknown atom is included
  //   i.e. a bond partner outside of cutoff length

  ncreate = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0)
      continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (ncreate == maxcreate) {
        maxcreate += DELTA;
        memory->grow(created, maxcreate, 2, "bond/create/destroy/MC:created");
      }
      created[ncreate][0] = tag[i];
      created[ncreate][1] = finalpartner[i];
      ncreate++;
    }
  }

  // update special neigh lists of all atoms affected by any created bond
  // also add angles/dihedrals/impropers induced by created bonds

  update_topology();

  // DEBUG
  // print_bb();

  //////////////////////////////////////////////////////////////////////
  // BOND BREAK SECTION
  // JORGE: Simply copy the contents of post_integrate from fix_bond_break
  stageflag = 1;
  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    distsq[i] = 0.0;
  }

  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  // loop over bond list
  // setup possible partner list of bonds to break
  npairs = 0;
  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    int typeb = bondlist[n][2];
    if (!(mask[i1] & groupbit))
      continue;
    if (!(mask[i2] & groupbit))
      continue;
    if (typeb != btype)
      continue;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    rsq = delx * delx + dely * dely + delz * delz;
    if (rsq <= cutminsq)
      continue;

    // Gillespie criterium
    Gi[npairs] = i1;
	//Gj[npairs] = atom->map(tag[i2]); // This would also work
	Gj[npairs] = i2;

	pairDist[npairs] = rsq;

    double fact = exp(UBondedSticker(rsq) / T);
    if (fact > maxG)
      fact = maxG;
    if (npairs == 0)
      Gaccumaij[npairs] = kA * fact;
    else
      Gaccumaij[npairs] = Gaccumaij[npairs - 1] + kA * fact;

    npairs++;
  }

  // Start loop for "final" partners to break
  nbreak = GetPoisson(dtGillespie * Gaccumaij[npairs - 1]);
  if (nbreak > npairs)
    nbreak = npairs;

  for (i = 0; i < nbreak; i++) {
    int done = 0;
    while (!done) {
      double aux = random->uniform() * Gaccumaij[npairs - 1];
      for (j = 0; j < npairs; j++) {
        if (Gaccumaij[j] > aux)
          break;
      }

      if (!partner[Gi[j]] && !partner[Gj[j]]) {
        partner[Gi[j]] = tag[Gj[j]];
		distsq[Gi[j]] = pairDist[j];
		probability[Gi[j]] = 1.0;

        partner[Gj[j]] = tag[Gi[j]];
		distsq[Gj[j]] = pairDist[j];
		probability[Gj[j]] = 1.0;

        done = 1;
      }
    }
  }

  // reverse comm of partner info
  if (force->newton_bond)
    comm->reverse_comm_fix(this);

  // each atom now knows its winning partner
  // for prob check, generate random value for each atom with a bond partner
  // forward comm of partner and random value, so ghosts have it

  // JAVI: No need to set probability
  /*if (poff < 1.0) {
    for (i = 0; i < nlocal; i++)
      if (partner[i]) probability[i] = random->uniform();
  }*/
  // JAVI: End of no-needed probability

  commflag = 1;
  comm->forward_comm_fix(this, 2);

  // break bonds
  // if both atoms list each other as winning bond partner
  // and probability constraint is satisfied
  bond_type = atom->bond_type;
  bond_atom = atom->bond_atom;
  num_bond = atom->num_bond;
  nspecial = atom->nspecial;
  special = atom->special;

  nbreak = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0)
      continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i])
      continue;

    // delete bond from atom I if I stores it
    // atom J will also do this

    for (m = 0; m < num_bond[i]; m++) {
      if (bond_atom[i][m] == partner[i]) {
        for (k = m; k < num_bond[i] - 1; k++) {
          bond_atom[i][k] = bond_atom[i][k + 1];
          bond_type[i][k] = bond_type[i][k + 1];
        }
        num_bond[i]--;
        break;
      }
    }

    // remove J from special bond list for atom I
    // atom J will also do this, whatever proc it is on

    slist = special[i];
    n1 = nspecial[i][0];
    for (m = 0; m < n1; m++)
      if (slist[m] == partner[i])
        break;
    n3 = nspecial[i][2];
    for (; m < n3 - 1; m++)
      slist[m] = slist[m + 1];
    nspecial[i][0]--;
    nspecial[i][1]--;
    nspecial[i][2]--;

    // Decrement bondcount
    // Atom J will also do this, whatever proc it is on
    bondcount[i]--;

    // store final broken bond partners and count the broken bond once
    finalpartner[i] = tag[j];
    finalpartner[j] = tag[i];
    if (tag[i] < tag[j])
      nbreak++;
  }

  // tally stats
  MPI_Allreduce(&nbreak, &breakcount, 1, MPI_INT, MPI_SUM, world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;

  // trigger reneighboring if any bonds were broken
  // this insures neigh lists will immediately reflect the topology changes
  // done if no bonds broken
  if (breakcount)
    next_reneighbor = update->ntimestep;
  if (!breakcount)
    return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds
  commflag = 2;
  comm->forward_comm_fix(this);

  // create list of broken bonds that influence my owned atoms
  //   even if between owned-ghost or ghost-ghost atoms
  // finalpartner is now set for owned and ghost atoms so loop over nall
  // OK if duplicates in broken list due to ghosts duplicating owned atoms
  // check J < 0 to insure a broken bond to unknown atom is included
  //   i.e. bond partner outside of cutoff length

  nbreak = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0)
      continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (nbreak == maxbreak) {
        maxbreak += DELTA;
        memory->grow(broken, maxbreak, 2, "bond/break:broken");
      }
      broken[nbreak][0] = tag[i];
      broken[nbreak][1] = finalpartner[i];
      nbreak++;
    }
  }

  // update special neigh lists of all atoms affected by any broken bond
  // also remove angles/dihedrals/impropers broken by broken bonds
  update_topology_break();
  // update_topology();

  // DEBUG
  // print_bb();
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly created
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is added
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixBondCreateDestroyMC::check_ghosts() {
  int i, j, n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0)
        flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
  if (flagall)
    error->all(
        FLERR,
        "Fix bond/create/destroy/MC needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
   double loop over my atoms and created bonds
   influenced = 1 if atom's topology is affected by any created bond
     yes if is one of 2 atoms in bond
     yes if either atom ID appears in as 1-2 or 1-3 in atom's special list
     else no
   if influenced by any created bond:
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
     check for angles/dihedrals/impropers to create due modified special list
------------------------------------------------------------------------- */

void FixBondCreateDestroyMC::update_topology() {
  int i, j, k, n, influence, influenced;
  tagint id1, id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  overflow = 0;

  // DEBUG
  // printf("NCREATE %d: ",ncreate);
  // for (i = 0; i < ncreate; i++)
  //  printf(" %d %d,",created[i][0],created[i][1]);
  // printf("\n");
  // END DEBUG

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < ncreate; j++) {
      id1 = created[j][0];
      id2 = created[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2)
        influence = 1;
      else {
        n = nspecial[i][1];
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2) {
            influence = 1;
            break;
          }
      }
      if (!influence)
        continue;
      influenced = 1;
    }

    // rebuild_special_one() first, since used by create_angles, etc

    if (influenced) {
      rebuild_special_one(i);
    }
  }

  int overflowall;
  MPI_Allreduce(&overflow, &overflowall, 1, MPI_INT, MPI_SUM, world);
  if (overflowall)
    error->all(FLERR, "Fix bond/create/destroy/MC induced too many "
                      "angles/dihedrals/impropers per atom");

  int newton_bond = force->newton_bond;
}

/* ----------------------------------------------------------------------
   double loop over my atoms and broken bonds
   influenced = 1 if atom's topology is affected by any broken bond
         yes if is one of 2 atoms in bond
         yes if both atom IDs appear in atom's special list
         else no
   if influenced:
         check for angles/dihedrals/impropers to break due to specific broken
bonds rebuild the atom's special list of 1-2,1-3,1-4 neighs
------------------------------------------------------------------------- */

void FixBondCreateDestroyMC::update_topology_break() {
  int i, j, k, n, influence, influenced, found;
  tagint id1, id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  // nangles = 0;
  // ndihedrals = 0;
  // nimpropers = 0;

  // printf("NBREAK %d: ",nbreak);
  // for (i = 0; i < nbreak; i++)
  //  printf(" %d %d,",broken[i][0],broken[i][1]);
  // printf("\n");

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < nbreak; j++) {
      id1 = broken[j][0];
      id2 = broken[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2)
        influence = 1;
      else {
        n = nspecial[i][2];
        found = 0;
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2)
            found++;
        if (found == 2)
          influence = 1;
      }
      if (!influence)
        continue;
      influenced = 1;

      // if (angleflag) break_angles(i, id1, id2);
      // if (dihedralflag) break_dihedrals(i, id1, id2);
      // if (improperflag) break_impropers(i, id1, id2);
    }

    if (influenced)
      rebuild_special_one(i);
  }

  int newton_bond = force->newton_bond;

  // int all;
  // if (angleflag) {
  //	MPI_Allreduce(&nangles, &all, 1, MPI_INT, MPI_SUM, world);
  //	if (!newton_bond) all /= 3;
  //	atom->nangles -= all;
  //}
  // if (dihedralflag) {
  //	MPI_Allreduce(&ndihedrals, &all, 1, MPI_INT, MPI_SUM, world);
  //	if (!newton_bond) all /= 4;
  //	atom->ndihedrals -= all;
  //}
  // if (improperflag) {
  //	MPI_Allreduce(&nimpropers, &all, 1, MPI_INT, MPI_SUM, world);
  //	if (!newton_bond) all /= 4;
  //	atom->nimpropers -= all;
  //}
}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs
------------------------------------------------------------------------- */

void FixBondCreateDestroyMC::rebuild_special_one(int m) {
  int i, j, n, n1, cn1, cn2, cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(
          FLERR,
          "Fix bond/create/destroy/MC needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m])
        copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1, cn2, copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR,
               "Special list size exceeded in fix bond/create/destroy/MC");

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(
          FLERR,
          "Fix bond/create/destroy/MC needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m])
        copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2, cn3, copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR,
               "Special list size exceeded in fix bond/create/destroy/MC");

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m], copy, cn3 * sizeof(int));
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates
------------------------------------------------------------------------- */

int FixBondCreateDestroyMC::dedup(int nstart, int nstop, tagint *copy) {
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop - 1];
        nstop--;
        break;
      }
    if (i == m)
      m++;
  }

  return nstop;
}

/* ---------------------------------------------------------------------- */

int FixBondCreateDestroyMC::pack_forward_comm(int n, int *list, double *buf,
                                              int pbc_flag, int *pbc) {
  int i, j, k, m, ns;

  if (stageflag==0) {

    m = 0;

    if (commflag == 1)
    {
      for (i = 0; i < n; i++)
      {
        j = list[i];
        buf[m++] = ubuf(bondcount[j]).d;
      }
      return m;
    }

    if (commflag == 2)
    {
      for (i = 0; i < n; i++)
      {
        j = list[i];
        buf[m++] = ubuf(partner[j]).d;
        buf[m++] = probability[j];
      }
      return m;
    }

    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      buf[m++] = ubuf(finalpartner[j]).d;
      ns = nspecial[j][0];
      buf[m++] = ubuf(ns).d;
      for (k = 0; k < ns; k++)
        buf[m++] = ubuf(special[j][k]).d;
    }
    return m;
  }
  else {
    if (commflag == 1)
    {
      m = 0;
      for (i = 0; i < n; i++)
      {
        j = list[i];
        buf[m++] = ubuf(partner[j]).d;
        buf[m++] = probability[j];
      }
      return m;
    }

    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      buf[m++] = ubuf(finalpartner[j]).d;
      ns = nspecial[j][0];
      buf[m++] = ubuf(ns).d;
      for (k = 0; k < ns; k++)
        buf[m++] = ubuf(special[j][k]).d;
    }
    return m;
  }
}

/* ---------------------------------------------------------------------- */

void FixBondCreateDestroyMC::unpack_forward_comm(int n, int first,
                                                 double *buf) {
  int i, j, m, ns, last;

  if (stageflag == 0)
  {

    m = 0;
    last = first + n;

    if (commflag == 1)
    {
      for (i = first; i < last; i++)
        bondcount[i] = (int)ubuf(buf[m++]).i;
    }
    else if (commflag == 2)
    {
      for (i = first; i < last; i++)
      {
        partner[i] = (tagint)ubuf(buf[m++]).i;
        probability[i] = buf[m++];
      }
    }
    else
    {
      int **nspecial = atom->nspecial;
      tagint **special = atom->special;

      m = 0;
      last = first + n;
      for (i = first; i < last; i++)
      {
        finalpartner[i] = (tagint)ubuf(buf[m++]).i;
        ns = (int)ubuf(buf[m++]).i;
        nspecial[i][0] = ns;
        for (j = 0; j < ns; j++)
          special[i][j] = (tagint)ubuf(buf[m++]).i;
      }
    }
  }
  else
  {
    if (commflag == 1)
    {
      m = 0;
      last = first + n;
      for (i = first; i < last; i++)
      {
        partner[i] = (tagint)ubuf(buf[m++]).i;
        probability[i] = buf[m++];
      }
    }
    else
    {

      int **nspecial = atom->nspecial;
      tagint **special = atom->special;

      m = 0;
      last = first + n;
      for (i = first; i < last; i++)
      {
        finalpartner[i] = (tagint)ubuf(buf[m++]).i;
        ns = (int)ubuf(buf[m++]).i;
        nspecial[i][0] = ns;
        for (j = 0; j < ns; j++)
          special[i][j] = (tagint)ubuf(buf[m++]).i;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixBondCreateDestroyMC::pack_reverse_comm(int n, int first, double *buf) {
  int i, m, last;

  if (stageflag == 0)
  {

    m = 0;
    last = first + n;

    if (commflag == 1)
    {
      for (i = first; i < last; i++)
        buf[m++] = ubuf(bondcount[i]).d;
      return m;
    }

    for (i = first; i < last; i++)
    {
      buf[m++] = ubuf(partner[i]).d;
      buf[m++] = distsq[i];
    }
    return m;
  }
  else
  {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++)
    {
      buf[m++] = ubuf(partner[i]).d;
      buf[m++] = distsq[i];
    }
    return m;
  }
}

/* ---------------------------------------------------------------------- */

void FixBondCreateDestroyMC::unpack_reverse_comm(int n, int *list,
                                                 double *buf) {
  int i, j, m;

  if (stageflag == 0)
  {
    m = 0;

    if (commflag == 1)
    {
      for (i = 0; i < n; i++)
      {
        j = list[i];
        bondcount[j] += (int)ubuf(buf[m++]).i;
      }
    }
    else
    {
      for (i = 0; i < n; i++)
      {
        j = list[i];
        if (buf[m + 1] < distsq[j])
        {
          partner[j] = (tagint)ubuf(buf[m++]).i;
          distsq[j] = buf[m++];
        }
        else
          m += 2;
      }
    }
  }
  else
  {
    m = 0;
    for (i = 0; i < n; i++)
    {
      j = list[i];
      if (buf[m + 1] > distsq[j])
      {
        partner[j] = (tagint)ubuf(buf[m++]).i;
        distsq[j] = buf[m++];
      }
      else
        m += 2;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixBondCreateDestroyMC::grow_arrays(int nmax) {
  memory->grow(bondcount, nmax, "bond/create/destroy/MC:bondcount");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixBondCreateDestroyMC::copy_arrays(int i, int j, int delflag) {
  bondcount[j] = bondcount[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixBondCreateDestroyMC::pack_exchange(int i, double *buf) {
  buf[0] = bondcount[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixBondCreateDestroyMC::unpack_exchange(int nlocal, double *buf) {
  bondcount[nlocal] = static_cast<int>(buf[0]);
  return 1;
}

/* ---------------------------------------------------------------------- */

double FixBondCreateDestroyMC::compute_vector(int n) {
  if (n == 0)
    return (double)createcount;
  if (n == 1)
    return (double)breakcount;
  if (n == 2)
    return (double)createcounttotal;
  if (n == 3)
    return (double)breakcounttotal;
  return (double)createcounttotal - breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondCreateDestroyMC::memory_usage() {
  int nmax = atom->nmax;
  double bytes = nmax * sizeof(int);
  bytes = 2 * nmax * sizeof(tagint);
  bytes += nmax * sizeof(double);

  /* JR, JAVI: ESTIMAR LA MEMORIA UTILIZADA POR NUESTRO FIX */

  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateDestroyMC::print_bb() {
  for (int i = 0; i < atom->nlocal; i++) {
    printf("TAG " TAGINT_FORMAT ": %d nbonds: ", atom->tag[i],
           atom->num_bond[i]);
    for (int j = 0; j < atom->num_bond[i]; j++) {
      printf(" " TAGINT_FORMAT, atom->bond_atom[i][j]);
    }
    printf("\n");
    printf("TAG " TAGINT_FORMAT ": %d %d %d nspecial: ", atom->tag[i],
           atom->nspecial[i][0], atom->nspecial[i][1], atom->nspecial[i][2]);
    for (int j = 0; j < atom->nspecial[i][2]; j++) {
      printf(" " TAGINT_FORMAT, atom->special[i][j]);
    }
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixBondCreateDestroyMC::print_copy(const char *str, tagint m, int n1,
                                        int n2, int n3, int *v) {
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ", str, m, n1, n2, n3);
  for (int j = 0; j < n3; j++)
    printf(" %d", v[j]);
  printf("\n");
}
