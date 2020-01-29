/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   Coded by Jorge Ramirez - Technical University of Madrid
   jorge.ramirez@upm.es - March 2016
------------------------------------------------------------------------- */

#define _USE_MATH_DEFINES
#include <cmath>
#include "stdio.h"
#include "string.h"
#include "fix_bd_ABP.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "random_mars.h"
#include "comm.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBDABP::FixBDABP(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix bd/ABP command");

  dynamic_group_allow = 1;
  time_integrate = 1;

  T = force->numeric(FLERR, arg[3]);
  Dt = force->numeric(FLERR,arg[4]);
  Dr = force->numeric(FLERR,arg[5]);  
  Fp = force->numeric(FLERR,arg[6]);
  seed = force->inumeric(FLERR,arg[7]);

// initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + comm->me);
}

/* ---------------------------------------------------------------------- */ 

int FixBDABP::setmask()
{
  int mask = 0;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBDABP::init()
{
  double **u = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
 
  dimension = domain->dimension;

  // WE USE THE VELOCITY AS THE UNIT ORIENTATION VECTOR 
  // IN 3D IT IS NECESSARY TO CREATE VELOCITIES BEFORE CALLING THIS FIX
  // If in 2D, we use the 3rd component of the velocity to store the angle theta
  // and the "velocities" are generated in this procedure
  for (int i = 0; i < nlocal; i++)
  	if (mask[i] & groupbit) {
                if (dimension==2) {
			u[i][2] = (random->uniform() - 0.5)*2.0*M_PI;
			u[i][0] = cos(u[i][2]);
			u[i][1] = sin(u[i][2]);
		}
		else if (dimension==3) {
	  		double modu=sqrt(u[i][0]*u[i][0]+u[i][1]*u[i][1]+u[i][2]*u[i][2]);
  			u[i][0] /= modu;
  			u[i][1] /= modu;
	  		u[i][2] /= modu;
		}
  	}

  //Dr = 3.0*Dt;
  dtDt_T = update->dt*Dt/T/force->mvv2e; 
  sqrt2Dtdt = sqrt(2.0*Dt*update->dt);
  sqrt2Drdt = sqrt(2.0*Dr*update->dt);
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */
void FixBDABP::final_integrate()
{
  double dtfm;

  // update v and x of atoms in group

  double **x = atom->x;
  double **u = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
  	if (mask[i] & groupbit) {
		if (dimension == 2) {
			// TRANSLATION OF CENTER OF MASS
			x[i][0]+=dtDt_T*(f[i][0]+Fp*u[i][0])+sqrt2Dtdt*random->gaussian();
			x[i][1]+=dtDt_T*(f[i][1]+Fp*u[i][1])+sqrt2Dtdt*random->gaussian();

			// ROTATION OF UNIT VECTOR
                        u[i][2]+=sqrt2Drdt*random->gaussian();
                        u[i][0] = cos(u[i][2]);
                        u[i][1] = sin(u[i][2]);
		}
		else if (dimension == 3) {
                        // TRANSLATION OF CENTER OF MASS
			x[i][0]+=dtDt_T*(f[i][0]+Fp*u[i][0])+sqrt2Dtdt*random->gaussian();
                        x[i][1]+=dtDt_T*(f[i][1]+Fp*u[i][1])+sqrt2Dtdt*random->gaussian();
                        x[i][2]+=dtDt_T*(f[i][2]+Fp*u[i][2])+sqrt2Dtdt*random->gaussian();

			// ROTATION OF UNIT VECTOR
	  		double I_uu[3][3] = {{1.0-u[i][0]*u[i][0],-u[i][0]*u[i][1],-u[i][0]*u[i][2]},
  					     {-u[i][1]*u[i][0],1.0-u[i][1]*u[i][1],-u[i][1]*u[i][2]},
  					     {-u[i][2]*u[i][0],-u[i][2]*u[i][1],1.0-u[i][2]*u[i][2]}};

  			double DW[3] = {random->gaussian(), random->gaussian(), random->gaussian()};

	  		// ROTATION OF UNIT VECTOR
  			double DWr[3] = {random->gaussian(), random->gaussian(), random->gaussian()};

  			double theta[3]={0.0, 0.0, 0.0};
	  		double modtheta=0.0;
  			for (int j=0;j<3;j++) {
  				for (int k=0;k<3;k++) 
  					theta[j]+=sqrt2Drdt*I_uu[j][k]*DWr[k];
	  			modtheta+=theta[j]*theta[j];
  			}
  			modtheta=sqrt(modtheta);
	  		for (int j=0;j<3;j++)
  				theta[j]/=modtheta;
	      		double u0 = u[i][0]*cos(modtheta)+(theta[1]*u[i][2]-theta[2]*u[i][1])*sin(modtheta);
      			double u1 = u[i][1]*cos(modtheta)+(theta[2]*u[i][0]-theta[0]*u[i][2])*sin(modtheta);
  			double u2 = u[i][2]*cos(modtheta)+(theta[0]*u[i][1]-theta[1]*u[i][0])*sin(modtheta);
	 		u[i][0]= u0;
  			u[i][1]= u1;
  			u[i][2]= u2;
		}
  	}
  }

/* ---------------------------------------------------------------------- */

void FixBDABP::reset_dt()
{
  dtDt_T = update->dt*Dt/T/force->mvv2e;
  sqrt2Dtdt = sqrt(2.0*Dt*update->dt);
  sqrt2Drdt = sqrt(2.0*Dr*update->dt);
}
