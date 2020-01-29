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
#include <cmath>
#include <stdlib.h>
#include "pair_RABP.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include <string.h>
#include "atom_vec_ellipsoid.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairRABP::PairRABP(LAMMPS *lmp) : Pair(lmp)
{
    single_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairRABP::~PairRABP()
{
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);

        memory->destroy(cut_lj);
        memory->destroy(cut_ljsq);
        //memory->destroy(cut_coul);
        //memory->destroy(cut_coulsq);
        memory->destroy(epsilon);
        memory->destroy(sigma);
        memory->destroy(epsilon_rep);
        memory->destroy(theta_max);
        memory->destroy(theta_tail);
        memory->destroy(with_the_patch);
        memory->destroy(ljRf);
        memory->destroy(ljRe);
        memory->destroy(ljAe1);
        memory->destroy(ljAe2);
        memory->destroy(ljAf1);
        memory->destroy(ljAf2);
        //memory->destroy(lj2);
        //memory->destroy(lj3);
        //memory->destroy(lj4);
        //memory->destroy(offset);
    }
}

/* ---------------------------------------------------------------------- */

void PairRABP::compute(int eflag, int vflag)
{
    int i, j, ii, jj, inum, jnum, itype, jtype;
    double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fx, fy, fz;
    double rsq, rinv, r2inv, r6inv, r3inv, r5inv, r7inv, r;
    double rs, ratract6inv, ratractinv;
    double forcecoulx, forcecouly, forcecoulz, crossx, crossy, crossz;
    double tixcoul, tiycoul, tizcoul, tjxcoul, tjycoul, tjzcoul;
    double tljattrx, tljattry, tljattrz;
    double fq, pdotp, pidotr, pjdotr, pre1, pre2, pre3, pre4;
    double forcelj, forceljrep, forceljattr, factor_coul, factor_lj;
    int *ilist, *jlist, *numneigh, **firstneigh;

    evdwl = ecoul = 0.0;
    if (eflag || vflag) ev_setup(eflag, vflag);
    else evflag = vflag_fdotr = 0;

    double **x = atom->x;
    double **f = atom->f;
    //double *q = atom->q;
    //double **mu = atom->mu;
    double **torque = atom->torque;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    //double *special_coul = force->special_coul;
    double *special_lj = force->special_lj;
    int newton_pair = force->newton_pair;
    double qqrd2e = force->qqrd2e;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    // quaternion operations
    double *shapei, *quati, *shapej, *quatj;
    AtomVecEllipsoid::Bonus *bonus = avec->bonus;
    int *ellipsoid = atom->ellipsoid;

    // loop over neighbors of my atoms

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        //qtmp = q[i];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        // Calculate the director vector of particle i (ex of the local reference of the particle)
        shapei = bonus[ellipsoid[i]].shape; // Is this necessary?
        quati = bonus[ellipsoid[i]].quat;
        double mui[3], p_i[3];
        mui[0] = quati[0] * quati[0] + quati[1] * quati[1] - quati[2] * quati[2] - quati[3] * quati[3];
        mui[1] = 2.0 * (quati[1] * quati[2] + quati[0] * quati[3]);
        mui[2] = 2.0 * (quati[1] * quati[3] - quati[0] * quati[2]);
        if (with_the_patch[itype][itype])
            for (int d = 0; d < 3; d++) p_i[d] = mui[d];
        else
            for (int d = 0; d < 3; d++) p_i[d] = -mui[d];

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            factor_lj = special_lj[sbmask(j)];
            //factor_coul = special_coul[sbmask(j)];
            j &= NEIGHMASK;

            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx*delx + dely*dely + delz*delz;
            jtype = type[j];

            if (rsq < cutsq[itype][jtype]) {

                // Calculate the director vector of particle j (ex of the local reference of the particle)
                shapej = bonus[ellipsoid[j]].shape; // Is this necessary?
                quatj = bonus[ellipsoid[j]].quat;
                double muj[3], p_j[3];
                muj[0] = quatj[0] * quatj[0] + quatj[1] * quatj[1] - quatj[2] * quatj[2] - quatj[3] * quatj[3];
                muj[1] = 2.0 * (quatj[1] * quatj[2] + quatj[0] * quatj[3]);
                muj[2] = 2.0 * (quatj[1] * quatj[3] - quatj[0] * quatj[2]);

                if (with_the_patch[jtype][jtype])
                    for (int d = 0; d < 3; d++) p_j[d] = muj[d];
                else
                    for (int d = 0; d < 3; d++) p_j[d] = -muj[d];



                // IMPLEMENTAR LA INTERACCION

                r2inv = 1.0 / rsq;
                rinv = sqrt(r2inv);

                // COMPUTE cos(thetai) cos(thetaj) (delx : vector desde j hasta i
                double costhetai = (-p_i[0] * delx - p_i[1] * dely - p_i[2] * delz)*rinv;
                double costhetaj = (p_j[0] * delx + p_j[1] * dely + p_j[2] * delz)*rinv;
                double thetai = acos(costhetai);
                double thetaj = acos(costhetaj);
                double phi_i, phi_j;
                double diffphiipref = 0.0;
                double diffphijpref = 0.0;
                double diffphii_pi_pref = 0.0;
                double diffphij_pj_pref = 0.0;

                if (thetai < theta_max[itype][itype])
                    phi_i = 1.0;
                else if (thetai < theta_max[itype][itype] + theta_tail[itype][itype]) {
                    phi_i = pow(cos(M_PI_2*(thetai - theta_max[itype][itype]) / theta_tail[itype][itype]), 2.0);
                    diffphiipref = M_PI / 2.0 / theta_tail[itype][itype] * sin(M_PI*(thetai - theta_max[itype][itype]) / theta_tail[itype][itype]) / sin(thetai); //!! 
                    diffphii_pi_pref = diffphiipref;
                }
                else
                    phi_i = 0.0;
                if (thetaj < theta_max[jtype][jtype])
                    phi_j = 1.0;
                else if (thetaj < theta_max[jtype][jtype] + theta_tail[jtype][jtype]) {
                    phi_j = pow(cos(M_PI_2*(thetaj - theta_max[jtype][jtype]) / theta_tail[jtype][jtype]), 2.0);
                    diffphijpref = M_PI / 2.0 / theta_tail[jtype][jtype] * sin(M_PI*(thetaj - theta_max[jtype][jtype]) / theta_tail[jtype][jtype]) / sin(thetaj); //!! 
                    diffphij_pj_pref = diffphijpref;
                }
                else
                    phi_j = 0.0;

                // atom can have both a charge and dipole
                // i,j = charge-charge, dipole-dipole, dipole-charge, or charge-dipole

                forcecoulx = forcecouly = forcecoulz = 0.0;
                tixcoul = tiycoul = tizcoul = 0.0;
                tjxcoul = tjycoul = tjzcoul = 0.0;

                /*if (rsq < cut_coulsq[itype][jtype]) {

                  if (qtmp != 0.0 && q[j] != 0.0) {
                    r3inv = r2inv*rinv;
                    pre1 = qtmp*q[j]*r3inv;

                    forcecoulx += pre1*delx;
                    forcecouly += pre1*dely;
                    forcecoulz += pre1*delz;
                  }

                  if (mu[i][3] > 0.0 && mu[j][3] > 0.0) {
                    r3inv = r2inv*rinv;
                    r5inv = r3inv*r2inv;
                    r7inv = r5inv*r2inv;

                    pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
                    pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
                    pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;

                    pre1 = 3.0*r5inv*pdotp - 15.0*r7inv*pidotr*pjdotr;
                    pre2 = 3.0*r5inv*pjdotr;
                    pre3 = 3.0*r5inv*pidotr;
                    pre4 = -1.0*r3inv;

                    forcecoulx += pre1*delx + pre2*mu[i][0] + pre3*mu[j][0];
                    forcecouly += pre1*dely + pre2*mu[i][1] + pre3*mu[j][1];
                    forcecoulz += pre1*delz + pre2*mu[i][2] + pre3*mu[j][2];

                    crossx = pre4 * (mu[i][1]*mu[j][2] - mu[i][2]*mu[j][1]);
                    crossy = pre4 * (mu[i][2]*mu[j][0] - mu[i][0]*mu[j][2]);
                    crossz = pre4 * (mu[i][0]*mu[j][1] - mu[i][1]*mu[j][0]);

                    tixcoul += crossx + pre2 * (mu[i][1]*delz - mu[i][2]*dely);
                    tiycoul += crossy + pre2 * (mu[i][2]*delx - mu[i][0]*delz);
                    tizcoul += crossz + pre2 * (mu[i][0]*dely - mu[i][1]*delx);
                    tjxcoul += -crossx + pre3 * (mu[j][1]*delz - mu[j][2]*dely);
                    tjycoul += -crossy + pre3 * (mu[j][2]*delx - mu[j][0]*delz);
                    tjzcoul += -crossz + pre3 * (mu[j][0]*dely - mu[j][1]*delx);
                  }

                  if (mu[i][3] > 0.0 && q[j] != 0.0) {
                    r3inv = r2inv*rinv;
                    r5inv = r3inv*r2inv;
                    pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
                    pre1 = 3.0*q[j]*r5inv * pidotr;
                    pre2 = q[j]*r3inv;

                    forcecoulx += pre2*mu[i][0] - pre1*delx;
                    forcecouly += pre2*mu[i][1] - pre1*dely;
                    forcecoulz += pre2*mu[i][2] - pre1*delz;
                    tixcoul += pre2 * (mu[i][1]*delz - mu[i][2]*dely);
                    tiycoul += pre2 * (mu[i][2]*delx - mu[i][0]*delz);
                    tizcoul += pre2 * (mu[i][0]*dely - mu[i][1]*delx);
                  }

                  if (mu[j][3] > 0.0 && qtmp != 0.0) {
                    r3inv = r2inv*rinv;
                    r5inv = r3inv*r2inv;
                    pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
                    pre1 = 3.0*qtmp*r5inv * pjdotr;
                    pre2 = qtmp*r3inv;

                    forcecoulx += pre1*delx - pre2*mu[j][0];
                    forcecouly += pre1*dely - pre2*mu[j][1];
                    forcecoulz += pre1*delz - pre2*mu[j][2];
                    tjxcoul += -pre2 * (mu[j][1]*delz - mu[j][2]*dely);
                    tjycoul += -pre2 * (mu[j][2]*delx - mu[j][0]*delz);
                    tjzcoul += -pre2 * (mu[j][0]*dely - mu[j][1]*delx);
                  }
                } */

                // LJ interaction

                // Repulsive LJ
                if (rsq < cut_ljsq[itype][jtype]) {
                    r6inv = r2inv*r2inv*r2inv;
                    forceljrep = r6inv * (ljRf[itype][jtype] * r6inv);
                    forceljrep *= factor_lj * r2inv;
                }
                else forcelj = 0.0;

                // Attractive LJ (radial)
                r = sqrt(rsq);
                if (rsq < cut_ljsq[itype][jtype]) {
                    rs = fabs(r - sigma[itype][jtype]);
                    ratractinv = 1.0 / (2 * rs + 1.122462048*sigma[itype][jtype]);
                    ratract6inv = pow(ratractinv, 6.0);
                    if (r - sigma[itype][jtype] > 0) {
                        forceljattr = ratract6inv*ratractinv*(ljAf1[itype][jtype] * ratract6inv - ljAf2[itype][jtype]);
                    }
                    else {
                        forceljattr = -ratract6inv*ratractinv*(ljAf1[itype][jtype] * ratract6inv - ljAf2[itype][jtype]);
                    }
                    forceljattr *= factor_lj;
                }
                else forcelj = 0.0;

                // Attractive LJ (orientational)
                double Uattr = ratract6inv*(ljAe1[itype][jtype] * ratract6inv - ljAe2[itype][jtype])*factor_lj;
                double fattrorx = -Uattr*(diffphiipref*(p_i[0] * rinv + costhetai / rsq*delx)*phi_j +
                    diffphijpref*(p_j[0] * rinv - costhetaj / rsq*delx)*phi_i);
                double fattrory = -Uattr*(diffphiipref*(p_i[1] * rinv + costhetai / rsq*dely)*phi_j +
                    diffphijpref*(p_j[1] * rinv - costhetaj / rsq*dely)*phi_i);
                double fattrorz = -Uattr*(diffphiipref*(p_i[2] * rinv + costhetai / rsq*delz)*phi_j +
                    diffphijpref*(p_j[2] * rinv - costhetaj / rsq*delz)*phi_i);

                // Torque magnitude                                                      
                double mag_torque_i = 0.0;
                mag_torque_i = -Uattr*phi_j*diffphii_pi_pref; // negative sign due to the force = -dU/dn_i                                                                         
                double mag_torque_j = 0.0;
                mag_torque_j = -Uattr*phi_i*diffphij_pj_pref; // negative sign due to the force = -dU/dn_j                                                                         

                // total force

                fx = delx*forceljrep + delx*forceljattr*phi_i*phi_j / r + fattrorx;
                fy = dely*forceljrep + dely*forceljattr*phi_i*phi_j / r + fattrory;
                fz = delz*forceljrep + delz*forceljattr*phi_i*phi_j / r + fattrorz;

                // force & torque accumulation

                f[i][0] += fx;
                f[i][1] += fy;
                f[i][2] += fz;
                //minus mag_torque_i, since delx,dely,delz are r_i - r_j FAO               
                torque[i][0] += (-mag_torque_i * (p_i[1] * delz - p_i[2] * dely) * rinv);
                torque[i][1] += (-mag_torque_i * (p_i[2] * delx - p_i[0] * delz) * rinv);
                torque[i][2] += (-mag_torque_i * (p_i[0] * dely - p_i[1] * delx) * rinv);

                if (newton_pair || j < nlocal) {
                    f[j][0] -= fx;
                    f[j][1] -= fy;
                    f[j][2] -= fz;

                    //delx,dely,delz are r_i - r_j FAO               
                    torque[j][0] += (mag_torque_j * (p_j[1] * delz - p_j[2] * dely) * rinv);
                    torque[j][1] += (mag_torque_j * (p_j[2] * delx - p_j[0] * delz) * rinv);
                    torque[j][2] += (mag_torque_j * (p_j[0] * dely - p_j[1] * delx) * rinv);

                }

                if (eflag) {

                    if (rsq < cut_ljsq[itype][jtype]) {
                        evdwl = r6inv*(ljRe[itype][jtype] * r6inv) + Uattr*phi_i*phi_j;
                        evdwl *= factor_lj;
                    }
                    else evdwl = 0.0;
                }

                if (evflag) ev_tally_xyz(i, j, nlocal, newton_pair,
                    evdwl, ecoul, fx, fy, fz, delx, dely, delz);
            }
        }
    }

    if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairRABP::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j++)
            setflag[i][j] = 0;

    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

    memory->create(cut_lj, n + 1, n + 1, "pair:cut_lj");
    memory->create(cut_ljsq, n + 1, n + 1, "pair:cut_ljsq");
    //memory->create(cut_coul,n+1,n+1,"pair:cut_coul");
    //memory->create(cut_coulsq,n+1,n+1,"pair:cut_coulsq");
    memory->create(epsilon, n + 1, n + 1, "pair:epsilon");
    memory->create(sigma, n + 1, n + 1, "pair:sigma");
    memory->create(epsilon_rep, n + 1, n + 1, "pair:epsilon_rep");
    memory->create(theta_max, n + 1, n + 1, "pair:theta_max");
    memory->create(theta_tail, n + 1, n + 1, "pair:theta_tail");
    memory->create(with_the_patch, n + 1, n + 1, "pair:with_the_patch");

    memory->create(ljRf, n + 1, n + 1, "pair:ljRf");
    memory->create(ljRe, n + 1, n + 1, "pair:ljRe");
    memory->create(ljAe1, n + 1, n + 1, "pair:ljAe1");
    memory->create(ljAe2, n + 1, n + 1, "pair:ljAe2");
    memory->create(ljAf1, n + 1, n + 1, "pair:ljAf1");
    memory->create(ljAf2, n + 1, n + 1, "pair:ljAf2");

    //memory->create(lj2,n+1,n+1,"pair:lj2");
    //memory->create(lj3,n+1,n+1,"pair:lj3");
    //memory->create(lj4,n+1,n+1,"pair:lj4");
    //memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairRABP::settings(int narg, char **arg)
{
    if (narg != 1)
        error->all(FLERR, "Incorrect args in pair_style command");

    cut_lj_global = force->numeric(FLERR, arg[0]);
    //if (narg == 1) cut_coul_global = cut_lj_global;
    //else cut_coul_global = force->numeric(FLERR,arg[1]);

    // reset cutoffs that have been explicitly set

    if (allocated) {
        int i, j;
        for (i = 1; i <= atom->ntypes; i++)
            for (j = i + 1; j <= atom->ntypes; j++)
                if (setflag[i][j]) {
                    cut_lj[i][j] = cut_lj_global;
                    //cut_coul[i][j] = cut_coul_global;
                }
    }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairRABP::coeff(int narg, char **arg)
{
    if (narg < 8 || narg > 9)
        error->all(FLERR, "Incorrect args for pair coefficients");
    if (!allocated) allocate();

    int ilo, ihi, jlo, jhi;
    force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
    force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);

    double epsilon_one = force->numeric(FLERR, arg[2]);
    double sigma_one = force->numeric(FLERR, arg[3]);
    double epsilon_rep_one = force->numeric(FLERR, arg[4]);
    double theta_max_one = force->numeric(FLERR, arg[5]);
    double theta_tail_one = force->numeric(FLERR, arg[6]);
    int with_the_patch_one = force->inumeric(FLERR, arg[7]);

    double cut_lj_one = cut_lj_global;
    //double cut_coul_one = cut_coul_global;
    if (narg > 8) cut_lj_one = force->numeric(FLERR, arg[8]);
    //if (narg == 6) cut_coul_one = force->numeric(FLERR,arg[5]);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
        for (int j = MAX(jlo, i); j <= jhi; j++) {
            epsilon[i][j] = epsilon_one;
            sigma[i][j] = sigma_one;
            epsilon_rep[i][j] = epsilon_rep_one;
            theta_max[i][j] = theta_max_one;
            theta_tail[i][j] = theta_tail_one;
            with_the_patch[i][j] = with_the_patch_one;
            cut_lj[i][j] = cut_lj_one;
            //cut_coul[i][j] = cut_coul_one;
            setflag[i][j] = 1;
            count++;
        }
    }

    if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairRABP::init_style()
{
    if (!atom->ellipsoid_flag || !atom->torque_flag)
        error->all(FLERR, "Pair RABP requires atom attributes quaternion, torque");

    avec = (AtomVecEllipsoid *)atom->style_match("ellipsoid");
    if (!avec)
        error->all(FLERR, "Compute nve/asphere requires atom style ellipsoid");


    neighbor->request(this, instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairRABP::init_one(int i, int j)
{
    if (setflag[i][j] == 0) {
        epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j],
            sigma[i][i], sigma[j][j]);
        sigma[i][j] = mix_distance(sigma[i][i], sigma[j][j]);
        epsilon_rep[i][j] = mix_energy(epsilon_rep[i][i], epsilon_rep[j][j],
            sigma[i][i], sigma[j][j]);
        theta_max[i][j] = mix_distance(theta_max[i][i], theta_max[j][j]); // CHECK!!!
        theta_tail[i][j] = mix_distance(theta_tail[i][i], theta_tail[j][j]); // CHECK!!!
        with_the_patch[i][j] = mix_distance(with_the_patch[i][i], with_the_patch[j][j]); // CHECK!!!
        cut_lj[i][j] = mix_distance(cut_lj[i][i], cut_lj[j][j]);
        //cut_coul[i][j] = mix_distance(cut_coul[i][i],cut_coul[j][j]);
    }

    double cut = cut_lj[i][j];
    cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];
    //cut_coulsq[i][j] = cut_coul[i][j] * cut_coul[i][j];

    ljRf[i][j] = 48.0 * epsilon_rep[i][j] * pow(sigma[i][j], 12.0);
    ljRe[i][j] = 4.0 * epsilon_rep[i][j] * pow(sigma[i][j], 12.0);
    ljAe1[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
    ljAe2[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);
    ljAf1[i][j] = 96.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
    ljAf2[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);

    //lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
    //lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

    //if (offset_flag) {
    //  double ratio = sigma[i][j] / cut_lj[i][j];
    //  offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
    //} else offset[i][j] = 0.0;

    cut_ljsq[j][i] = cut_ljsq[i][j];
    //cut_coulsq[j][i] = cut_coulsq[i][j];
    ljRf[j][i] = ljRf[i][j];
    ljRe[j][i] = ljRe[i][j];
    ljAe1[j][i] = ljAe1[i][j];
    ljAe2[j][i] = ljAe2[i][j];
    //lj2[j][i] = lj2[i][j];
    //lj3[j][i] = lj3[i][j];
    //lj4[j][i] = lj4[i][j];
    //offset[j][i] = offset[i][j];

    return cut;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairRABP::write_restart(FILE *fp)
{
    write_restart_settings(fp);

    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
        for (j = i; j <= atom->ntypes; j++) {
            fwrite(&setflag[i][j], sizeof(int), 1, fp);
            if (setflag[i][j]) {
                fwrite(&epsilon[i][j], sizeof(double), 1, fp);
                fwrite(&sigma[i][j], sizeof(double), 1, fp);
                fwrite(&cut_lj[i][j], sizeof(double), 1, fp);
                //fwrite(&cut_coul[i][j],sizeof(double),1,fp);
            }
        }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairRABP::read_restart(FILE *fp)
{
    read_restart_settings(fp);

    allocate();

    int i, j;
    int me = comm->me;
    for (i = 1; i <= atom->ntypes; i++)
        for (j = i; j <= atom->ntypes; j++) {
            if (me == 0) fread(&setflag[i][j], sizeof(int), 1, fp);
            MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
            if (setflag[i][j]) {
                if (me == 0) {
                    fread(&epsilon[i][j], sizeof(double), 1, fp);
                    fread(&sigma[i][j], sizeof(double), 1, fp);
                    fread(&cut_lj[i][j], sizeof(double), 1, fp);
                    //fread(&cut_coul[i][j],sizeof(double),1,fp);
                }
                MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
                MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
                MPI_Bcast(&cut_lj[i][j], 1, MPI_DOUBLE, 0, world);
                //MPI_Bcast(&cut_coul[i][j],1,MPI_DOUBLE,0,world);
            }
        }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairRABP::write_restart_settings(FILE *fp)
{
    fwrite(&cut_lj_global, sizeof(double), 1, fp);
    //fwrite(&cut_coul_global,sizeof(double),1,fp);
    fwrite(&offset_flag, sizeof(int), 1, fp);
    fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairRABP::read_restart_settings(FILE *fp)
{
    if (comm->me == 0) {
        fread(&cut_lj_global, sizeof(double), 1, fp);
        //fread(&cut_coul_global,sizeof(double),1,fp);
        fread(&offset_flag, sizeof(int), 1, fp);
        fread(&mix_flag, sizeof(int), 1, fp);
    }
    MPI_Bcast(&cut_lj_global, 1, MPI_DOUBLE, 0, world);
    //MPI_Bcast(&cut_coul_global,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
    MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}
