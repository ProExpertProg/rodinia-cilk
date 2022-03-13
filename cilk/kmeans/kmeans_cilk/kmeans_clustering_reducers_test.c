/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**                                                                                                                                             **/
/**   Edited by: Sang-Ha  Lee                                                                                   **/
/**                              University of Virginia                                                                 **/
/**                                                                                                                                             **/
/**   Description:      No longer supports fuzzy c-means clustering;            **/
/**                                     only regular k-means clustering.                                        **/
/**                                     Simplified for main functionality: regular k-means      **/
/**                                     clustering.                                                                                     **/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"
/* #include <omp.h> */
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>

// TODO: Look into performing memory allocation of reducer views more
// cleverly.

// Methods for float summation reducer
void plusf(void *key, void *l, void *r) {
  *(float*)l += *(float*)r;
}

void zerof(void *key, void *v) {
  *(float*)v = 0;
}

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);
extern int num_omp_threads;

int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index, i;
    float min_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims)
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}

int global_nclusters = 0;
int global_nfeatures = 0;

// The type of the reducer view: a struct containing an array of
// lengths and an array of data.
typedef struct new_centers_rv {
  int *len;
  float *data;
} new_centers_rv;

void new_centers_identity(void *key, void *value) {
  new_centers_rv *new_centers = (new_centers_rv *)value;
  new_centers->len = (int*) calloc(global_nclusters, sizeof(int));
  new_centers->data = (float*)calloc(global_nclusters * global_nfeatures,
				     sizeof(float));
}

void new_centers_destroy(void *key, void *value) {
  new_centers_rv *new_centers = (new_centers_rv *)value;
  free(new_centers->data);
  free(new_centers->len);
}

void new_centers_reduce(void *key, void *left, void *right) {
  new_centers_rv *left_view = (new_centers_rv *)left;
  new_centers_rv *right_view = (new_centers_rv *)right;

  for (int i = 0; i < global_nclusters; ++i) {
    left_view->len[i] += right_view->len[i];
    for (int j = 0; j < global_nfeatures; ++j) {
      left_view->data[i * global_nfeatures + j] +=
	right_view->data[i * global_nfeatures + j];
    }
  }
}

/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{
  int      /* i, j, k,  */n=0, index, loop=0;
  // int     *new_centers_len;                   /* [nclusters]: no. of points in each cluster */
  global_nclusters = nclusters;
  global_nfeatures = nfeatures;

  new_centers_rv new_centers __attribute__((hyperobject, reducer(new_centers_reduce, new_centers_identity, new_centers_destroy)));

  /* float  **new_centers;                           /\* [nclusters][nfeatures] *\/ */
  float  **clusters;                                      /* out: [nclusters][nfeatures] */
  /* float    delta; */
  float delta __attribute__((hyperobject, reducer(plusf, zerof)));

  double   timing;

  /* int      nblocks; */
  /* int    **partial_new_centers_len; */
  /* float ***partial_new_centers; */

  
  /* nblocks = num_omp_threads;  */

  /* allocate space for returning variable clusters[] */
  clusters    = (float**) malloc(nclusters *             sizeof(float*));
  clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
  for (int i=1; i<nclusters; i++)
    clusters[i] = clusters[i-1] + nfeatures;

  /* randomly pick cluster centers */
  for (int i=0; i<nclusters; i++) {
    //n = (int)rand() % npoints;
    for (int j=0; j<nfeatures; j++)
      clusters[i][j] = feature[n][j];
    n++;
  }

  for (int i=0; i<npoints; i++)
    membership[i] = -1;

  /* need to initialize new_centers_len and new_centers[0] to all 0 */
  /* new_centers_len = (int*) calloc(nclusters, sizeof(int)); */

  /* new_centers    = (float**) malloc(nclusters *            sizeof(float*)); */
  /* new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float)); */
  /* for (int i=1; i<nclusters; i++) */
  /*   new_centers[i] = new_centers[i-1] + nfeatures; */

  new_centers.len = (int*) calloc(nclusters, sizeof(int));
  new_centers.data = (float*) calloc(nclusters * nfeatures, sizeof(float));

  do {
    delta = 0.0;
    /* omp_set_num_threads(num_omp_threads); */
    /* #pragma omp parallel \ */
    /* shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len) */
#pragma cilk grainsize 512
    cilk_for(int i = 0; i < npoints; ++i) {
      /* find the index of nestest cluster centers */                                 
      int index = find_nearest_point(feature[i],
				     nfeatures,
				     clusters,
				     nclusters);                                
      /* if membership changes, increase delta by 1 */
      if (membership[i] != index) delta += 1.0;

      /* assign the membership to object i */
      membership[i] = index;

      new_centers.len[index]++;
      for (int j = 0; j < nfeatures; ++j)
	new_centers.data[index * nfeatures + j] += feature[i][j];

    }

    /* replace old cluster centers with new_centers */
    for (int i=0; i<nclusters; i++) {
      for (int j=0; j<nfeatures; j++) {
        if (new_centers.len[i] > 0)
          clusters[i][j] = new_centers.data[i * nfeatures + j] / new_centers.len[i];
        new_centers.data[i * nfeatures + j] = 0.0;   /* set back to 0 */
      }
      new_centers.len[i] = 0;   /* set back to 0 */
    }
  } while (delta > threshold && loop++ < 500);

  /* free(new_centers[0]); */
  /* free(new_centers); */
  /* free(new_centers_len); */
  free(new_centers.data);
  free(new_centers.len);

  return clusters;
}

