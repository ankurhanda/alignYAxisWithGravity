/*
 * Author: Ankur Handa
 * Date:   26 Mar 2015
 * University of Cambridge
 *
 * */

#pragma once

#undef isfinite
#undef isnan

#include <stdio.h>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>
#include <boost/math/common_factor.hpp>

namespace GPUReduction{

struct JTJ3x3_shfl{

    float    j00,   j01, j02;
    float  /*j10,*/ j11, j12;
    float  /*j20, j21,*/ j22;

    float   j00e, j01e, j02e;
    float inliers;

//    __host__ __device__ JTJ3x3_shfl(float _j00=0,
//           float _j01=0,
//           float _j02=0,

//           float _j11=0,
//           float _j12=0,

//           float _j22=0,

//           float _j00e=0,
//           float _j01e=0,
//           float _j02e=0,
//           float _inliers=0):

//        j00(_j00),
//        j01(_j01),
//        j02(_j02),

//        j11(_j11),
//        j12(_j12),

//        j22(_j22),
//        j00e(_j00e),
//        j01e(_j01e),
//        j02e(_j02e),
//        inliers(_inliers)
//    {

//    }


    __device__ inline void add(const JTJ3x3_shfl & a)
       {
           j00 += a.j00;
           j01 += a.j01;
           j02 += a.j02;

           j11 += a.j11;
           j12 += a.j12;

           j22 += a.j22;

           j00e += a.j00e;
           j01e += a.j01e;
           j02e += a.j02e;

           inliers += a.inliers;
       }
};


//__host__ __device__ inline JTJ3x3_shfl make_JTJ3x3_shfl(float _j00,
//                                             float _j01,
//                                             float _j02,

//                                             float _j11,
//                                             float _j12,
//                                             float _j22,

//                                             float _j00e,
//                                             float _j01e,
//                                             float _j02e,
//                                             float _inliers)
//{
//   return JTJ3x3_shfl(_j00,
//                      _j01,
//                      _j02,

//                      _j11,
//                      _j12,

//                      _j22,

//                     _j00e,
//                     _j01e,
//                     _j02e,

//                    _inliers);

//}

void alignYAxisWithGravity(float4 *d_normals,
                           const unsigned int stridef4,
                           const unsigned int width,
                           const unsigned int height,
                           float4 &gravity_dir,
                           std::vector<float> &degree_thresholds,
                           std::vector<int> &max_iterations);

void reductionCheck(const unsigned int width,
                    const unsigned int height);
}
