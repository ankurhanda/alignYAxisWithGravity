/*
 * Author: Ankur Handa
 * Date:   26 Mar 2015
 * University of Cambridge
 *
 * */


#include <iostream>
#include "align_with_gravity.h"
#include <cuda_runtime.h>

#include <cutil_math.h>
#include <cutil.h>
#include <TooN/TooN.h>
#include <TooN/SymEigen.h>
#include <TooN/SVD.h>

# define M_PI  3.14159265358979323846

namespace GPUReduction{


inline  __device__ JTJ3x3_shfl warpReduceSum(JTJ3x3_shfl val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.j00  += __shfl_down(val.j00, offset);
        val.j01  += __shfl_down(val.j01, offset);
        val.j02  += __shfl_down(val.j02, offset);

        val.j11  += __shfl_down(val.j11, offset);
        val.j12  += __shfl_down(val.j12, offset);

        val.j22  += __shfl_down(val.j22, offset);

        val.j00e += __shfl_down(val.j00e, offset);
        val.j01e += __shfl_down(val.j01e, offset);
        val.j02e += __shfl_down(val.j02e, offset);

        val.inliers += __shfl_down(val.inliers, offset);
    }

    return val;
}

inline  __device__ JTJ3x3_shfl blockReduceSum(JTJ3x3_shfl val)
{
    static __shared__ JTJ3x3_shfl shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    /// Write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const JTJ3x3_shfl zero = {0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0};

    /// Ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(JTJ3x3_shfl * in, JTJ3x3_shfl* out, int N)
{
    JTJ3x3_shfl sum = {0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

__global__  void    cuFillwithOnes(const unsigned int width,
                                   const unsigned int height,
                                   JTJ3x3_shfl* sumN_parallel,
                                   JTJ3x3_shfl* sumN_perp,
                                   const unsigned int strideJTJ3x3_shfl)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    JTJ3x3_shfl non_zero = {1, 1, 1, 1, 1,
                            1, 1, 0, 0, 0};

    JTJ3x3_shfl  zero    = {0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0};

    if ( y < height && x < width )
    {
        sumN_parallel[y*width+x] = non_zero;
    }
    else
    {
        sumN_parallel[y*width+x] = zero;
    }

}


/// TODO: Think you can put it all together like you do the
/// other shfl reductions
__global__ void cualignYAxisWithGravity(float4* normals,
                                        const unsigned int stridef4,
                                        const unsigned int height,
                                        const unsigned int width,
                                        /// cos( degree_thresh * M_PI / 180.0f )
                                        float cos_degree_thresh_in_rad,
                                        /// sin( degree_thresh * M_PI / 180.0f )
                                        float sin_degree_thresh_in_rad,
                                        float4 gravity_dir,
                                        JTJ3x3_shfl* sumN_parallel,
                                        JTJ3x3_shfl* sumN_perp/*,
                                        const unsigned int strideJTJ3x3_shfl*/)
{

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    float4 normal = normals[y*stridef4+x];

    if ( x > width || y > height )
    {
        return;
    }

    JTJ3x3_shfl zero         = {0,
                                0,
                                0,

                                0,
                                0,

                                0,

                                0,0,0,0};


    sumN_parallel[y*width+x] = zero;
        sumN_perp[y*width+x] = zero;





    if (  normal.w == 1 )
    {
        float dot_product = dot(make_float3(normal.x,normal.y,normal.z),
                                make_float3(gravity_dir.x,gravity_dir.y,gravity_dir.z));

        JTJ3x3_shfl jacob = {normal.x * normal.x,
                             normal.x * normal.y,
                             normal.x * normal.z,

                             normal.y * normal.y,
                             normal.y * normal.z,

                             normal.z * normal.z,

                             0,0,0,0};

        if ( fabs(dot_product) > cos_degree_thresh_in_rad )
        {
            sumN_parallel[y*width+x] =   jacob;

        }
        else if ( fabs(dot_product) < sin_degree_thresh_in_rad )
        {
            sumN_perp[y*width+x]     =   jacob;
        }

    }

}

void alignYAxisWithGravity(float4* d_normals,
                           const unsigned int stridef4,
                           const unsigned int width,
                           const unsigned int height,
                           float4&  gravity_dir,
                           std::vector<float>& degree_thresholds,
                           std::vector<int>& max_iterations)
{

    JTJ3x3_shfl* sumN_parallel;
    JTJ3x3_shfl* sumN_perp;
    JTJ3x3_shfl* sumJTJ_parallel;
    JTJ3x3_shfl* sumJTJ_perp;
    JTJ3x3_shfl* out;

//    JTJ3x3_shfl* sumJTJ_perp;
//    JTJ3x3_shfl zero = {0, 0, 0, 0, 0,
//                        0, 0, 0, 0, 0};
//    unsigned int strideJTJ3x3_shfl;


    size_t pitchJTJ3x3_shfl  = width*sizeof(JTJ3x3_shfl);

    cudaMallocPitch((void **)&sumN_parallel,
                    &pitchJTJ3x3_shfl,
                    width*sizeof(JTJ3x3_shfl),
                    height);

    cudaMallocPitch((void **)&sumN_perp,
                    &pitchJTJ3x3_shfl,
                    width*sizeof(JTJ3x3_shfl),
                    height);

    pitchJTJ3x3_shfl = 1024 * sizeof(JTJ3x3_shfl);

    cudaMallocPitch((void **)&sumJTJ_parallel,
                    &pitchJTJ3x3_shfl,
                    1024*sizeof(JTJ3x3_shfl),
                    1);

    cudaMallocPitch((void **)&sumJTJ_perp,
                    &pitchJTJ3x3_shfl,
                    1024*sizeof(JTJ3x3_shfl),
                    1);

    pitchJTJ3x3_shfl = 1 * sizeof(JTJ3x3_shfl);
    cudaMallocPitch((void **)&out,
                    &pitchJTJ3x3_shfl,
                    1*sizeof(JTJ3x3_shfl),
                    1);

    float h_sumJTJ_parallel[10];
    float h_sumJTJ_perp[10];


//    cudaMemcpy(sumJTJ_parallel,&zero,sizeof(JTJ3x3_shfl),cudaMemcpyHostToDevice);

//    cudaMallocPitch((void **)&sumJTJ_perp,
//                    &pitchJTJ3x3_shfl,
//                    1 * sizeof(JTJ3x3_shfl),
//                    1);


//   strideJTJ3x3_shfl = pitchJTJ3x3_shfl / sizeof(JTJ3x3_shfl);

    const int blockWidthx = 32;
    const int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width,(unsigned int)blockWidthx),
                        boost::math::gcd<unsigned>(height,(unsigned int)blockWidthy),
                        1);

    const dim3 dimGrid(width / dimBlock.x,
                       height / dimBlock.y,
                       1);


//    float angle = 45.0f;

    for(int run = 0; run < max_iterations.size(); run++)
    {

        float angle = degree_thresholds.at(run);

        float cos_degree_thresh_in_rad = cos(angle * M_PI/180.0f);
        float sin_degree_thresh_in_rad = sin(angle * M_PI/180.0f);



        for(int itr = 0 ; itr < max_iterations.at(run); itr++)
        {

            cudaFuncSetCacheConfig(cualignYAxisWithGravity, cudaFuncCachePreferL1);

            cualignYAxisWithGravity<<<dimGrid, dimBlock>>>(d_normals,
                                                           stridef4,
                                                           height,
                                                           width,
                                                           /// cos( degree_thresh * M_PI / 180.0f )
                                                           cos_degree_thresh_in_rad,
                                                           /// sin( degree_thresh * M_PI / 180.0f )
                                                           sin_degree_thresh_in_rad,
                                                           gravity_dir,
                                                           sumN_parallel,
                                                           sumN_perp);


            int threads = 32;
            int blocks = 1024;

            reduceSum<<<blocks, threads>>>(sumN_parallel, sumJTJ_parallel, width*height);//width*height);
            reduceSum<<<1, 1024>>>(sumJTJ_parallel, out, 1024);

            cudaMemcpy((JTJ3x3_shfl*)&h_sumJTJ_parallel,
                       out,
                       sizeof(JTJ3x3_shfl),
                       cudaMemcpyDeviceToHost);

            reduceSum<<<blocks, threads>>>(sumN_perp, sumJTJ_perp, width*height);//width*height);
            reduceSum<<<1, 1024>>>(sumJTJ_perp, out, 1024);

            cudaMemcpy((JTJ3x3_shfl*)&h_sumJTJ_perp,
                       out,
                       sizeof(JTJ3x3_shfl),
                       cudaMemcpyDeviceToHost);

            TooN::Matrix<3>sumN_perp_TooN = TooN::Data(h_sumJTJ_perp[0],h_sumJTJ_perp[1],h_sumJTJ_perp[2],
                    h_sumJTJ_perp[1],h_sumJTJ_perp[3],h_sumJTJ_perp[4],
                    h_sumJTJ_perp[2],h_sumJTJ_perp[4],h_sumJTJ_perp[5]);

            TooN::Matrix<3>sumN_parallel_TooN = TooN::Data(h_sumJTJ_parallel[0],h_sumJTJ_parallel[1],h_sumJTJ_parallel[2],
                    h_sumJTJ_parallel[1],h_sumJTJ_parallel[3],h_sumJTJ_parallel[4],
                    h_sumJTJ_parallel[2],h_sumJTJ_parallel[4],h_sumJTJ_parallel[5]);



//            std::cout<<std::endl;
//            std::cout<<"sum_parallel_TooN = " << sumN_parallel_TooN << std::endl;
//            std::cout<<"sum_perp_TooN = " << sumN_perp_TooN << std::endl;

            TooN::Vector<3>yDir = TooN::makeVector(gravity_dir.x,
                                                   gravity_dir.y,
                                                   gravity_dir.z);

            /// Subtract the N_perp . N_perp^T - N_parallel . N_parallel^T
            TooN::Matrix<3>Nperp_Nparallel = sumN_perp_TooN - sumN_parallel_TooN;

            /// Do the SVD and find out the minimum eigen value
            TooN::SymEigen<3> eigN(Nperp_Nparallel);

            /// Find out the minimum eigen value
            TooN::Vector<3>eigen_vals = eigN.get_evalues();

            float min_eigen_Val = eigen_vals[0];

            int min_ind = 0;

            for(int i = 0; i < 3; i++)
            {
                if ( eigen_vals[i] < min_eigen_Val)
                {
                    min_eigen_Val = eigen_vals[i];
                    min_ind = i;
                }
            }

            TooN::Vector<3>minEigenVec = eigN.get_evectors()[min_ind];

            int sign_ = (yDir * minEigenVec > 0) -1*(yDir * minEigenVec <= 0 );

            yDir = minEigenVec * sign_;

            gravity_dir = make_float4(yDir[0],yDir[1],yDir[2],1.0f);

        }

    }

    cudaFree(sumN_parallel);
    cudaFree(sumJTJ_parallel);

    cudaFree(sumN_perp);
    cudaFree(sumJTJ_perp);

    cudaFree(out);

    cudaDeviceSynchronize();
//    delete sumN_parallel;
//    delete sumJTJ_parallel;

//    delete sumN_perp;
//    delete sumJTJ_perp;

//    delete out;



//    std::cout<<std::endl;
//    cudaSafeCall( cudaDeviceSynchronize() );


}

void reductionCheck(const unsigned int width,
                    const unsigned int height)
{

    JTJ3x3_shfl* sumN_parallel;
    JTJ3x3_shfl* sumN_perp;
    JTJ3x3_shfl* sumJTJ_parallel;
    JTJ3x3_shfl* out;

//    JTJ3x3_shfl* sumJTJ_perp;



//    JTJ3x3_shfl zero = {0, 0, 0, 0, 0,
//                        0, 0, 0, 0, 0};

//    unsigned int strideJTJ3x3_shfl;


    size_t pitchJTJ3x3_shfl  = width*sizeof(JTJ3x3_shfl);

    cudaMallocPitch((void **)&sumN_parallel,
                    &pitchJTJ3x3_shfl,
                    width*sizeof(JTJ3x3_shfl),
                    height);

    cudaMallocPitch((void **)&sumN_perp,
                    &pitchJTJ3x3_shfl,
                    width*sizeof(JTJ3x3_shfl),
                    height);

    cudaMallocPitch((void **)&sumJTJ_parallel,
                    &pitchJTJ3x3_shfl,
                    1024*sizeof(JTJ3x3_shfl),
                    1);

    cudaMallocPitch((void **)&out,
                    &pitchJTJ3x3_shfl,
                    1*sizeof(JTJ3x3_shfl),
                    1);

//    cudaMemcpy(sumJTJ_parallel,&zero,sizeof(JTJ3x3_shfl),cudaMemcpyHostToDevice);

//    cudaMallocPitch((void **)&sumJTJ_perp,
//                    &pitchJTJ3x3_shfl,
//                    1 * sizeof(JTJ3x3_shfl),
//                    1);


    //strideJTJ3x3_shfl = pitchJTJ3x3_shfl / sizeof(JTJ3x3_shfl);

    const int blockWidthx = 32;
    const int blockWidthy = 32;

    const dim3 dimBlock(boost::math::gcd<unsigned>( width,(unsigned int)blockWidthx),
                        boost::math::gcd<unsigned>(height,(unsigned int)blockWidthy),
                        1);

    const dim3 dimGrid(width / dimBlock.x,
                       height / dimBlock.y,
                       1);


    cuFillwithOnes<<<dimGrid, dimBlock>>>(width,
                                          height,
                                          sumN_parallel,
                                          sumN_perp,
                                          width);

//    float angle = 45.0f;

//    float cos_degree_thresh_in_rad = cos(angle * M_PI/180.0f);
//    float sin_degree_thresh_in_rad = sin(angle * M_PI/180.0f);

//    cualignYAxisWithGravity<<<dimGrid, dimBlock>>>(d_normals,
//                                                   stridef4,
//                                                   height,
//                                                   width,
//                                                    /// cos( degree_thresh * M_PI / 180.0f )
//                                                   cos_degree_thresh_in_rad,
//                                                    /// sin( degree_thresh * M_PI / 180.0f )
//                                                   sin_degree_thresh_in_rad,
//                                                   gravity_dir,
//                                                   sumN_parallel,
//                                                   sumN_perp);


//    int height_c = height;
//    int width_c = width;

//    cudaMemcpy(host_sumN_parallel,
//               sumN_parallel,
//               width*height*sizeof(JTJ3x3_shfl),
//               cudaMemcpyDeviceToHost);

//    JTJ3x3_shfl  zero = {1, 0, 0, 0, 0,
//                         0, 0, 0, 0, 0};

//    for(int y = 0; y < height_c; y++)
//    {
//        for(int x = 0; x < width_c; x++)
//        {
//            std::cout<<host_sumN_parallel[x+y*width_c].j00<<" ";
//        }
//        std::cout<<std::endl;
//    }

//    std::cout<<"Zero = " << zero.j00 << std::endl;

    int threads = 512;
    int blocks = 1024;

    reduceSum<<<blocks, threads>>>(sumN_parallel, sumJTJ_parallel, width*height);//width*height);
    reduceSum<<<1, 1024>>>(sumJTJ_parallel, out, 1024);



    JTJ3x3_shfl* host_sumN_parallel = new JTJ3x3_shfl[1024];

    cudaMemcpy(host_sumN_parallel,
               sumJTJ_parallel,
               1024*sizeof(JTJ3x3_shfl),
               cudaMemcpyDeviceToHost);


    for(int i = 0; i < 1024; i++)
        std::cout<<host_sumN_parallel[i].j00<<" ";
    std::cout<<std::endl;


    float host_array[10];

    cudaMemcpy((JTJ3x3_shfl*)&host_array,
               out,
               sizeof(JTJ3x3_shfl),
               cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++)
        std::cout << host_array[i] << "  ";



//    std::cout<<std::endl;
//    cudaSafeCall( cudaDeviceSynchronize() );


}



}
