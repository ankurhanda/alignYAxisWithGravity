#include <iostream>
#include <TooN/TooN.h>
#include <TooN/SVD.h>
#include <iostream>
#include <vector_types.h>
#include <TooN/se3.h>
#include <iu/iucore.h>
#include <iu/iucore/volume_allocator_cpu.h>
#include <iu/iucore/image_allocator_cpu.h>
#include <iu/iucore/memorydefs.h>
#include <boost/thread.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/display.h>
//#include <pangolin/glcuda.h>

#include <cvd/image.h>
#include <cvd/rgb.h>
#include <cvd/image_io.h>

//#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <iu/iucore.h>
#include <iu/iuio.h>
#include "cudakernels/marchingcubes/daniel_mc.h"

////#include "utils/vtk_utils.h"


#include "cudakernels/sdf_vol/vol.h"

#include "utils/tum_utils.h"

//#include <pangolin/pangolin.h>
//#include <pangolin/display.h>
#include "rendering/openglrendering.h"
#include "utils/cuda_mem_time_utils.h"
#include <OpenNI.h>
#include <cvd/image_io.h>
#include <OniProperties.h>

#include "utils/povray_utils.h"

////#include "FrameGrabber/frame_grabber.h"


#include "utils/align_with_gravity.h"

///// Think about PFH features.

#include<fstream>
#include<TooN/SymEigen.h>

using namespace pangolin;

#define RADPERDEG 0.0174533

void Arrow(GLdouble x1,GLdouble y1,GLdouble z1,GLdouble x2,GLdouble y2,GLdouble z2,GLdouble D)
{
  double x=x2-x1;
  double y=y2-y1;
  double z=z2-z1;
  double L=sqrt(x*x+y*y+z*z);

    GLUquadricObj *quadObj;

    glPushMatrix ();

      glTranslated(x1,y1,z1);

      if((x!=0.)||(y!=0.)) {
        glRotated(atan2(y,x)/RADPERDEG,0.,0.,1.);
        glRotated(atan2(sqrt(x*x+y*y),z)/RADPERDEG,0.,1.,0.);
      } else if (z<0){
        glRotated(180,1.,0.,0.);
      }

      glTranslatef(0,0,L-4*D);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluCylinder(quadObj, 2*D, 0.0, 4*D, 32, 1);
      gluDeleteQuadric(quadObj);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluDisk(quadObj, 0.0, 2*D, 32, 1);
      gluDeleteQuadric(quadObj);

      glTranslatef(0,0,-L+4*D);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluCylinder(quadObj, D, D, L-4*D, 32, 1);
      gluDeleteQuadric(quadObj);

      quadObj = gluNewQuadric ();
      gluQuadricDrawStyle (quadObj, GLU_FILL);
      gluQuadricNormals (quadObj, GLU_SMOOTH);
      gluDisk(quadObj, 0.0, D, 32, 1);
      gluDeleteQuadric(quadObj);

    glPopMatrix ();

}
void drawAxes(GLdouble length)
{
    glPushMatrix();
    glColor3f(1.0,0,0);
    glTranslatef(-length,0,0);
    Arrow(0,0,0, 2*length,0,0, 0.1);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0.0,1.0,0);
    glTranslatef(0,-length,0);
    Arrow(0,0,0, 0,2*length,0, 0.1);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0.0,0.0,1.0);
    glTranslatef(0,0,-length);
    Arrow(0,0,0, 0,0,2*length, 0.1);
    glPopMatrix();
}


/// https://github.com/s-gupta/rgbd/blob/38a084551d674bf9c9f2b7c40fb9edcc3ff910b3/utils/getRMatrix.m

TooN::SO3<> getRMatrix(TooN::Vector<3>yi, TooN::Vector<3>yf)
{
    yi = yi / TooN::norm(yi);
    yf = yf / TooN::norm(yf);

    float phi_rad = acos(yi*yf);
    float phi_deg = phi_rad * 180/M_PI;

    TooN::SO3<>RotMat;

    if ( abs(phi_deg) > 0.1 )
    {
        TooN::Vector<3>ax = yi ^ yf;

        RotMat = TooN::SO3<>(ax);
    }

    return RotMat;
}

/// inspired by https://github.com/s-gupta/rgbd/blob/38a084551d674bf9c9f2b7c40fb9edcc3ff910b3/utils/getYDir.m

void getYAlignedWithGravity(float4* normals,
                            float degree_thresh,
                            TooN::Vector<3>& yDir,
                            int height,
                            int width,
                            int max_iter)
{

    for(int itr = 0 ; itr < max_iter; itr++)
    {
        std::cout<<"iter = " << itr << std::endl;

        TooN::Matrix<3>sumN_perp     = TooN::Zeros(3);
        TooN::Matrix<3>sumN_parallel = TooN::Zeros(3);


//#pragma omp parallel for reduction(+:sumN_parallel_00,sumN_parallel_01,sumN_parallel_02,sumN_parallel_10,sumN_parallel_11,sumN_parallel_12,sumN_parallel_20,sumN_parallel_21,sumN_parallel_22,sumN_perp_00,sumN_perp_01,sumN_perp_02,sumN_perp_10,sumN_perp_11,sumN_perp_12,sumN_perp_20,sumN_perp_21,sumN_perp_22)

        for(int yy = 0; yy < height; yy++ )
        {
            for(int xx = 0; xx < width; xx++)
            {
                float4 normal = normals[yy*width+xx];

                if ( normal.w == 1.0f && !isnan(normal.x) && !isnan(normal.y) && !isnan(normal.z) )
                {
                    /// Do the dot product of the gravity vector at k_th iteration with the normal vector

                    float dot_product = dot(make_float3(yDir[0],yDir[1],yDir[2]),
                                            make_float3(normal.x,normal.y,normal.z));

                    /// Find out the N_perp and N_parallel
                    /// Sum the normal vector outer products of N_perp and N_parallel independently

                    TooN::Matrix<3,1>n_vec = TooN::Data(normal.x,normal.y,normal.z);

                    if ( fabs(dot_product)     > cos(degree_thresh * M_PI / 180.0f) )
                    {
                        /// Floor
                        sumN_parallel     += n_vec * n_vec.T();

//                        sumN_parallel_00    += sumN_parallel(0,0);
//                        sumN_parallel_01    += sumN_parallel(0,1);
//                        sumN_parallel_02    += sumN_parallel(0,2);

//                        sumN_parallel_10    += sumN_parallel(1,0);
//                        sumN_parallel_11    += sumN_parallel(1,1);
//                        sumN_parallel_12    += sumN_parallel(1,2);

//                        sumN_parallel_20    += sumN_parallel(2,0);
//                        sumN_parallel_21    += sumN_parallel(2,1);
//                        sumN_parallel_22    += sumN_parallel(2,2);
                    }
                    else if ( fabs(dot_product) < sin(degree_thresh * M_PI / 180.0f) )
                    {
                        /// Wall
                        sumN_perp         += n_vec * n_vec.T();

//                        sumN_perp_00    += sumN_perp(0,0);
//                        sumN_perp_01    += sumN_perp(0,1);
//                        sumN_perp_02    += sumN_perp(0,2);

//                        sumN_perp_10    += sumN_perp(1,0);
//                        sumN_perp_11    += sumN_perp(1,1);
//                        sumN_perp_12    += sumN_perp(1,2);

//                        sumN_perp_20    += sumN_perp(2,0);
//                        sumN_perp_21    += sumN_perp(2,1);
//                        sumN_perp_22    += sumN_perp(2,2);

                    }
                }
            }
        }

//        sumN_parallel(0,0) = sumN_parallel_00; sumN_parallel(0,1) = sumN_parallel_01; sumN_parallel(0,2) = sumN_parallel_02;
//        sumN_parallel(1,0) = sumN_parallel_10; sumN_parallel(1,1) = sumN_parallel_11; sumN_parallel(1,2) = sumN_parallel_12;
//        sumN_parallel(2,0) = sumN_parallel_20; sumN_parallel(2,1) = sumN_parallel_21; sumN_parallel(2,2) = sumN_parallel_22;

//        sumN_perp(0,0) = sumN_perp_00; sumN_perp(0,1) = sumN_perp_01; sumN_perp(0,2) = sumN_perp_02;
//        sumN_perp(1,0) = sumN_perp_10; sumN_perp(1,1) = sumN_perp_11; sumN_perp(1,2) = sumN_perp_12;
//        sumN_perp(2,0) = sumN_perp_20; sumN_perp(2,1) = sumN_perp_21; sumN_perp(2,2) = sumN_perp_22;

//        sumN_parallel = TooN::makeVector(sumN_parallel_0,sumN_parallel_1,sumN_parallel_2);
//        sumN_perp     = TooN::makeVector(sumN_perp_0,sumN_perp_1,sumN_perp_2);

        std::cout<<"sum_parallel = " << sumN_parallel << std::endl;

        std::cout<<"sum_perp = " << sumN_perp << std::endl;

        /// Subtract the N_perp . N_perp^T - N_parallel . N_parallel^T
        TooN::Matrix<3>Nperp_Nparallel = sumN_perp - sumN_parallel;

//        if ( itr == 0)
//        {
//            std::cout<<"sumN_perp = " << sumN_perp << std::endl;
//            std::cout<<"sumN_parallel = " << sumN_parallel << std::endl;
//            std::cout<<"Nperp_parallel = " << Nperp_Nparallel << std::endl;
//        }

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

//        if ( itr == 0 )
//            std::cout<<"eigen_vals = " << eigen_vals << std::endl;

//        TooN::Matrix<3>UMat = svdN.get_U();

//        std::cout<<"UMat = " << UMat << std::endl;

        TooN::Vector<3>minEigenVec = eigN.get_evectors()[min_ind];

        int sign_ = (yDir * minEigenVec > 0) -1*(yDir * minEigenVec <= 0 );

        yDir = minEigenVec * sign_;

    }


}

int main(void)
{

//    GPUReduction::reductionCheck(640,480);

//    Rotation for all chairs  = [0.979762 0.19166 0.0577393
//                               -0.19166 0.815034 0.546796
//                                0.0577393 -0.546796 0.835273];

//    Rotation for scr chair = [0.980317 0.189287 0.0561221
//                             -0.189287 0.820299 0.539704
//                              0.0561221 -0.539704 0.839982];

//    exit(1);

    /// Scale 1 means 640x480 images
    /// Scale 2 means 320x240 images

    float scale_val = 5000.0f;

    int scale              = 1;

//    std::string dir_base_name = "/home/ankur/workspace/code/kufrap/data/ncr/";
//    std::string dir_base_name = "/home/ankur/workspace/code/kufrap/data/viorik_bedroom_scan1";

//    std::string dir_base_name = "/home/ankur/workspace/code/kufrap/data/ncr";
//    std::string dir_base_name = "/home/ankur/workspace/code/kufrap/data/2chairs_and_tables";
    std::string dir_base_name = "/home/ankur/workspace/code/nyu_dataset/depth_images";
    std::cout<<"dir_base_name = " << dir_base_name << std::endl;

    unsigned int found = dir_base_name.find_last_of("/\\");

    std::cout<<"Try running with sudo if it crashes right after you run.." << std::endl;


    int width              = 640/scale;
    int height             = 480/scale;

    int w_width  = 640;
    int w_height = 480;
    const int UI_WIDTH = 150;

    ///This one needs old pangolin to work with.

    pangolin::CreateGlutWindowAndBind("sdfTrack",w_width+150,w_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glewInit();

    /// Create a Panel
    pangolin::View& d_panel = pangolin::CreatePanel("ui")
            .SetBounds(1.0, 0.0, 0, pangolin::Attach::Pix(150));

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
      ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      ModelViewLookAt(3,3,3, 0,0,0, AxisY)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    View& d_cam = pangolin::Display("cam")
      .SetBounds(0.0, 0.75, Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
      .SetHandler(new Handler3D(s_cam));

    View& depth_lvl0 = Display("depth_lvl0")
            .SetAspect((float)width/(float)height)
            .SetBounds(0.75,1,0.19,0.20+0.19,false);

    View& depth_lvl1 = Display("depth_lvl1")
            .SetAspect((float)width/(float)height)
            .SetBounds(0.75,1,0.20+0.19,2*0.20+0.19,false);

    View& depth_lvl2 = Display("depth_lvl2")
            .SetAspect((float)width/(float)height)
            .SetBounds(0.75,1,0.19+2*0.20,0.20*3+0.19,false);

    View& depth_lvl3 = Display("depth_lvl3")
            .SetAspect((float)width/(float)height)
            .SetBounds(0.75,1.0,0.20*3+0.19,1,false);



    // Create vertex and colour buffer objects and register them with CUDA
    GlBufferCudaPtr vertex_array_0(
        GlArrayBuffer, width * height * sizeof(float4),
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    );

    GlBufferCudaPtr colour_array_0(
        GlArrayBuffer, width * height * sizeof(uchar4),
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    );

    iu::ImageCpu_32f_C4* points3d      = new iu::ImageCpu_32f_C4(width,height);
//    iu::ImageCpu_32f_C4* d_normals      = new iu::ImageGpu_32f_C4(width,height);

    iu::ImageCpu_32f_C4* initial3dpoints      = new iu::ImageCpu_32f_C4(width,height);

    iu::ImageGpu_32f_C4* d_points_l0_t = new iu::ImageGpu_32f_C4(width,height);
    iu::ImageGpu_32f_C4* d_normals     = new iu::ImageGpu_32f_C4(width,height);

    float4* points3d_data = points3d->data();
    float4* initial3dpoints_data = initial3dpoints->data();

    iu::ImageGpu_8u_C4* d_colour_l0 = new iu::ImageGpu_8u_C4(width,height);

    iu::ImageCpu_8u_C4* h_colour_l0 = new iu::ImageCpu_8u_C4(width,height);

    uchar4* colour_data = h_colour_l0->data();

    uchar4 colour = make_uchar4(255,255,255,1);

//    iu::setValue(colour,d_colour_l0,d_colour_l0->roi());

    CVD::Image<u_int16_t>depthImage(CVD::ImageRef(width,height));
    CVD::Image< CVD::Rgb<CVD::byte> >rgbImage(CVD::ImageRef(width,height));

//    std::string dir_base_name = "/home/ankur/workspace/code/nyu_dataset";

    char fileName[300];

    TooN::Matrix<3> R_basis = TooN::Data(1,0,0,
                                         0,0,1,
                                         0,1,0);

    TooN::Matrix<3>R_basis_inv = R_basis.T();

    std::vector<float> degree_thresholds;
    degree_thresholds.push_back(45.f);
    degree_thresholds.push_back(30.f);
    degree_thresholds.push_back(15.f);

    std::vector<int>max_iterations;
    max_iterations.push_back(10);
    max_iterations.push_back(5);
    max_iterations.push_back(5);


    TooN::Vector<3>yDir = TooN::makeVector(0,1,0);
    TooN::Vector<3>gravity = yDir;

    TooN::Matrix<3>ExpectedRotation = TooN::Data( 0.998622,  0.011428,  0.051225,
                                                 -0.000226,  0.976934, -0.213539,
                                                 -0.052483,  0.213233,  0.975591);

    iu::ImageCpu_32f_C1* h_depth = new iu::ImageCpu_32f_C1(width,height);
    float* h_depth_data = h_depth->data();

    iu::ImageGpu_32f_C1* d_depth = new iu::ImageGpu_32f_C1(width,height);
    iu::ImageGpu_32f_C1* d_depth_bf = new iu::ImageGpu_32f_C1(width,height);

    char rot_mat_fileName[300];

    sprintf(rot_mat_fileName,"camera_rotations_%s.txt",dir_base_name.substr(found+1).c_str());

    std::ofstream rotation_matrices_file(rot_mat_fileName);

    TooN::SO3<>FrameRotation;

    std::cout<<"Going inside pangolin window" << std::endl;


    float2 fl              = make_float2(520.9f,521.0f)/scale;
    float2 pp              = make_float2(325.1f,249.7f)/scale;


    /// viorik_bedroom_scan1
///    FrameRotation = 0.999892 0.013922 -0.00468165
///    -0.013922 0.796715 -0.604194
///    -0.00468165 0.604194 0.796823



    float fx = fl.x, fy = fl.y;
    float u0 = pp.x, v0 = pp.y;

    while( !pangolin::ShouldQuit() )
    {
//        static Var<float>u0("ui.u0",313,0,width);
//        static Var<float>v0("ui.v0",238,0,height);
//        static Var<float>fx("ui.fx",582,0,width);
//        static Var<float>fy("ui.fy",582,0,width);

        static Var<float>rx("ui.rx",0,0,M_PI);
        static Var<float>ry("ui.ry",0,0,M_PI);
        static Var<float>rz("ui.rz",0,0,M_PI);

        static Var<float>tx("ui.tx",0,0,10);
        static Var<float>ty("ui.ty",0,0,10);
        static Var<float>tz("ui.tz",0,0,10);

        static Var<float>degree_threshold_lvl1("ui.degree_thresh_lvl1",45.0f,0.0f,90.0f);
        static Var<float>degree_threshold_lvl2("ui.degree_thresh_lvl2",15.0f,0.0f,90.0f);
        static Var<float>degree_threshold_lvl3("ui.degree_thresh_lvl3",15.0f,0.0f,90.0f);
//        static Var<float>degree_threshold_lvl2("ui.degree_thresh_lvl2",15.0f,0.0f,90.0f);

        static Var<int>itr_lvl1("ui.itr_lvl1",5,0,10);
        static Var<int>itr_lvl2("ui.itr_lvl2",5,0,10);
        static Var<int>itr_lvl3("ui.itr_lvl3",5,0,10);

        static Var<bool>increment("ui.increment",false);

        degree_thresholds.at(0) = (float)degree_threshold_lvl1;
        degree_thresholds.at(1) = (float)degree_threshold_lvl2;
        degree_thresholds.at(2) = (float)degree_threshold_lvl3;

        max_iterations.at(0)    = (int)itr_lvl1;
        max_iterations.at(1)    = (int)itr_lvl2;
        max_iterations.at(2)    = (int)itr_lvl3;

        static Var<int>which_img("ui.which_img",1,1,1449);

        static Var<bool>AlignWithGravity("ui.AlignWithGravity",true);

        static Var<int> max_grid_unit("ui.grid_max",10,0,100);


        float grid_max = (int)max_grid_unit;


//        sprintf(fileName,"%s/raycastDepth_%04d.png",dir_base_name.c_str(),(int)which_img);
        sprintf(fileName,"%s/nyu_depth_img_%04d.png",dir_base_name.c_str(),(int)which_img);

        std::cout<<"Reading the depth file" << fileName << std::endl;

        if ( 1 )
        {
            CVD::img_load(depthImage,fileName);

#pragma omp parallel for
            for(int yy = 0; yy < height; yy++)
            {
                for(int xx = 0; xx < width;  xx++)
                {
                     h_depth_data[yy*width+xx] = depthImage[CVD::ImageRef(xx,yy)]*1.0f/scale_val;

                     float depth= h_depth_data[yy*width+xx];

                     if ( depth > 0 )
                     {

                     TooN::Vector<4> point3D = TooN::makeVector(((xx-(float)u0)/(float)fx)*(depth),
                                                                ((yy-(float)v0)/(float)fy)*(depth),
                                                                 depth,
                                                                 1.0f);

                     initial3dpoints_data[xx+yy*width].x = point3D[0];
                     initial3dpoints_data[xx+yy*width].y = point3D[1];
                     initial3dpoints_data[xx+yy*width].z = point3D[2];
                     initial3dpoints_data[xx+yy*width].w = 1.0f;

                     }
                     else
                     {
                         initial3dpoints_data[xx+yy*width].x = 0.0f;
                         initial3dpoints_data[xx+yy*width].y = 0.0f;
                         initial3dpoints_data[xx+yy*width].z = 0.0f;
                         initial3dpoints_data[xx+yy*width].w = 0.0f;
                     }

                }
            }

            iu::copy(h_depth,d_depth);

            float bf_sigma_s = 2.00;
            float bf_sigma_r = 1.0f;

            aux_math::filterBilateral(d_depth->data(),
                                      d_depth_bf->data(),
                                      d_depth_bf->stride(),
                                      bf_sigma_s,
                                      bf_sigma_r,
                                      make_int2(width,height),
                                      make_int2(5,5),
                                      0.5f);

            iu::copy(d_depth_bf,h_depth);

//#pragma omp parallel for collapse(2)
            for(int yy = 0; yy < height; yy++)
            {
                for(int xx = 0; xx < width; xx++ )
                {
                    float depth = h_depth_data[yy*width+xx];

                    if ( depth > 0 )
                    {
                        TooN::Vector<4> point3D = TooN::makeVector(((xx-(float)u0)/(float)fx)*(depth),
                                                                   ((yy-(float)v0)/(float)fy)*(depth),
                                                                   depth,
                                                                   1.0f);

                        points3d_data[xx+yy*width].x = point3D[0];
                        points3d_data[xx+yy*width].y = point3D[1];
                        points3d_data[xx+yy*width].z = point3D[2];
                        points3d_data[xx+yy*width].w = 1.0f;
                    }
                    else
                    {
                        points3d_data[xx+yy*width].x = 0.f;
                        points3d_data[xx+yy*width].y = 0.f;
                        points3d_data[xx+yy*width].z = 0.f;
                        points3d_data[xx+yy*width].w = 0.f;
                    }

//                    std::cout<<"( " << point3D[0] <<", " << point3D[1]<<", "<<point3D[2]<<", "<<point3D[3]<<") ";
                }
            }


            std::cout<<"Read the 3D points and copied to GPU" << std::endl;

            iu::copy(points3d,d_points_l0_t);


            {
                cuMemTimeUtils::cuTime Timer = cuMemTimeUtils::cuTime();
                Timer.setTimerName("Normal computation");

                aux_math::ComputeNormalsFromVertex(d_normals->data(),
                                                   d_points_l0_t->data(),
                                                   d_points_l0_t->stride(),
                                                   width,
                                                   height);
            }


            std::cout<<"Computed the normals" << std::endl;

            iu::copy(d_normals,points3d);


            yDir = TooN::makeVector(0,1,0);

//            if ( AlignWithGravity )
//            {
//                for(int i = 0; i < degree_thresholds.size() ; i++)
//                {
//                    /// points3d have these normals for now..
//                    getYAlignedWithGravity(points3d->data(),
//                                           degree_thresholds.at(i),
//                                           yDir,
//                                           height,
//                                           width,
//                                           max_iterations.at(i));
//                }

//                FrameRotation = getRMatrix(gravity,yDir);
//            }

            float4 floor_dir = make_float4(0,1,0,0);

//            iu::copy(points3d,d_normals);
            {
                /// TODO: Need to speed it up, just being lazy
                /// in putting the two summations together in the shfl as
                /// well as the standard shfl based code that runs faster.
                cuMemTimeUtils::cuTime Timer = cuMemTimeUtils::cuTime();
                Timer.setTimerName("alignment with gravity");

                GPUReduction::alignYAxisWithGravity(d_normals->data(),
                                                    d_normals->stride(),
                                                    width,
                                                    height,
                                                    floor_dir,
                                                    degree_thresholds,
                                                    max_iterations);
            }

            TooN::Vector<3>final_floor_dir = TooN::makeVector(floor_dir.x,floor_dir.y,floor_dir.z);

            FrameRotation = getRMatrix(gravity,final_floor_dir);

            std::cout<<"FrameRotation = " << FrameRotation << std::endl;

//            exit(1);

            TooN::SO3<>RotOffset = TooN::SO3<>(TooN::makeVector((float)rx,(float)ry,(float)rz));

            TooN::SO3<>FrameRotationWithoutOffset = FrameRotation;

            FrameRotation = RotOffset * FrameRotation ;

#pragma omp parallel for collapse(2)
            for(int yy = 0; yy < height ; yy++)
            {
                for(int xx =0; xx < width; xx++)
                {
                    int ind = yy*width+xx;

                    float4 normal = points3d_data[ind];

                    TooN::SE3<>T_wc;

                    if ( AlignWithGravity )
                        T_wc = TooN::SE3<>(TooN::SO3<>(-FrameRotation.ln()), TooN::makeVector(0,0,0));

                    TooN::Vector<4>n_rotated = T_wc * TooN::makeVector(normal.x,normal.y,normal.z,1.0f);

//                    ofile_nx << n_rotated[0] <<" ";
//                    ofile_ny << n_rotated[1] <<" ";
//                    ofile_nz << n_rotated[2] <<" ";

//                    float4 point3D = initial3dpoints_data[ind];

//                    TooN::Vector<4>point3D_rotated = T_wc * TooN::makeVector(point3D.x,point3D.y,point3D.z,1.0f);

                    n_rotated[0] = (n_rotated[0] + 1)*127.0f;
                    n_rotated[1] = (n_rotated[1] + 1)*127.0f;
                    n_rotated[2] = (n_rotated[2] + 1)*127.0f;

                    colour_data[ind] = make_uchar4((unsigned char)n_rotated[0],
                                                   (unsigned char)n_rotated[1],
                                                   (unsigned char)n_rotated[2],
                                                   1);

//                    ofile_height<< point3D_rotated[1]<<" ";
                }
            }



//            ofile_nx.close();
//            ofile_ny.close();
//            ofile_nz.close();
//            ofile_height.close();

            std::cout<<"FrameRotationWithoutOffset = " << FrameRotationWithoutOffset << std::endl;

            TooN::SO3<>Rot2Write(-FrameRotation.ln());

            std::cout<<"Rotation to write = " << Rot2Write << std::endl;

            rotation_matrices_file << Rot2Write << std::endl;

//            std::cout<<"What we expect = " << ExpectedRotation << std::endl;

            iu::copy(initial3dpoints,d_points_l0_t);

            iu::copy(h_colour_l0,d_colour_l0);

//            prev_img_no = (int)which_img;

//            if ( increment )
            which_img = which_img + 1;
        }

        d_cam.ActivateScissorAndClear(s_cam);
        glEnable(GL_DEPTH_TEST);

        const float sizeL = 3.f;
        const float grid = 2.f;

        glPointSize(sizeL);
        glBegin(GL_LINES);
        glColor3f(.25,.25,.25);
        for(float i=-grid_max; i<=grid_max; i+=grid)
        {
            glVertex3f(-grid_max, i, 0.f);
            glVertex3f(grid_max, i, 0.f);

            glVertex3f(i, -grid_max, 0.f);
            glVertex3f(i, grid_max, 0.f);
        }
        glEnd();


        drawAxes(2.0f);


        glColor3f(1.0,1.0,1.0);

        glPushMatrix();

        if ( AlignWithGravity )
        {

            TooN::Matrix<3>Rot = FrameRotation.get_matrix();
//            Rot = R_basis* Rot * R_basis_inv ;


            TooN::SO3<>so3Rot(Rot);

            Vector<3> aa = -so3Rot.ln();

            double angle = norm(aa);

            if ( angle != 0.0 )
            {
                glRotatef( angle * 180.0 / M_PI, aa[0], aa[1], aa[2]);
            }

        }

        openglrendering::render3dpoints(vertex_array_0,
                                        d_points_l0_t,
                                        colour_array_0,
                                        d_colour_l0,
                                        width,
                                        height);

        glPopMatrix();

        d_panel.Render();
        pangolin::FinishGlutFrame();

    }

    return 0;
}


////#include "../src/utils/align_with_gravity.h"

//int main(void)
//{
////    GPUReduction::alignYAxisWithGravity(640,480,make_float4(1));

//    GPUReduction::reductionCheck(640,480);
//}

