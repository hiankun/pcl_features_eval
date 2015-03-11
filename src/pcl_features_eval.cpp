#include <time.h>
#include <string>
#include <pcl/common/time.h> //fps calculations
#include <iostream>
#include <vector>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/usc.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/spin_image.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/esf.h>
#include <pcl/features/gfpfh.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>

const double search_radius = 0.03;

void print_usage(const char* prog_name) {
    std::cout << "\n\nUsage: " << prog_name << " point_cloud_file.pcd [options]\n\n"
        << "options:\n"
        << "------------------------------------------------------------\n"
        << "-m <0-13>   method (see the list below)\n"
        << "-d <float>  voxel grid size\n"
        << "-o          with original point cloud withoud downsampling\n"
        << "            [Warning] this could take a long time...\n"
        << "-v          launch viewer\n"
        << "-p          print point cloud in terminal\n"
        << "-h          this help\n"
        << "\n"
        << "------------------------------------------------------------\n"
        << "method list:\n"
        << " 0:  PFH\n"
        << " 1:  FPFH\n"
        << " 2:  FPFH_OMP\n"
        << " 3:  3DSC\n"
        << " 4:  USC\n"
        << " 5:  SHOT\n"
        << " 6:  SHOT_OMP\n"
        << " 7:  SI\n"
        << " 8:  NARF\n"
        << " 9:  VFH\n"
        << "10:  CVFH\n"
        << "11:  OUR_CVFH\n"
        << "12:  ESF\n"
        << "13:  GFPFH\n"
        << "------------------------------------------------------------\n";
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> cloudViewer (
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        std::string str = "") {

    std::string viewer_name;
    if (str.size() > 0)
        viewer_name = "PCL Viewer (" + str +")";
    else
        viewer_name = "PCL Viewer";

    boost::shared_ptr<pcl::visualization::PCLVisualizer> \
        viewer(new pcl::visualization::PCLVisualizer(viewer_name.c_str()));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
    viewer->addCoordinateSystem(0.3);

    return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> cloudViewer (
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
        pcl::PointCloud<pcl::Normal>::ConstPtr normals,
        std::string str = "") {

    std::string viewer_name;
    if (str.size() > 0)
        viewer_name = "PCL Viewer (" + str +")";
    else
        viewer_name = "PCL Viewer";

    boost::shared_ptr<pcl::visualization::PCLVisualizer> \
        viewer(new pcl::visualization::PCLVisualizer(viewer_name.c_str()));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, normals);
    viewer->addCoordinateSystem(0.3);

    return (viewer);
}

void get_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setRadiusSearch(search_radius);//0.03
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);
    //std::cout<< "get_normals: size: " << normals->points.size() << std::endl;
}

void do_downsampling(pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &out_cloud, float leaf_size = 0.02f){

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(in_cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*out_cloud);
}

void do_PFH(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;

    pfh.setInputCloud(cloud);
    pfh.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    pfh.setSearchMethod(kdtree);
    pfh.setRadiusSearch(search_radius);

    pcl::PointCloud<pcl::PFHSignature125>::Ptr \
        descriptors(new pcl::PointCloud<pcl::PFHSignature125>());

    std::cout  << std::setw(40) << std::left << "calculating PFH features...";
    double last = pcl::getTime();
    pfh.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";

    //-- plot the descriptor
    //pcl::visualization::PCLPlotter plotter;
    //plotter.addFeatureHistogram(*descriptors, 125);
    //plotter.setTitle("PFH");
    //plotter.setShowLegend(false);
    //plotter.plot();

}

void do_FPFH(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;

    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod(kdtree);
    fpfh.setRadiusSearch(search_radius);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr \
        descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());

    std::cout  << std::setw(40) << std::left << "calculating FPFH features...";
    double last = pcl::getTime();
    fpfh.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";

    //-- plot the descriptor
    //pcl::visualization::PCLPlotter plotter;
    //plotter.addFeatureHistogram(*descriptors, 33);
    //plotter.setTitle("FPFH");
    //plotter.setShowLegend(false);
    //plotter.plot();

}

void do_FPFH_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_omp;

    fpfh_omp.setInputCloud(cloud);
    fpfh_omp.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh_omp.setSearchMethod(kdtree);
    fpfh_omp.setRadiusSearch(search_radius);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr \
        descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());

    std::cout  << std::setw(40) << std::left << "calculating FPFH_OMP features...";
    double last = pcl::getTime();
    fpfh_omp.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";

}

void do_3DSC(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::ShapeContext3DEstimation<pcl::PointXYZ, pcl::Normal, pcl::ShapeContext1980> sc3d;

    sc3d.setInputCloud(cloud);
    sc3d.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    sc3d.setSearchMethod(kdtree);
    sc3d.setRadiusSearch(search_radius);
    sc3d.setMinimalRadius(search_radius/10.0);
    sc3d.setPointDensityRadius(search_radius/5.0);

    pcl::PointCloud<pcl::ShapeContext1980>::Ptr \
        descriptors(new pcl::PointCloud<pcl::ShapeContext1980>());

    std::cout  << std::setw(40) << std::left << "calculating 3DSC features...";
    double last = pcl::getTime();
    sc3d.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

void do_USC(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {

    pcl::UniqueShapeContext<pcl::PointXYZ, pcl::ShapeContext1980, pcl::ReferenceFrame> usc;

    usc.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    usc.setSearchMethod(kdtree);
    usc.setRadiusSearch(search_radius);
    usc.setMinimalRadius(search_radius/10.0);
    usc.setPointDensityRadius(search_radius/5.0);
    usc.setLocalRadius(search_radius);

    pcl::PointCloud<pcl::ShapeContext1980>::Ptr \
        descriptors(new pcl::PointCloud<pcl::ShapeContext1980>());

    std::cout  << std::setw(40) << std::left << "calculating USC features...";
    double last = pcl::getTime();
    usc.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

void do_SHOT(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;

    shot.setInputCloud(cloud);
    shot.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    shot.setSearchMethod(kdtree);
    //-- NOTE: don't set the radius too small or there will be error of
    // ``[pcl::SHOTEstimation::computeFeature]
    // The local reference frame is not valid! Aborting description of point with index xxx''
    shot.setRadiusSearch(search_radius*2);

    pcl::PointCloud<pcl::SHOT352>::Ptr \
        descriptors(new pcl::PointCloud<pcl::SHOT352>());

    std::cout  << std::setw(40) << std::left << "calculating SHOT features...";
    double last = pcl::getTime();
    shot.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

void do_SHOT_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_omp;

    shot_omp.setInputCloud(cloud);
    shot_omp.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    shot_omp.setSearchMethod(kdtree);
    //-- NOTE: don't set the radius too small or there will be error of
    // ``[pcl::SHOTEstimation::computeFeature]
    // The local reference frame is not valid! Aborting description of point with index xxx''
    shot_omp.setRadiusSearch(search_radius*2);

    pcl::PointCloud<pcl::SHOT352>::Ptr \
        descriptors(new pcl::PointCloud<pcl::SHOT352>());

    std::cout  << std::setw(40) << std::left << "calculating SHOT_OMP features...";
    double last = pcl::getTime();
    shot_omp.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

typedef pcl::Histogram<153> SpinImage;
void do_SI(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, SpinImage> si;

    si.setInputCloud(cloud);
    si.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    si.setSearchMethod(kdtree);
    si.setRadiusSearch(search_radius);
    si.setImageWidth(8);

    pcl::PointCloud<SpinImage>::Ptr \
        descriptors(new pcl::PointCloud<SpinImage>());

    std::cout << std::setw(40) << std::left << "calculating SpinImage features...";
    double last = pcl::getTime();
    si.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

#if 0
void get_range_image(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    Eigen::Affine3f scene_sensor_pose(Eigen::Affine3f::Identity());

    scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(
                cloud->sensor_origin_[0],
                cloud->sensor_origin_[1],
                cloud->sensor_origin_[2])) *
        Eigen::Affine3f(cloud->sensor_orientation_);

    //-- create range image from the cloud
    float noise_level = 0.0f;
    float min_range = 0.0f;
    int border_size = 1;
    boost::shared_ptr<pcl::RangeImage> range_image(new pcl::RangeImage);
    //pcl::RangeImage& range_image = *range_image_ptr;


    // --------------------
    // -----Parameters-----
    // --------------------
    float angular_resolution_x = 0.5f,
          angular_resolution_y = angular_resolution_x;
    angular_resolution_x = pcl::deg2rad (angular_resolution_x);
    angular_resolution_y = pcl::deg2rad (angular_resolution_y);
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    bool live_update = false;

    range_image->createFromPointCloud(*cloud, angular_resolution_x, angular_resolution_y,
            pcl::deg2rad(360.0f), pcl::deg2rad(180.0f),
            scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);


    //-- visualization
    pcl::visualization::RangeImageVisualizer viewer ("Range image");
    viewer.showRangeImage (*range_image);
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
        pcl_sleep(0.1);
    }
}
#endif

void get_range_image(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      pcl::RangeImagePlanar& range_image) {
    float image_size_x = 640; //cloud->width;
    float image_size_y = 480; //cloud->height;

    float center_x = image_size_x * 0.5f;
    float center_y = image_size_y * 0.5f;
    float focal_length_x = 200.0f; //todo
    float focal_length_y = focal_length_x;

    Eigen::Affine3f scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(
                cloud->sensor_origin_[0],
                cloud->sensor_origin_[1],
                cloud->sensor_origin_[2])) *
        Eigen::Affine3f(cloud->sensor_orientation_);

    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    float noise_level = 0.0f;
    float min_range = 0.0f;

    //pcl::RangeImagePlanar range_image;
    range_image.createFromPointCloudWithFixedSize(
            *cloud, image_size_x, image_size_y,
            center_x, center_y, focal_length_x, focal_length_y,
            scene_sensor_pose, coordinate_frame,
            noise_level, min_range);
#if 0
    //-- visualization
    pcl::visualization::RangeImageVisualizer viewer("planar range image");
    viewer.showRangeImage(range_image);
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
        pcl_sleep(0.1);
    }
#endif
}

void do_NARF(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    //-- get range image
    pcl::RangeImagePlanar range_image;
    get_range_image(cloud, range_image);

    //--find borders
    pcl::RangeImageBorderExtractor border_extractor(&range_image);
    pcl::PointCloud<pcl::BorderDescription>::Ptr borders(\
            new pcl::PointCloud<pcl::BorderDescription>);
    border_extractor.compute(*borders);

    pcl::visualization::RangeImageVisualizer* viewer = NULL;
    viewer = pcl::visualization::RangeImageVisualizer::getRangeImageBordersWidget(range_image,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            false, *borders, "Borders");
    while (!viewer->wasStopped()) {
        viewer->spinOnce();
        pcl_sleep(0.1);
    }

}

//-- Global features
void do_VFH(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;

    vfh.setInputCloud(cloud);
    vfh.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    vfh.setSearchMethod(kdtree);
    //--optional settings
    vfh.setNormalizeBins(true);
    vfh.setNormalizeDistance(false);

    pcl::PointCloud<pcl::VFHSignature308>::Ptr \
        descriptors(new pcl::PointCloud<pcl::VFHSignature308>());

    std::cout << std::setw(40) << std::left << "calculating VFH features...";
    double last = pcl::getTime();
    vfh.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

void do_CVFH(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::CVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> cvfh;

    cvfh.setInputCloud(cloud);
    cvfh.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    cvfh.setSearchMethod(kdtree);
    cvfh.setEPSAngleThreshold(5.0/180.0*M_PI);
    cvfh.setCurvatureThreshold(1.0);
    //--optional settings
    cvfh.setNormalizeBins(false);

    pcl::PointCloud<pcl::VFHSignature308>::Ptr \
        descriptors(new pcl::PointCloud<pcl::VFHSignature308>());

    std::cout << std::setw(40) << std::left << "calculating CVFH features...";
    double last = pcl::getTime();
    cvfh.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

void do_OUR_CVFH(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        pcl::PointCloud<pcl::Normal>::Ptr &normals) {

    pcl::OURCVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> our_cvfh;

    our_cvfh.setInputCloud(cloud);
    our_cvfh.setInputNormals(normals);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    our_cvfh.setSearchMethod(kdtree);
    our_cvfh.setEPSAngleThreshold(5.0/180.0*M_PI);
    our_cvfh.setCurvatureThreshold(1.0);
    our_cvfh.setNormalizeBins(false);
    our_cvfh.setAxisRatio(0.8);

    pcl::PointCloud<pcl::VFHSignature308>::Ptr \
        descriptors(new pcl::PointCloud<pcl::VFHSignature308>());

    std::cout << std::setw(40) << std::left << "calculating OUR_CVFH features...";
    double last = pcl::getTime();
    our_cvfh.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

void do_ESF(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {

    pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;

    esf.setInputCloud(cloud);

    pcl::PointCloud<pcl::ESFSignature640>::Ptr \
        descriptors(new pcl::PointCloud<pcl::ESFSignature640>());

    std::cout << std::setw(40) << std::left << "calculating ESF features...";
    double last = pcl::getTime();
    esf.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}

void do_GFPFH(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudxyz) {

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::copyPointCloud(*cloudxyz, *cloud);

    //-- classification
    for (size_t i = 0; i < cloud->points.size(); ++i)
        cloud->points[i].label = 1 + i % 4;

    pcl::GFPFHEstimation<pcl::PointXYZL, pcl::PointXYZL, pcl::GFPFHSignature16> gfpfh;

    gfpfh.setInputCloud(cloud);
    gfpfh.setInputLabels(cloud);
    gfpfh.setOctreeLeafSize(0.01);
    gfpfh.setNumberOfClasses(4);

    pcl::PointCloud<pcl::GFPFHSignature16>::Ptr \
        descriptors(new pcl::PointCloud<pcl::GFPFHSignature16>());

    std::cout  << std::setw(40) << std::left << "calculating GFPFH features...";
    double last = pcl::getTime();
    gfpfh.compute(*descriptors);
    double now = pcl::getTime();
    std::cout << "processing time: " << std::fixed << std::setw(10) << std::setprecision(3) << std::right << (now-last)*1000. << " ms\n";
}


int main(int argc, char** argv) {
    if (pcl::console::find_argument(argc, argv, "-h") >= 0
            || argc < 2 ) {
        print_usage(argv[0]);
        return -1;
    }

    std::string pcd_file = argv[1];
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal> ());
    pcl::PointCloud<pcl::Normal>::Ptr normals_(new pcl::PointCloud<pcl::Normal> ());

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_file, *cloud) == -1) {
        std::string pcl_err_str;
        pcl_err_str = "Couldn't read file " + pcd_file;
        PCL_ERROR(pcl_err_str.c_str());
        return -1;
    }

    std::cout << "Loaded " << cloud->width * cloud->height
        << " data points from " << argv[1] << " ..." << std::endl;

    if (pcl::console::find_argument(argc, argv, "-p") >= 0) {
        for (size_t i = 0; i < cloud->points.size(); ++i)
            std::cout << " " << cloud->points[i].x
                << " " << cloud->points[i].y
                << " " << cloud->points[i].z
                << std::endl;
    }

    std::cout << "point size: " << cloud->points.size() << std::endl;

    bool downsample = true;
    if (pcl::console::find_argument(argc, argv, "-o") >= 0)
        downsample = false;

    if (downsample){
        float leaf_size;
        if (pcl::console::parse(argc, argv, "-d", leaf_size) >= 0)
            do_downsampling(cloud, cloud_, leaf_size);
        else
            do_downsampling(cloud, cloud_);

        get_normals(cloud_, normals_);
        std::cout << "reduced point size: " << cloud_->points.size() << std::endl;
    } else {
        std::cout << "Calculating normals with original point cloud...\n";
        get_normals(cloud, normals); //--! this could take long time
        *cloud_ = *cloud;
        *normals_ = *normals;
    }


    int method = 0;
    if (pcl::console::parse(argc, argv, "-m", method) < 0)
        method = -1; //run all methods

    switch (method) {
        case 0:
            do_PFH(cloud_, normals_);
            break;
        case 1:
            do_FPFH(cloud_, normals_);
            break;
        case 2:
            do_FPFH_OMP(cloud_, normals_);
            break;
        case 3:
            do_3DSC(cloud_, normals_); //failed in some pcd files
            break;
        case 4:
            do_USC(cloud_);
            break;
        case 5:
            do_SHOT(cloud_, normals_);
            break;
        case 6:
            do_SHOT_OMP(cloud_, normals_);
            break;
        case 7:
            do_SI(cloud_, normals_);
            break;
        case 8:
            do_NARF(cloud_);
            break;
        case 9:
            do_VFH(cloud_, normals_);
            break;
        case 10:
            do_CVFH(cloud_, normals_);
            break;
        case 11:
            do_OUR_CVFH(cloud_, normals_);
            break;
        case 12:
            do_ESF(cloud_);
            break;
        case 13:
            do_GFPFH(cloud_);
            break;
        default: // run all methods
            do_PFH(cloud_, normals_);
            do_FPFH(cloud_, normals_);
            do_FPFH_OMP(cloud_, normals_);
            //do_3DSC(cloud_, normals_); //failed in some pcd files
            do_USC(cloud_);
            //do_SHOT(cloud_, normals_);
            //do_SHOT_OMP(cloud_, normals_);
            do_SI(cloud_, normals_);
            do_NARF(cloud_);
            do_VFH(cloud_, normals_);
            do_CVFH(cloud_, normals_);
            do_OUR_CVFH(cloud_, normals_);
            do_ESF(cloud_);
            do_GFPFH(cloud_);
            break;
    }


    //-- viewer

    if (pcl::console::find_argument(argc, argv, "-v") >= 0) {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;

        viewer = cloudViewer(cloud);
        viewer_ = cloudViewer(cloud_, normals_, "downsampled");
        while (!viewer->wasStopped() && !viewer_->wasStopped()) {
            viewer->spinOnce(100);
            viewer_->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(1000));
        }
    }
    return 0;
}
