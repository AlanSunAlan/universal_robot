# Test interative calibration method deal with noisy data
from open3d import *
import numpy as np
import copy, tf
from calibration import get_calibrate
from sensor_stick.pcl_helper import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#####################################################################################
# calculate calibration data points from point cloud

data_path = "/home/bionicdl/kinect_simulation_data"
# flange segmentation
for i in range(75):
    cloud = pcl.load(data_path+'/waypoint_%s.pcd'%(i))
    passthrough = cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.2
    axis_max = 0.95
    passthrough.set_filter_limits (axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    # TODO: Statistical outlier filter
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    outlier_filter.set_std_dev_mul_thresh(1.0)
    cloud_filtered = outlier_filter.filter()
    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_filtered)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.002)    # Set tolerances for distance threshold
    ec.set_MinClusterSize(500)
    ec.set_MaxClusterSize(100000)   # as well as minimum and maximum cluster size (in points)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    tool_index = []
    # select the tool0 plane with more than 700 points and range < 0.065m
    for j in range(len(cluster_indices)):
        cluster = cluster_indices[j]
        cloud = white_cloud.extract(cluster)
        cloud_array = np.array(cloud)
        seg = cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        max_distance = 0.0005
        seg.set_distance_threshold(max_distance)
        seg.set_MaxIterations(10000)
        seg.set_optimize_coefficients("true")
        seg.set_method_type(0)
        inliers, coefficients = seg.segment()
        if len(inliers)<700:
            continue
        flange = cloud.extract(inliers, negative=False)
        if sum(np.ptp(np.array(flange),axis=0)<0.065)==3:
            print("Saved flange_%s"%i)
            pcl.save(flange, "/home/bionicdl/kinect_simulation_data/tool0_%s.pcd"%i)

# circle fitting
maxD = 0.3/1000
R_FLANGE = 31.0/1000
detR = 0.001 #
p_camera = []
for i in range(75):
    try:
        tool0 = pcl.load("/home/bionicdl/kinect_simulation_data/tool0_%s.pcd"%i)
    except:
        continue
    points = tool0.to_array()
    max_num_inliers = 0
    for k in range(10000):
        idx = np.random.randint(points.shape[0], size=3)
        A, B, C = points[idx,:]
        a = np.linalg.norm(C - B)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(B - A)
        s = (a + b + c) / 2
        R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
        if (R_FLANGE-R)>detR or R_FLANGE-R<0:
            continue
        b1 = a*a * (b*b + c*c - a*a)
        b2 = b*b * (a*a + c*c - b*b)
        b3 = c*c * (a*a + b*b - c*c)
        P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
        P /= b1 + b2 + b3
        num_inliers = 0
        inliers = []
        outliers = []
        for point in points:
            d = np.abs(np.linalg.norm(point-P)-R)
            if d < maxD:
                num_inliers += 1
                inliers.append(point)
            else:
                outliers.append(point)
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            max_A = A
            max_B = B
            max_C = C
            max_R = R
            max_P = P
            max_inliers = np.array(inliers)
            max_outliers = np.array(outliers)
    points_list = []
    for data in max_inliers:
        points_list.append([data[0], data[1], data[2], rgb_to_float([0,255,0])])
    for data in max_outliers:
        points_list.append([data[0], data[1], data[2], rgb_to_float([255,0,0])])
    tool0_c = pcl.PointCloud_PointXYZRGB()
    tool0_c.from_list(points_list)
    pcl.save(tool0_c, "/home/bionicdl/kinect_simulation_data/Tool0_c%s.pcd"%(i))
    p_camera.append(max_P)
    max_R
    max_P

p_robot_pose = np.load("/home/bionicdl/kinect_simulation_data/p_robot_mat.npz")["arr_1"][:]
p_robot_mat = [[p.pose.position.x, p.pose.position.y, p.pose.position.z,] for p in p_robot_pose]
p_robot_mat = np.array(p_robot_mat).transpose()
p_robot_mat = np.delete(p_robot_mat, [17,62,68,74], axis=1)
p_camera_mat = np.array(p_camera).transpose()
np.save("/home/bionicdl/kinect_simulation_data/p_robot_mat.npy",p_robot_mat)
np.save("/home/bionicdl/kinect_simulation_data/p_camera_mat.npy",p_camera_mat)


H_true = tf.transformations.quaternion_matrix([0.707107, -0.707107, 0, 0])
H_true[:3,3] = np.array([0.6, -0.0125, 1])
H_inverse = np.matrix(H_true).getI()
p_camera_true =  H_inverse[:3,:3] * p_robot_mat + np.tile(np.array(H_inverse[:3,3]).reshape([3,1]), [1,71])

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(p_camera_mat[0,:], p_camera_mat[1,:], p_camera_mat[2,:], c='r', marker='x')
ax1.scatter(p_camera_true[0,:], p_camera_true[1,:], p_camera_true[2,:], c='g', marker='o',alpha=0.5,s=30)
# ax1.set_xlim([-0.15, 0.15])
# ax1.set_ylim([-0.15, 0.15])
# ax1.set_zlim([0.6, 0.8])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.show()
#####################################################################################
# test
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def evaluate_calibration(s,t,H):
    sTt = copy.deepcopy(s)
    sTt.transform(H)
    HI = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    reg_p2p = registration_icp(sTt, t, 0.0003, HI, TransformationEstimationPointToPoint(),ICPConvergenceCriteria(max_iteration = 2000))
    R = reg_p2p.transformation[:3,:3]
    T = reg_p2p.transformation[:3,3]
    al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')
    # print("xyz= [%2.2f, %2.2f, %2.2f]"%(T[0]*1000,T[1]*1000,T[2]*1000))
    # print("rpy= [%2.5f, %2.5f, %2.5f]"%(al,be,ga))
    return T*1000, np.array([al, be, ga])

# read calibration points
p_robot_mat = np.load("/home/bionicdl/kinect_simulation_data/p_robot_mat.npy")
p_camera_mat = np.load("/home/bionicdl/kinect_simulation_data/p_camera_mat.npy")

calibrate = get_calibrate(4)
#######################################################
# method 1: calibration using all data points
H = calibrate(p_robot_mat, p_camera_mat)
d = H.getI() * H_true
xyz = np.array(d[:3,3]).transpose()[0] *1000
rpy =  tf.transformations.euler_from_matrix(np.array(d[:3,:3]), 'sxyz')
print("Error xyz:[%2.2f, %2.2f, %2.2f] mm rpy:[%2.5f, %2.5f, %2.5f]"%(xyz[0],xyz[1],xyz[2], rpy[0],rpy[1],rpy[2]))
np.save('/home/bionicdl/kinect_simulation_data/H.npy',H)
#########################################################
# method 4: interative adding new points

idx = np.random.choice(p_robot_mat.shape[1], p_robot_mat.shape[1],0)
p_robot_mat_i = p_robot_mat[:,idx[:5]]
p_camera_mat_i = p_camera_mat[:,idx[:5]]
error_xyz = 1000
for i in range(5,len(idx)):
    for j in range(5):
        p_robot_mat_j = p_robot_mat_i
        p_camera_mat_j = p_camera_mat_i
        p_robot_mat_j = np.delete(p_robot_mat_j, j,1)
        p_camera_mat_j = np.delete(p_camera_mat_j, j,1)
        H_j = calibrate(p_robot_mat_j, p_camera_mat_j)
        T_j = H_j[:3,3]
        d = H_j.getI() * H_true
        xyz_j = np.array(d[:3,3]).transpose()[0] *1000
        rpy_j =  tf.transformations.euler_from_matrix(np.array(d[:3,:3]), 'sxyz')
        if 0<(np.sum(xyz_j**2))**(0.5)<error_xyz:
            error_xyz = (np.sum(xyz_j**2))**(0.5)
            H_i = H_j
            xyz_i = xyz_j
            rpy_i =rpy_j
            p_robot_mat_s = p_robot_mat_j
            p_camera_mat_s = p_camera_mat_j
    print("error_xyz:%2.2f xyz:[%2.2f, %2.2f, %2.2f] mm rpy_i:[%2.5f, %2.5f, %2.5f]"%(error_xyz, xyz_i[0],xyz_i[1],xyz_i[2], rpy_i[0],rpy_i[1],rpy_i[2]))
    # if error_xyz < 0.5:
    #     print("Terminated: calibration error is within 0.5 mm!")
    #     break
    p_robot_mat_i = np.concatenate((p_robot_mat_s, p_robot_mat[:,idx[i]].reshape([3,1])), axis=1)
    p_camera_mat_i = np.concatenate((p_camera_mat_s, p_camera_mat[:,idx[i]].reshape([3,1])), axis=1)

np.save('/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/H_i.npy',H_i)
draw_registration_result(s, t, H_i)
