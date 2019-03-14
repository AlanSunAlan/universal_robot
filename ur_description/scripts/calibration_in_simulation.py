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
for i in range(74):
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

p_robot_pose = np.load("/home/bionicdl/kinect_simulation_data/p_robot_mat.npz")["arr_1"][:74]
p_robot_mat = [[p.pose.position.x, p.pose.position.y, p.pose.position.z,] for p in p_robot_pose]
p_robot_mat = np.array(p_robot_mat).transpose()
p_camera_mat = np.array(p_camera).transpose()
np.save("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/p_robot_mat.npy",p_robot_mat)
np.save("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/p_camera_mat.npy",p_camera_mat)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p_camera_mat[0,:10], p_camera_mat[1,:10], p_camera_mat[2,:10], c='r', marker='x')
ax.set_xlim()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
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

# read point cloud for hand-eye calibration validation
target = read_point_cloud("/home/bionicdl/photoneo_data/20181217/aubo-i5-EndFlange_cropped_m.pcd")
source = read_point_cloud("/home/bionicdl/photoneo_data/20181217/tool0_5.pcd")
H_offset = np.matrix([[-1,0,0,0],[0,1,0,0],[0,0,-1,-0.006],[0,0,0,1]])
H_base_tool = tf.transformations.quaternion_matrix([-0.08580920098798522, -0.3893105864494028, 0.9148593368686363, 0.06408152657751885])
H_base_tool[:3,3] = np.array([0.27704067625331485, -0.573055166920657, 0.26205388882758757])
s = copy.deepcopy(source)
t = copy.deepcopy(target)
t.transform(H_offset)
t.transform(H_base_tool)

# read calibration points
p_robot_mat = np.load("/home/bionicdl/kinect_simulation_data/p_robot_mat.npy")
p_camera_mat = np.load("/home/bionicdl/kinect_simulation_data/p_camera_mat.npy")

calibrate = get_calibrate(4)
#######################################################
# method 1: calibration using all data points
H = calibrate(p_robot_mat, p_camera_mat)
R = H[:3,:3]
T = H[:3,3]
al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')
print("xyz= %s"%(T.getT()))
print("rpy= %s %s %s"%(al,be,ga))
np.save('/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/H.npy',H)
error_matrix = p_robot_mat - np.matmul(H[:3,:3],p_camera_mat) - np.tile(H[:3,3],[1,p_camera_mat.shape[1]])
error = np.mean( (np.sum((np.asarray(error_matrix))**2,axis=0))**(0.5) )
draw_registration_result(s, t, H)

#########################################################
# method 4: interative adding new points
idx = np.random.choice(p_robot_mat.shape[1], 63,0)
p_robot_mat_i = p_robot_mat[:,idx[:5]]
p_camera_mat_i = p_camera_mat[:,idx[:5]]
error_xyz = 1000
for i in range(5,len(idx)):
    for j in range(4):
        p_robot_mat_j = p_robot_mat_i
        p_camera_mat_j = p_camera_mat_i
        p_robot_mat_j = np.delete(p_robot_mat_j, j,1)
        p_camera_mat_j = np.delete(p_camera_mat_j, j,1)
        H_j = calibrate(p_robot_mat_j, p_camera_mat_j)
        xyz_j, rpy_j = evaluate_calibration(s,t,H_j)
        if 0<(np.sum(xyz_j**2))**(0.5)<error_xyz:
            error_xyz = (np.sum(xyz_j**2))**(0.5)
            H_i = H_j
            xyz_i = xyz_j
            rpy_i =rpy_j
            p_robot_mat_s = p_robot_mat_j
            p_camera_mat_s = p_camera_mat_j
    error_matrix = p_robot_mat_s - np.matmul(H_i[:3,:3],p_camera_mat_s) - np.tile(H_i[:3,3],[1, p_camera_mat_s.shape[1]])
    error = (np.sum((np.asarray(error_matrix))**2,axis=0))**(0.5)
    print("Iteration:{}  Error:{} mm xyz:{} mm rpy_i:{}".format(i, np.mean(error*1000), xyz_i, rpy_i))
    if error_xyz < 0.5:
        print("Terminated: calibration error is within 0.5 mm!")
        break
    p_robot_mat_i = np.concatenate((p_robot_mat_s, p_robot_mat[:,idx[i]].reshape([3,1])), axis=1)
    p_camera_mat_i = np.concatenate((p_camera_mat_s, p_camera_mat[:,idx[i]].reshape([3,1])), axis=1)

np.save('/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/H_i.npy',H_i)
draw_registration_result(s, t, H_i)
