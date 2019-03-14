import sys, time
import rospy
import moveit_commander
import moveit_msgs.msg
import moveit_msgs.srv
import geometry_msgs.msg
from std_msgs.msg import String
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
import tf, pcl
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sensor_stick.pcl_helper import *

# Construct 3D calibration grid across workspace
workspace_limits = np.asarray([[0.45, 0.75], [-0.15,0.15], [0.2, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
calib_grid_step = 0.07
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], 1 + (workspace_limits[0][1] - workspace_limits[0][0])/calib_grid_step)
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], 1 + (workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step)
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], 1 + (workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step)
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]
calib_grid_x.shape = (num_calib_grid_pts,1)
calib_grid_y.shape = (num_calib_grid_pts,1)
calib_grid_z.shape = (num_calib_grid_pts,1)
calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

# waypoint1
calib_grid_joint_values = []
calib_grid_pose_values = []
iter = 1
def pcl_callback(pcl_msg):
    global calib_grid_pts, calib_grid_joint_values, calib_grid_pose_values, iter, group
    cloud = ros_to_pcl(pcl_msg)
    pcl.save(cloud, "/home/bionicdl/kinect_simulation_data/waypoint_%s.pcd"%(iter-1), format="pcd")
    print("Saved waypint_%s"%(iter-1))
    calib_grid_joint_values.append(group.get_current_joint_values())
    calib_grid_pose_values.append(group.get_current_pose())

    if iter>calib_grid_pts.shape[0]:
        print("Finished!")
        return
    tool = PoseStamped()
    tool.header.frame_id = "world"
    tool.pose.position.x = calib_grid_pts[iter,0]
    tool.pose.position.y = calib_grid_pts[iter,1]
    tool.pose.position.z = calib_grid_pts[iter,2]
    q = tf.transformations.quaternion_from_euler(0.3*np.random.uniform(-1,1,1), 0.3*np.random.uniform(-1,1,1), 0.3*np.random.uniform(-1,1,1), axes='sxyz') # 0.35
    tool.pose.orientation.x = q[0]
    tool.pose.orientation.y = q[1]
    tool.pose.orientation.z = q[2]
    tool.pose.orientation.w = q[3]
    group.set_pose_target(tool, end_effector_link='tool0')
    plan = group.plan()
    group.execute(plan)
    time.sleep(1)
    iter = iter + 1
    # raw_input('Press enter to execute: ')

# rosrun point_cloud_io write _topic:=/kinect/kinect/depth/points _folder_path:=/home/bionicdl/kinect_simulation_data/
if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('perception', anonymous=True)

    moveit_commander.roscpp_initialize(sys.argv)
    global group, robot, scene, calib_grid_pts
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("manipulator")
    group.set_planner_id('RRTConnectkConfigDefault')
    group.set_num_planning_attempts(5)
    group.set_planning_time(5)
    group.set_max_velocity_scaling_factor(0.5)

    # waypoint0
    tool = PoseStamped()
    tool.header.frame_id = "world"
    tool.pose.position.x = calib_grid_pts[0,0]
    tool.pose.position.y = calib_grid_pts[0,1]
    tool.pose.position.z = calib_grid_pts[0,2]
    q = tf.transformations.quaternion_from_euler(0.3*np.random.uniform(-1,1,1), 0.3*np.random.uniform(-1,1,1), 0.3*np.random.uniform(-1,1,1), axes='sxyz') # 0.35
    tool.pose.orientation.x = q[0]
    tool.pose.orientation.y = q[1]
    tool.pose.orientation.z = q[2]
    tool.pose.orientation.w = q[3]
    group.set_pose_target(tool, end_effector_link='tool0')
    plan = group.plan()
    group.execute(plan)
    time.sleep(2)

    pcl_sub = rospy.Subscriber("/kinect/kinect/depth/points", PointCloud2, pcl_callback, queue_size=1)

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
    print("Saving waypoint in calibration_waypoints.npz")
    np.savez("/home/bionicdl/kinect_simulation_data/p_robot_mat.npz",calib_grid_joint_values,calib_grid_pose_values)
