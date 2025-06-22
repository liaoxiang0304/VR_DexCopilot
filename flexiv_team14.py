from .players import TeleopPlayer
import numpy as np

import gym
import time
# import pytorch3d.transforms as p3dt
from scipy.spatial.transform import Rotation
import numpy as np
from datetime import datetime
import pickle
import cv2
import h5py
import pickle as pkl
import sys, os, glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib_py'))

from ik import IKController
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from scipy.spatial.transform import Rotation as R

import torch 
import threading
import flexivrdk

from typing import List
import yaml

# from dex_retargeting.constants import RobotName,RetargetingType,HandType,get_default_config_path
# from dex_retargeting.retargeting_config import get_retargeting_config,RetargetingConfig
# init Leap Motion Detector

def rescale_actions(low, high, action):
	d = (high - low) / 2.0
	m = (high + low) / 2.0
	scaled_action =  action * d + m
	return scaled_action

def scale(x, lower, upper):
	return (0.5 * (x + 1.0) * (upper - lower) + lower)


def unscale(x, lower, upper):
	return (2.0 * x - upper - lower) / (upper - lower)

# =============== Hyper params BEGIN ======================
scaling_factor = 4                                  # scale leap hand motion -> ee pos motion
sigma_scaling_factor = 2.5
leap_lower_bound = np.array([-0.2, 0, -0.1])
leap_upper_bound = np.array([0.2, 0.4, 0.35])
vr_lower_bound = np.array([-1, -1, 0])
vr_upper_bound = np.array([1,1, 2])
isaacgym_lower_bound = torch.tensor([-2, -2, -2])
isaacgym_upper_bound = torch.tensor([2, 2, 2])

tcp_lower_bound = torch.tensor([0.3, -0.24, 0.03])  # used for real world robot
tcp_upper_bound = torch.tensor([0.87,  0.32, 0.60])

home_joint = torch.tensor([0, -40, 0, 90, 0, 40, 0])# degree
home_joint = home_joint / 180. * 3.1415             # degree -> rad

frequency = 60 # Hz
robot_ip = "192.168.2.100"
local_ip = "192.168.2.103"
log = flexivrdk.Log()
mode = flexivrdk.Mode
test_on_real_robot = True  
test_on_real_hand = False
save_realsense = True
robot = None
gripper = None
leap_hand = None
recording_data: bool = True
cam_agent_view = []
cam_wrist_view = []
cam_bird_view = []
cam_left_view = []
cam_right_view = []

keep_hand_pos=True


# =============== Hyper params END ======================

def set_leap_hand_pos(leaphand,action,nail_handles_):
	# nail_handles_=np.asarray(nail_handles_).reshape(-1)
	set_pose=np.zeros(16)
	for i, key in enumerate([12,22,27]):
		tip = key
		tip = tip-1
		finger_tip = tip-1
		dip = tip - 2
		pip = tip - 3

		set_pose[i*4]   = action[dip]
		set_pose[i*4+1] = action[pip]
		set_pose[i*4+2] = action[finger_tip]
		set_pose[i*4+3] = action[tip]

	tip = 16
	finger_tip = tip - 1
	dip = tip - 2
	pip = tip - 3
	set_pose[12] = action[pip]
	set_pose[13] = action[dip]
	set_pose[14] = action[finger_tip]*-1
	set_pose[15] = action[tip]*-1

	print(set_pose)
	leaphand.set_allegro(set_pose)
	time.sleep(0.03)

def handle_viewer_events(task, viewer, /, save_folder="/home/rhos/Desktop/Gripper/Pick"):
	env = task.env
	gym = env.gym
	for evt in gym.query_viewer_action_events(viewer):
		# Reset the DOF states and actor root state to their initial states
		global recording_data, gripper
		
		if (evt.action == "reset") and evt.value > 0:
			print("Reset Env: Home()")
			reset_home(task)    
			recording_data               = False
			task.initialized             = False
			task.init_ee_state           = None
			task.prev_motion_joint       = None
			task.obs_dict                = []
		
		elif (evt.action == "flexiv_data_collection") and evt.value > 0:
			if recording_data:
				recording_data = False
				print("End recording data")
			else:
				recording_data = True
				
				try:
					gripper.move(0.09, 0.1, 20)
				except Exception as e:
					log.error(str(e))
				reset_home(task)
				task.initialized             = False
				task.prev_motion_joint       = None
				task.obs_dict                = []
				cam_agent_view.clear()
				cam_wrist_view.clear()
				cam_bird_view.clear()
				cam_left_view.clear()
				cam_right_view.clear()
				print("Start recording data")
		
		elif (evt.action == "space_shoot") and evt.value > 0:
			if save_realsense:
				dirname = f"{save_folder}/Flexiv_{task.timestamp}"
				os.makedirs(dirname, exist_ok=True)
				
				cnt = len(glob.glob(os.path.join(dirname, "*/")))
				dirname = os.path.join(dirname, f"{cnt:03}")
				os.makedirs(dirname, exist_ok=True)
				for f in os.listdir(dirname):
					os.remove(os.path.join(dirname, f))
					
				filename = os.path.join(dirname, "data.pkl")
				with open(filename, 'wb') as f:
					pkl.dump(task.obs_dict, f)
				
				# save agent view
				if len(cam_agent_view)>0:
					with h5py.File(os.path.join(dirname, "agent_img.hdf5"), 'w') as hf:
						images = np.array([view["image"] for view in cam_agent_view])
						depths = np.array([view["depth"] for view in cam_agent_view])
						hf.create_dataset("images", data=images)
						hf.create_dataset("depths", data=depths)

				# save wrist view
				if len(cam_wrist_view)>0:
					with h5py.File(os.path.join(dirname, "wrist_img.hdf5"), 'w') as hf:
						images = np.array([view["image"] for view in cam_wrist_view])
						depths = np.array([view["depth"] for view in cam_wrist_view])
						hf.create_dataset("images", data=images)
						hf.create_dataset("depths", data=depths)
				
				# save bird view
				if len(cam_bird_view)>0:
					with h5py.File(os.path.join(dirname, "bird_img.hdf5"), 'w') as hf:
						images = np.array([view["image"] for view in cam_bird_view])
						depths = np.array([view["depth"] for view in cam_bird_view])
						hf.create_dataset("images", data=images)
						hf.create_dataset("depths", data=depths)

				# save left view
				if len(cam_left_view)>0:
					with h5py.File(os.path.join(dirname, "left_img.hdf5"), 'w') as hf:
						images = np.array([view["image"] for view in cam_left_view])
						depths = np.array([view["depth"] for view in cam_left_view])
						hf.create_dataset("images", data=images)
						hf.create_dataset("depths", data=depths)
					
					print("left saved!")

				# save right view
				if len(cam_right_view)>0:
					with h5py.File(os.path.join(dirname, "right_img.hdf5"), 'w') as hf:
						images = np.array([view["image"] for view in cam_right_view])
						depths = np.array([view["depth"] for view in cam_right_view])
						hf.create_dataset("images", data=images)
						hf.create_dataset("depths", data=depths)
					
					print("right saved!")
				print("Realsense Camera hdf5 Saved!")
			
			# filename = "obs_0.pkl"
			# motionfile = "faliure_motion_0.pkl"
			# counter = 1 
			# os.makedirs(f"../../data/FlexivLeap_{task.timestamp}", exist_ok=True)
			# # os.makedirs(f"../../data/Motion_{task.timestamp}", exist_ok=True)
			# while os.path.exists(f"../../data/FlexivLeap_{task.timestamp}/{filename}"):
			#     filename = f"obs_{counter}.pkl"
			#     motionfile = f"faliure_motion_{counter}.pkl"
			#     counter += 1
			# with open(f"../../data/FlexivLeap_{task.timestamp}/{filename}", 'wb') as f:
			#     pkl.dump(task.obs_dict, f)
			# # with open(f"../../data/Motion_{task.timestamp}/{motionfile}", 'wb') as f:
			# #     pkl.dump(task.motion_dict, f)
			# print("Saved!")

def draw_bounding_box_global(task, tcp_upper_bound, tcp_lower_bound):
	"""xuyue: why not share same draw_bounding_box here?"""
	task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
	task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
	task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

	task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
	task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
	task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

	task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
	task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
	task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

	task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
	task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
	task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))


def reset_home(task):
	env = task.env
	gym = env.gym
	env.reset_idx(torch.tensor([0]),torch.tensor([0]))
	env.cur_targets = torch.zeros_like(env.cur_targets)
	
	if test_on_real_robot:
		move_home(robot)
		reset_joints = get_robot_joints(robot)
		reset_joints = to_torch(reset_joints)
		robot.setMode(mode.NRT_JOINT_POSITION)
		# global gripper
		# gripper.move(0.09, 0.1, 10)
	else:
		reset_joints = home_joint

	if test_on_real_hand:
		set_leap_hand_pos(leap_hand,np.zeros(28),env.nail_handles)

		
	env.cur_targets[:, :reset_joints.shape[-1]] = reset_joints
	# env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+reset_joints.shape[-1]] = reset_joints
	
	gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.cur_targets))
	start_time = time.time()    
	while True:
		task.step()
		if time.time() - start_time > 2.5:
			break
					
def reset_joints(task: TeleopPlayer, joints: torch.tensor):
	"""reset joints to the given position

	Args:
		task (TeleopPlayer): _description_
		joints (torch.tensor): _description_
	"""
	env = task.env
	gym = env.gym
	env.reset_idx(torch.tensor([0]),torch.tensor([0]))
	env.cur_targets = torch.zeros_like(env.cur_targets)
	env.cur_targets[:, :joints.shape[-1]] = joints
	# env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+joints.shape[-1]] = joints
	
	gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.cur_targets))
	task.step()
	task.initialized             = False
	task.prev_motion_joint       = None
	task.obs_dict                = []



def get_robot_force():
	"""
	Print robot states data
	"""
	robot_states = flexivrdk.RobotStates()
	robot.getRobotStates(robot_states)

	force = {
		"FT_sensor_raw": robot_states.ftSensorRaw,
		"ext_tcp": 	robot_states.extWrenchInTcp,
		"ext_base": robot_states.extWrenchInBase,
		"q": robot_states.q,
		"tau": robot_states.tau,
		# "tau_e": robot_states.tau_e,
		# "tau_ext": robot_states.tau_ext,
	}

	return force
	

def print_robot_states(robot, log):
	"""
	Print robot states data
	"""
	# Data struct storing robot states
	robot_states = flexivrdk.RobotStates()
	
	# Get the latest robot states
	robot.getRobotStates(robot_states)

	# Print all gripper states, round all float values to 2 decimals
	log.info("Current robot states:")
	# fmt: off
	print("{")
	print("q: ",  ['%.2f' % i for i in robot_states.q])
	print("theta: ", ['%.2f' % i for i in robot_states.theta])
	print("dq: ", ['%.2f' % i for i in robot_states.dq])
	print("dtheta: ", ['%.2f' % i for i in robot_states.dtheta])
	print("tau: ", ['%.2f' % i for i in robot_states.tau])
	print("tau_des: ", ['%.2f' % i for i in robot_states.tauDes])
	print("tau_dot: ", ['%.2f' % i for i in robot_states.tauDot])
	print("tau_ext: ", ['%.2f' % i for i in robot_states.tauExt])
	print("tcp_pose: ", ['%.2f' % i for i in robot_states.tcpPose])
	print("tcp_pose_d: ", ['%.2f' % i for i in robot_states.tcpPoseDes])
	print("tcp_velocity: ", ['%.2f' % i for i in robot_states.tcpVel])
	print("camera_pose: ", ['%.2f' % i for i in robot_states.camPose])
	print("flange_pose: ", ['%.2f' % i for i in robot_states.flangePose])
	print("FT_sensor_raw_reading: ", ['%.2f' % i for i in robot_states.ftSensorRaw])
	print("F_ext_tcp_frame: ", ['%.2f' % i for i in robot_states.extWrenchInTcp])
	print("F_ext_base_frame: ", ['%.2f' % i for i in robot_states.extWrenchInBase])
	print("}")
	
def print_gripper_states(gripper, log):
	"""
	Print gripper states data @ 1Hz.

	"""
	# Data struct storing gripper states
	gripper_states = flexivrdk.GripperStates()
	
	# Get the latest gripper states
	gripper.getGripperStates(gripper_states)

	# Print all gripper states, round all float values to 2 decimals
	log.info("Current gripper states:")
	print("width: ", round(gripper_states.width, 2))
	print("force: ", round(gripper_states.force, 2))
	print("max_width: ", round(gripper_states.maxWidth, 2))
	print("is_moving: ", gripper_states.isMoving)

def get_robot_joints(robot) -> List[float]:
	"""get robot current joints in rad

	Args:
		robot (_type_): _description_

	Returns:
		List[float]: joint positions
	"""
	
	robot_states = flexivrdk.RobotStates()
	robot.getRobotStates(robot_states)
	return robot_states.q

def get_hand_joints(hand):
	return hand.read_pos()

def get_gripper_width(gripper) -> float:
	gripper_states = flexivrdk.GripperStates()
	gripper.getGripperStates(gripper_states)
	return gripper_states.width

def get_flange_pose(robot) -> List[float]:
	robot_states = flexivrdk.RobotStates()
	robot.getRobotStates(robot_states)
	return robot_states.flangePose

def safe_move(robot, dt: float):
	stale_time = time.time() + dt
	robot_states = flexivrdk.RobotStates()
	while True:
		'''force update'''
		robot.getRobotStates(robot_states)
		force = np.array(robot_states.extWrenchInTcp[:3])
		force_norm = np.linalg.norm(force)
		if robot_states.tcpPose[2] < 0.04:
			print("tcp is too low")
			raise RuntimeError
		if force_norm > 20:
			print("larger than force threshold")
			raise RuntimeError
		if time.time() > stale_time:
			break
		
def get_obs(task,hand=None, cam_agent=None, cam_wrist=None, cam_bird=None, cam_left=None, cam_right=None):
	obs_dict = {}
	obs_dict["obs"] = torch.clamp(task.env.obs_buf, -task.env.clip_obs, task.env.clip_obs).to(task.env.rl_device)
	obs_dict['target'] = task.env.cur_targets
	obs_dict['force'] = get_robot_force()

	global recording_data
	if not recording_data:
		return 
	
	if test_on_real_robot:
		global robot, gripper
		robot_states = flexivrdk.RobotStates()
		robot.getRobotStates(robot_states)
		
		obs_dict["joints"] = robot_states.q
		obs_dict["tcpPose"] = robot_states.tcpPose
		obs_dict["flangePose"] = robot_states.flangePose
		obs_dict["gripperWidth"] = get_gripper_width(gripper)

	if test_on_real_hand:
		hand_joints = hand.read_pos()-3.14
		obs_dict['leap_hand']=hand_joints

		set_pose=np.zeros(28)
		for i, key in enumerate([12,22,27]):
			tip = key
			tip = tip-1
			finger_tip = tip-1
			dip = tip - 2
			pip = tip - 3

			set_pose[dip] = hand_joints[i*4]
			set_pose[pip] = hand_joints[i*4+1] 
			set_pose[finger_tip] = hand_joints[i*4+2] 
			set_pose[tip] = hand_joints[i*4+3] 

		tip = 16
		finger_tip = tip - 1
		dip = tip - 2
		pip = tip - 3
		set_pose[pip] = hand_joints[12]
		set_pose[dip] = hand_joints[13]
		set_pose[finger_tip] = hand_joints[14]*-1
		set_pose[tip] = hand_joints[15]*-1
		
		obs_dict['hand_joints'] = set_pose

	task.obs_dict.append(obs_dict)

	if cam_agent is not None:
		img, depth = cam_agent.get_data()
		img = img.copy()
		depth = depth.copy()
		cam_agent_view.append({
			"image": img,
			"depth": depth,
		})
		
	if cam_wrist is not None:
		img, depth = cam_wrist.get_data()
		img = img.copy()
		depth = depth.copy()
		cam_wrist_view.append({
			"image": img,
			"depth": depth,
		})
	
	if cam_bird is not None:
		img, depth = cam_bird.get_data()
		img = img.copy()
		depth = depth.copy()
		cam_bird_view.append({
			"image": img,
			"depth": depth,
		})
		
	if cam_left is not None:
		img, depth = cam_left.get_data()
		img = img.copy()
		depth = depth.copy()
		cam_left_view.append({
			"image": img,
			"depth": depth,
		})

			
	if cam_right is not None:
		img, depth = cam_right.get_data()
		img = img.copy()
		depth = depth.copy()
		cam_right_view.append({
			"image": img,
			"depth": depth,
		})

def is_gripper_moving(gripper) -> bool: 
	gripper_states = flexivrdk.GripperStates()
	gripper.getGripperStates(gripper_states)
	return gripper_states.isMoving

def self_exam(log):
	global robot
	try:
		robot = flexivrdk.Robot(robot_ip, local_ip)

		# Clear fault on robot server if any
		if robot.isFault():
			log.warn("Fault occurred on robot server, trying to clear ...")
			# Try to clear the fault
			robot.clearFault()
			time.sleep(2)
			# Check again
			if robot.isFault():
				log.error("Fault cannot be cleared, exiting ...")
				return
			log.info("Fault on robot server is cleared")

		# Enable the robot, make sure the E-stop is released before enabling
		log.info("Enabling robot ...")
		robot.enable()

		# Wait for the robot to become operational
		while not robot.isOperational():
			time.sleep(1)

		log.info("Robot is now operational")
		
		# Enable the robot, make sure the E-stop is released before enabling
		log.info("Enabling robot ...")
		robot.enable()
		while not robot.isOperational():
			time.sleep(1)
		log.info("Robot is now operational")
		
	except Exception as e:
		# Print exception error message
		log.error(str(e))


# def self_exam_leaphand(log):
#     global leap_hand
#     try:
#         leap_hand = LeapNode()
#         while True:
#             leap_hand.set_allegro(np.zeros(16))
#             time.sleep(0.03)


	except Exception as e:
		# Print exception error message
		log.error(str(e))


@torch.jit.script
def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[0:3] * torch.sign(q_r[3]).unsqueeze(-1)

def leap_axis_correction(vec):
	""" zxy coordinate -> xyz coordinate"""
	vec = vec[..., [2, 0, 1]]
	vec[..., :2] = -vec[..., :2]
	return vec

def sigma_axis_correction(vec):
	vec[..., :2] = -vec[..., :2]
	return vec

def is_validpos(tcp_pos):
	"""check if tcp_pos is within the valid range"""
	return torch.all(tcp_pos > tcp_lower_bound) and torch.all(tcp_pos < tcp_upper_bound)

def trans_quat(q):
	""" quaternion xyzw -> wxyz"""
	return torch.cat([q[3:], q[:3]])

def move_home(robot):
	log.info("Moving to home pose")
	robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
	robot.executePrimitive("Home()")
	# Wait for the primitive to finish
	while robot.isBusy():
		time.sleep(1)        
	robot.executePrimitive("ZeroFTSensor()")
		
def main(task):
	# replay_traj(task)
	# diffusion_teleop(task)
	# realworld_exec(task)
	# sigma_teleop(task)
	# leap_teleop(task)
	# flexiv_leap_teleop(task)
	# flexiv_leap_teleop_dexretarget(task)
	vr_gripper_teleop(task)
	# vr_teleop(task)
	# vr_teleop_dexretarget(task)
	# leap_diffusion_teleop(task)
	
	# leap_umi_gopro_teleop(task)
	# koch_umi_gopro_teleop(task)
	# koch_umi_gopro_tactile_teleop(task)
	# umi_trajectory_only_teleop(task)
	print(">>>>>>>>>>>>>>>>>Warning!<<<<<<<<<<<<<<<<")
	print("test_on_real_robot: ", test_on_real_robot)

def realworld_exec(task: TeleopPlayer):
	assert test_on_real_robot
	
	from .calibration.camera import CameraD400
	cam_agent = CameraD400(1)
	cam_wrist = CameraD400(0)
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		gripper.move(0.09, 0.1, 10)
		
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval",
	)

	assert DOF > 0, "DOF must be greater than 0"
	
	# ================== Diffusion Actor ====================
	from torch.utils.data import DataLoader
	from .diffusha_flexiv.diffusion.train import Trajectory
	from .diffusha_flexiv.config.default_args import Args
	from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
	from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor
	
	print(env.flexiv_dof_lower_limits.shape)
	device = env.device
	lower_limits = env.flexiv_dof_lower_limits[:DOF]
	upper_limits = env.flexiv_dof_upper_limits[:DOF]
	lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
	upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])
	
	# dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=4)
	# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
	
	# model_pt = "/home/rhos/Desktop/flexiv/data/ddpm/diffusha-flexiv/m5yq5o60/step_00004000.pt"
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/x084y7qb/step_00002500.pt"   # beta_max=1e-2
	# model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/ybbhnl43/step_00002500.pt" # beta_max=5e-3
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/usx3fnbh/step_00008000.pt"   # beta_max=1e-1, step=50
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006s000.pt"   # hidden_dim=256, step=50, fwd_ratio=0.5
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/zqlkkewt/step_00007000.pt"   #hidden_dim=512, step=100, beta_max=1e-2, horizon=6
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/4wrbxope/step_00007000.pt"   #horizon=12, hidden_dim=512, step=100, beta_max=1e-2
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/00v2rgcl/step_00006000.pt"  # horizon=8, diff+omega(20+20), beta_max=1e-2
	# model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/pfo1i9r4/step_00006000.pt"  # horizon=8, diff+hum(20+20), beta_max=1e-2
	
	diffusion = DiffusionModel(
		diffusion_core=DiffusionCore(),
		num_diffusion_steps=Args.num_diffusion_steps,
		input_size=(Args.copilot_obs_size + Args.act_size),
		beta_schedule=Args.beta_schedule,
		beta_min=Args.beta_min,
		beta_max=Args.beta_max,
		cond_dim=Args.copilot_obs_size
	)
	
	diffusion.load_ckpt(model_pt)
	diffusion_actor = DiffusionAssistedActor(-1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio)
	# ======================= End of Diffusion Actor ==================
	
	env.actions = torch.zeros((1, env.num_dofs))    
	
	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	start_sample_time = time.time()
	while not env.gym.query_viewer_has_closed(env.viewer):    
		
			task.step()
			
			handle_viewer_events(task, viewer)  
			draw_bounding_box()
			
			if not recording_data:
				continue
			
			if test_on_real_robot and robot.isFault():
				raise Exception("Fault occurred on robot server, exiting ...")
			
			# =============== Diffusion Robot ============
			tq = torch.from_numpy(np.array(get_flange_pose(robot))).to(torch.float32)
			tq[3:] /= torch.norm(tq[3:])
			griv = get_gripper_width(gripper)
			
			agent_state = torch.cat([tq, torch.tensor([griv]).to(device)]).unsqueeze(0)
			
			agent_img, _ = cam_agent.get_data()
			agent_img = cv2.resize(agent_img, (320, 240))
			agent_img = agent_img.astype(np.float32).transpose((2, 0, 1)) / 255
			agent_img = torch.from_numpy(agent_img).unsqueeze(0).to(device)
			
			wrist_img, _ = cam_wrist.get_data()
			wrist_img = cv2.resize(wrist_img, (320, 240))
			wrist_img = wrist_img.astype(np.float32).transpose((2, 0, 1)) / 255
			wrist_img = torch.from_numpy(wrist_img).unsqueeze(0).to(device)
			
			batch = (agent_state, wrist_img, agent_img)
			
			agent_action = torch.zeros((1, (DOF+1) * Args.horizon)).to(device)

			agent_action = diffusion_actor.assisted_act(agent_action, batch).to(device).squeeze()
			agent_action = tensor_clamp(agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action))
			agent_action = agent_action.view(-1, 8)
			agent_action = scale(agent_action, lower_limits, upper_limits)
			
			# _ = input()
			
			# agent_action = agent_action.mean(dim=0).unsqueeze(0) # TODO:
			for agent_act in agent_action: # number of Args.horizon action
				
				# ==================== End Diffusion Robot ========================
				env.cur_targets[:, :DOF] = 0.7 * env.cur_targets[:, :DOF] + 0.3 * agent_act[:DOF]
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

				griv = get_gripper_width(gripper)
				print(griv, agent_act[-1])
				griv = agent_act[-1] # TODO:
				
				try:
					# action = env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF]
					action = env.cur_targets[:, :DOF]
					action = action[0][:DOF].tolist()
					
					robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
					
					flag = False
					# open gripper
					if griv > 0.08 and get_gripper_width(gripper) < 0.03 and not is_gripper_moving(gripper):  
						gripper.move(0.09, 0.1, 20)
						flage = True
					# close gripper
					if griv < 0.055 and get_gripper_width(gripper) > 0.08 and not is_gripper_moving(gripper):
						gripper.move(0, 0.5, 5)
						flag = True

					safe_move(robot, dt)

					if flag:
						break
				
				except Exception as e:
					log.error(str(e))
					exit(0) 
			# print("="*10)
			# _ = input()
			
			
	exit(0)  
	
def replay_traj(task: TeleopPlayer):
	# sys.path.append('/home/rhos/Desktop/flexiv')
	# from dataloader import *
	from torch.utils.data import DataLoader
	from .diffusha_flexiv.diffusion.train import Trajectory
	from .diffusha_flexiv.config.default_args import Args
	from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
	from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	
	griv = 0.
	DOF = 7
	
	print(env.flexiv_dof_lower_limits.shape)
	device = env.device
	lower_limits = env.flexiv_dof_lower_limits[:DOF]
	upper_limits = env.flexiv_dof_upper_limits[:DOF]
	lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
	upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])
	
	dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=6)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
	
	# model_pt = "/home/rhos/Desktop/flexiv/data/ddpm/diffusha-flexiv/m5yq5o60/step_00009999.pt"
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/x084y7qb/step_00002500.pt"   # beta_max=1e-2, step=100
	# model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/ybbhnl43/step_00002500.pt" # beta_max=5e-3, step=100
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/usx3fnbh/step_00005000.pt"   # beta_max=1e-1, step=50
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006000.pt"   #hidden_dim=256, step=50
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/zqlkkewt/step_00006000.pt"   #hidden_dim=512, step=100, beta_max=1e-2, horizon=6
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/00v2rgcl/step_00006000.pt"  # horizon=8, diff+omega(20+20), beta_max=1e-2
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/i01k9rov/step_00000500.pt"  # horizon=8, hum+omega(20+20), beta_max=1e-2, seed=233
	
	diffusion = DiffusionModel(
		diffusion_core=DiffusionCore(),
		num_diffusion_steps=Args.num_diffusion_steps,
		input_size=(Args.copilot_obs_size + Args.act_size),
		beta_schedule=Args.beta_schedule,
		beta_min=Args.beta_min,
		beta_max=Args.beta_max,
		cond_dim=Args.copilot_obs_size
	)
	
	diffusion.load_ckpt(model_pt)
	diffusion_actor = DiffusionAssistedActor(-1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio)
		
	env.actions = torch.zeros((1, env.num_dofs))    
	
	for _ in range(1):
		reset_joints(task, home_joint)    
		for _ in range(100):
			task.step()
			
		len_data= len(dataloader)
		for _, batch in enumerate(dataloader):
		
		# for idx in range(len_data):
			# batch = dataloader[idx]
			
			batch = [b.to(device) for b in batch]
			
			task.step()
			
			act = batch[-1]
			act = scale(act.view(-1, 8), lower_limits, upper_limits).reshape(-1, Args.act_size)
			
			act_dof = act[..., :DOF]
			print(act_dof, lower_limits)
			
			act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
			act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
			
			print(act_dof, lower_limits, upper_limits)
			
			# =============== Diffusion Robot ============
			
			# cur_state = torch.squeeze(env.rigid_body_states[:, env.num_flexiv_bodies:2*env.num_flexiv_bodies])
			cur_state = torch.squeeze(env.rigid_body_states[:, :env.num_flexiv_bodies])
			cur_rb_idx = 8 # flange
			tq = cur_state[cur_rb_idx][:7]
			
			tq[3:] /= torch.norm(tq[3:])
			tq[3:] = trans_quat(tq[3:])
			# print(tq, batch[0][..., :DOF])
			batch[0][..., :DOF] = tq
			
			agent_action = diffusion_actor.assisted_act(torch.zeros_like(act), batch).to(device).squeeze()
			agent_action = tensor_clamp(agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action))
			
			agent_action = agent_action.view(-1, 8)
			
			# print("@@@: ", batch[0])
			for agent_act in agent_action:
				agent_act = scale(agent_act, lower_limits, upper_limits)
				# print(batch[0], agent_action)
				# print(act, agent_action)
				# ============================================
				# print(agent_act)
				env.cur_targets[:, :DOF] = 0.7 * env.cur_targets[:, :DOF] + 0.3 * agent_act[:DOF]
				# env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF] = 0.7 * env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF] + 0.3 * agent_action[:DOF]
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
				
				task.step()
			# print("="*10)
			# idx += Args.horizon - 1
			
			# _ = input()
			# agent_action = diffusion_actor.assisted_act(act, batch)
			
	exit(0)
		
def diffusion_teleop(task: TeleopPlayer):
	sys.path.append('/home/rhos/sigma_sdk')
	import sigma7
	from .calibration.camera import CameraD400
	ik  = IKController(damping=0.05)
	task.init_ee_state = None
	cam_agent = CameraD400(0)
	cam_wrist = CameraD400(1)
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		gripper.move(0.09, 0.1, 10)
		
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval",
	)

	assert DOF > 0, "DOF must be greater than 0"
	
	# ================== Diffusion Actor ====================
	from torch.utils.data import DataLoader
	from .diffusha_flexiv.diffusion.train import Trajectory
	from .diffusha_flexiv.config.default_args import Args
	from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
	from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor
	
	print(env.flexiv_dof_lower_limits.shape)
	device = env.device
	lower_limits = env.flexiv_dof_lower_limits[:DOF]
	upper_limits = env.flexiv_dof_upper_limits[:DOF]
	lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
	upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])
	
	# dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=4)
	# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
	
	# model_pt = "/home/rhos/Desktop/flexiv/data/ddpm/diffusha-flexiv/m5yq5o60/step_00004000.pt"
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/x084y7qb/step_00002500.pt" # beta_max=1e-2
	# model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/ybbhnl43/step_00003500.pt" # beta_max=5e-3
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006000.pt"   # hidden_dim=256, step=50
	
	diffusion = DiffusionModel(
		diffusion_core=DiffusionCore(),
		num_diffusion_steps=Args.num_diffusion_steps,
		input_size=(Args.copilot_obs_size + Args.act_size),
		beta_schedule=Args.beta_schedule,
		beta_min=Args.beta_min,
		beta_max=Args.beta_max,
		cond_dim=Args.copilot_obs_size
	)
	
	diffusion.load_ckpt(model_pt)
	diffusion_actor = DiffusionAssistedActor(-1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio)
	# ======================= End of Diffusion Actor ==================
	
	env.actions = torch.zeros((1, env.num_dofs))    
	
	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	sigma7.drdOpen()
	sigma7.drdAutoInit()
	sigma7.drdStart()
	sigma7.drdMoveTo(np.array([0, 0, 0, 0, 0, 0, 0, 0]), block=True)
	sigma7.drdRegulatePos(on = False)
	sigma7.drdRegulateRot(on = False)
	sigma7.drdRegulateGrip(on = False)
	sigma7.drdSetForceAndWristJointTorquesAndGripperForce(0, 0, 0.3, 0.1, 0, 0, 0)

	def display_images():
		while True:
			img_agent, _ = cam_agent.get_data()
			img_agent = img_agent.copy()
			img_wrist, _ = cam_wrist.get_data()
			img_wrist = img_wrist.copy()
			img = np.concatenate((img_agent, img_wrist), axis=1)
			cv2.imshow('Image', img)
			cv2.waitKey(50)

	# thread = threading.Thread(target=display_images)
	# thread.start()

	start_sample_time = time.time()
	set_real_robot_time = time.time()
	
	while not env.gym.query_viewer_has_closed(env.viewer):
	# for _ in range(10):
	#     reset_home(task)
	#     task.initialized             = False
	#     task.prev_motion_joint       = None
	#     task.obs_dict                = []
	#     for _, batch in enumerate(dataloader):
	#         batch = [b.to(device) for b in batch]
			
			task.step()
			handle_viewer_events(task, viewer)
			draw_bounding_box()
			
			if not recording_data: # TODO: turn on when testing on real robot
				continue
			
			if time.time() - start_sample_time > dt:
				get_obs(task, cam_agent, cam_wrist)
				start_sample_time = time.time()
			
			if test_on_real_robot and robot.isFault():
				raise Exception("Fault occurred on robot server, exiting ...")
				
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])

			if task.prev_motion_joint is None:
				print("\033[91mPlease move sigma.7 to a comfortable pose...\033[0m")
				cur_time = time.time()
				while True:
					task.step()
					if time.time() - cur_time > 2.0:
						break
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = sigma7.drdGetPositionAndOrientation()
				task.init_ee_state = cur_state[8].clone().cpu()
				
			# === X, Y, Z === 
			_, px, py, pz, oa, ob, og, pg, mat = sigma7.drdGetPositionAndOrientation()
			delta_pos = (np.array([px, py, pz]) - task.prev_motion_joint[1:4]) * sigma_scaling_factor
			delta_pos[1] *= 1.5
			delta_pos = sigma_axis_correction(delta_pos)
			
			dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
			mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
			
			cur_rb_idx = 8
			cur_pos = cur_state[cur_rb_idx][:3]
			cur_rot_q = cur_state[cur_rb_idx][3:7]
			cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
			z_axis = cur_rot_mat[:, 2]
			
			target_pos = task.init_ee_state[:3] + delta_pos
			
			dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
			mask[cur_rb_idx, :3]  = True
			
			delta_euler = np.array([oa, ob, og]) - task.prev_motion_joint[4:7]
			init_rot_e = R.from_quat(task.init_ee_state[3:7].cpu().numpy()).as_euler('xyz')
			delta_euler[1:] *= -1
			target_rot_q =  R.from_euler('xyz', init_rot_e) * R.from_euler('xyz', delta_euler)
			target_rot_q = torch.from_numpy(target_rot_q.as_quat()).type_as(cur_rot_q)
			# print(delta_euler, target_rot_q)
			
			target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
			task.draw_line(target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
			task.draw_line(target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
			task.draw_line(target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))
			
			dpose[cur_rb_idx, 3:] = orientation_error(target_rot_q, cur_rot_q)
			mask[cur_rb_idx, 3:]  = True
			
			tcp_pos = target_pos + z_axis * 0.15
			
			task.draw_sphere(target_pos)
			task.draw_sphere(tcp_pos, color=(0, 0, 0))
				
			# if not is_validpos(tcp_pos):
			#     continue
			
			fake_slice = torch.ones((1, 6, env.num_flexiv_dofs), device=env.device)
			fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

			action = ik.control_ik(dpose, fake_jacobian, mask)
			next_dof_pos = env.flexiv_dof_pos + action
			
			act_dof = 0.90 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.10 * next_dof_pos
			act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
			
			# =============== Diffusion Robot ============
			delta_action = act_dof - env.flexiv_dof_pos
			agent_action = env.flexiv_dof_pos + delta_action 
			
			griv = 0.09 if pg < -0.015 else 0.01 #TODO:
			# griv = get_gripper_width(gripper)
			
			cur_state = torch.squeeze(env.rigid_body_states[:, :env.num_flexiv_bodies])
			cur_rb_idx = 8 # flange
			tq = cur_state[cur_rb_idx][:7]
			tq[3:] /= torch.norm(tq[3:])
			tq[3:] = trans_quat(tq[3:])
			
			if test_on_real_robot:
				tq = torch.from_numpy(np.array(get_flange_pose(robot))).to(torch.float32)
				tq[3:] /= torch.norm(tq[3:])
			
			# batch[0][..., :DOF] = tq   # change it into current xyz + quat 
			# batch[0][..., DOF:] = griv # change it into current grivWidth
			
			agent_state = torch.cat([tq, torch.tensor([griv]).to(device)]).unsqueeze(0)
			
			agent_img, _ = cam_agent.get_data()
			agent_img = cv2.resize(agent_img, (320, 240))
			agent_img = agent_img.astype(np.float32).transpose((2, 0, 1)) / 255
			agent_img = torch.from_numpy(agent_img).unsqueeze(0).to(device)
			
			wrist_img, _ = cam_wrist.get_data()
			wrist_img = cv2.resize(wrist_img, (320, 240))
			wrist_img = wrist_img.astype(np.float32).transpose((2, 0, 1)) / 255
			wrist_img = torch.from_numpy(wrist_img).unsqueeze(0).to(device)
			
			batch = (agent_state, wrist_img, agent_img)
			
			agent_action = torch.cat([agent_action[..., :DOF].squeeze(), torch.tensor([griv]).to(device)]) # add griv dimension
			agent_action = unscale(agent_action.unsqueeze(0), lower_limits, upper_limits)
			agent_action = agent_action.repeat(1, Args.horizon).view(-1, 8 * Args.horizon) # repeat horizon times
			# agent_action = torch.zeros_like(agent_action) # TODO: autonomous agent's input

			agent_action = diffusion_actor.assisted_act(agent_action, batch).to(device).squeeze()
			agent_action = tensor_clamp(agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action))
			agent_action = scale(agent_action.view(-1, 8), lower_limits, upper_limits)
			
			# ==================== End Diffusion Robot ========================
			agent_action = agent_action[:4].mean(dim=0).unsqueeze(0)
			
			for act_next in agent_action:
				# env.cur_targets[:, :DOF] = 0.7 * env.cur_targets[:, :DOF] + 0.3 * act_next[:DOF]
				env.cur_targets[:, :DOF] = 0.3 * env.cur_targets[:, :DOF] + 0.7 * act_next[:DOF]
		
				env.cur_targets[:, :DOF] = 0.7 * env.cur_targets[:, :DOF] + 0.3 * act_dof[..., :DOF] # TODO: blend with original human action
				
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

				task.step(clear_lines=False)
				handle_viewer_events(task, viewer)
				draw_bounding_box()
	
				if not recording_data: # TODO: turn on when testing on real robot
					break
				
				griv = act_next[-1] # TODO:
				print('grivWidth: ', griv)
				
				if test_on_real_robot:
					try:
						# action = env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF]
						action = env.cur_targets[:, :DOF]
						action = action[0][:DOF].tolist()
						
						robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
						# open gripper
						if griv > 0.06 and get_gripper_width(gripper) < 0.03 and not is_gripper_moving(gripper):  
							gripper.move(0.09, 0.1, 20)
						# close gripper
						if griv < 0.075 and get_gripper_width(gripper) > 0.08 and not is_gripper_moving(gripper):
							gripper.grasp(5) 
							
						safe_move(robot, dt)
						
					except Exception as e:
						log.error(str(e)) 
						exit(0)   
				
				# ===== sync joint state =====
				if test_on_real_robot:
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
				
			
					
	exit(0)  
				
def calibration(task: TeleopPlayer):
	sys.path.append('/home/rhos/sigma_sdk')
	import sigma7, cv2
	ik  = IKController(damping=0.05)
	task.init_ee_state = None
	from .calibration.camera import CameraD400
	cam = CameraD400(1)
	chessboard_size = (8, 11)
	saved_img = 0
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		gripper = flexivrdk.Gripper(robot)
		
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
		
		try:
			gripper.move(0.01, 0.1, 20)
		except Exception as e:
			log.error(str(e))
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval",
	)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))
	
	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	sigma7.drdOpen()
	sigma7.drdAutoInit()
	sigma7.drdStart()
	sigma7.drdMoveTo(np.array([0, 0, 0, 0, 0, 0, 0, 0]), block=True)
	sigma7.drdRegulatePos(on = False)
	sigma7.drdRegulateRot(on = False)
	sigma7.drdRegulateGrip(on = False)
	sigma7.drdSetForceAndWristJointTorquesAndGripperForce(0, 0, 0.3, 0.1, 0, 0, 0)
	
	calib_time = time.time()
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		get_obs(task)
		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		# joints = task.detector.detect_right_joints()
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])

		if task.prev_motion_joint is None:
			print("\033[91mPlease move sigma.7 to a comfortable pose...\033[0m")
			cur_time = time.time()
			while True:
				task.step()
				if time.time() - cur_time > 3.0:
					break
			print("\033[92mSampled\033[0m")
			task.prev_motion_joint = sigma7.drdGetPositionAndOrientation()
			task.init_ee_state = cur_state[8].clone().cpu()
			
		# === X, Y, Z === 
		_, px, py, pz, oa, ob, og, pg, mat = sigma7.drdGetPositionAndOrientation()
		delta_pos = (np.array([px, py, pz]) - task.prev_motion_joint[1:4]) * sigma_scaling_factor
		delta_pos[1] *= 1.5
		delta_pos = sigma_axis_correction(delta_pos)
		   
		dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
		mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
		
		cur_rb_idx = 8
		cur_pos = cur_state[cur_rb_idx][:3]
		cur_rot_q = cur_state[cur_rb_idx][3:7]
		cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
		z_axis = cur_rot_mat[:, 2]
		
		target_pos = task.init_ee_state[:3] + delta_pos
		
		dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
		mask[cur_rb_idx, :3]  = True
		
		delta_euler = np.array([oa, ob, og]) - task.prev_motion_joint[4:7]
		init_rot_e = R.from_quat(task.init_ee_state[3:7].cpu().numpy()).as_euler('xyz')
		delta_euler[1:] *= -1
		target_rot_q =  R.from_euler('xyz', init_rot_e) * R.from_euler('xyz', delta_euler)
		target_rot_q = torch.from_numpy(target_rot_q.as_quat()).type_as(cur_rot_q)
		# print(delta_euler, target_rot_q)
		
		target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
		task.draw_line(target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
		task.draw_line(target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
		task.draw_line(target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))
		
		dpose[cur_rb_idx, 3:] = orientation_error(target_rot_q, cur_rot_q)
		mask[cur_rb_idx, 3:]  = True
		
		tcp_pos = target_pos + z_axis * 0.15
		
		task.draw_sphere(target_pos)
		task.draw_sphere(tcp_pos, color=(0, 0, 0))
			
		if not is_validpos(tcp_pos):
			continue
		
		fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
		fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

		action = ik.control_ik(dpose, fake_jacobian, mask)
		next_dof_pos = env.flexiv_dof_pos + action
		
		act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
		act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
		
		env.cur_targets[:, :env.num_flexiv_dofs] = act_dof
		env.actions = act_dof
		gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
			
		if test_on_real_robot:
			try:
				action = act_dof[0][:DOF].tolist()
				robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
					
			except Exception as e:
				log.error(str(e))    
			
			
			img, _ = cam.get_data()
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cv2.imshow('Image', img)
			cv2.waitKey(1) 
			
			if time.time() - calib_time < 1.5:
				continue
			
			calib_time = time.time()
			ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
			if ret == True:
				saved_img += 1
				filename = f"/home/rhos/Desktop/cali/{saved_img:03}.png"
				print("chessboard scanned!")
				cv2.imwrite(filename, img)
				flange_pose = get_flange_pose(robot)
				np.savetxt(f"/home/rhos/Desktop/cali/{saved_img:03}.txt", flange_pose)
				np.savetxt(f"/home/rhos/Desktop/cali/pose/{saved_img:03}.txt", action)
				
				cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
				cv2.imshow('Image', img)
				cv2.waitKey(500)  
				 
def sigma_teleop(task: TeleopPlayer):
	sys.path.append('/home/rhos/sigma_sdk')
	import sigma7
	from .calibration.camera import CameraD400
	ik  = IKController(damping=0.05)
	task.init_ee_state = None
	cam_agent = CameraD400(0)
	cam_wrist = CameraD400(1)
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
		
		try:
			gripper.move(0.01, 0.1, 20)
		except Exception as e:
			log.error(str(e))
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval",
	)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))
	
	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	sigma7.drdOpen()
	sigma7.drdAutoInit()
	sigma7.drdStart()
	sigma7.drdMoveTo(np.array([0, 0, 0, 0, 0, 0, 0, 0]), block=True)
	sigma7.drdRegulatePos(on = False)
	sigma7.drdRegulateRot(on = False)
	sigma7.drdRegulateGrip(on = False)
	sigma7.drdSetForceAndWristJointTorquesAndGripperForce(0, 0, 0.3, 0.1, 0, 0, 0)

	def display_images():
		while True:
			img_agent, _ = cam_agent.get_data()
			img_agent = img_agent.copy()
			img_wrist, _ = cam_wrist.get_data()
			img_wrist = img_wrist.copy()
			img = np.concatenate((img_agent, img_wrist), axis=1)
			cv2.imshow('Image', img)
			cv2.waitKey(50)

	# thread = threading.Thread(target=display_images)
	# thread.start()

	start_sample_time = time.time()
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		
		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if not recording_data:
			continue
		
		if time.time() - start_sample_time > dt:
			get_obs(task, cam_agent, cam_wrist)
			start_sample_time = time.time()
		cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])

		if task.prev_motion_joint is None:
			print("\033[91mPlease move sigma.7 to a comfortable pose...\033[0m")
			cur_time = time.time()
			while True:
				task.step()
				if time.time() - cur_time > 2.0:
					break
			print("\033[92mSampled\033[0m")
			task.prev_motion_joint = sigma7.drdGetPositionAndOrientation()
			task.init_ee_state = cur_state[8].clone().cpu()
			
		# === X, Y, Z === 
		_, px, py, pz, oa, ob, og, pg, mat = sigma7.drdGetPositionAndOrientation()
		delta_pos = (np.array([px, py, pz]) - task.prev_motion_joint[1:4]) * sigma_scaling_factor
		delta_pos[1] *= 1.5
		delta_pos = sigma_axis_correction(delta_pos)
		   
		dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
		mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
		
		cur_rb_idx = 8
		cur_pos = cur_state[cur_rb_idx][:3]
		cur_rot_q = cur_state[cur_rb_idx][3:7]
		cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
		z_axis = cur_rot_mat[:, 2]
		
		target_pos = task.init_ee_state[:3] + delta_pos
		
		dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
		mask[cur_rb_idx, :3]  = True
		
		delta_euler = np.array([oa, ob, og]) - task.prev_motion_joint[4:7]
		init_rot_e = R.from_quat(task.init_ee_state[3:7].cpu().numpy()).as_euler('xyz')
		delta_euler[1:] *= -1
		target_rot_q =  R.from_euler('xyz', init_rot_e) * R.from_euler('xyz', delta_euler)
		target_rot_q = torch.from_numpy(target_rot_q.as_quat()).type_as(cur_rot_q)
		# print(delta_euler, target_rot_q)
		
		target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
		task.draw_line(target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
		task.draw_line(target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
		task.draw_line(target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))
		
		dpose[cur_rb_idx, 3:] = orientation_error(target_rot_q, cur_rot_q)
		mask[cur_rb_idx, 3:]  = True
		
		tcp_pos = target_pos + z_axis * 0.15
		
		task.draw_sphere(target_pos)
		task.draw_sphere(tcp_pos, color=(0, 0, 0))
			
		if not is_validpos(tcp_pos):
			# task.prev_motion_joint = sigma7.drdGetPositionAndOrientation()
			# task.init_ee_state = cur_state[8].clone().cpu()
			continue
		
		fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
		fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

		action = ik.control_ik(dpose, fake_jacobian, mask)
		next_dof_pos = env.flexiv_dof_pos + action
		
		act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
		act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
		
		env.cur_targets[:, :env.num_flexiv_dofs] = act_dof
		env.actions = act_dof
		gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
			
		if test_on_real_robot:
			try:
				action = act_dof[0][:DOF].tolist()
				robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
				
				# open gripper
				if pg < -0.015 and get_gripper_width(gripper) < 0.05 and not is_gripper_moving(gripper):  
					gripper.move(0.09, 0.1, 20)
				# close gripper
				if pg > -0.007 and get_gripper_width(gripper) > 0.06 and not is_gripper_moving(gripper):
					# gripper.move(0.015, 0.1, 20) 
					gripper.grasp(5)
					
			except Exception as e:
				log.error(str(e))        
				
def leap_diffusion_teleop(task: TeleopPlayer):
	OPERATOR2MANO_RIGHT = np.array(
	[
		[0, 1, 0.],
		[0, 0, 1],
		[-1, 0, 0],
	])
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from leap.LeapDualJointDetector import Detector
	from .calibration.camera import CameraD400
	
	task.detector = Detector(up_axis='y',scale=0.001)
	ik  = IKController(damping=0.05)
	cam_agent = CameraD400(0)
	cam_wrist = CameraD400(1)
	
	task.init_ee_state = None
	last_delta_pos = np.zeros(6)
	last_wrist_pos = np.zeros(3)
	last_detect_time = time.time()
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		try:
			gripper.move(0.09, 0.1, 20)
		except Exception as e:
			log.error(str(e))
			
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval",
	)

	# ================== Diffusion Actor ====================
	from torch.utils.data import DataLoader
	from .diffusha_flexiv.diffusion.train import Trajectory
	from .diffusha_flexiv.config.default_args import Args
	from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
	from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor
	
	print(env.flexiv_dof_lower_limits.shape)
	device = env.device
	lower_limits = env.flexiv_dof_lower_limits[:DOF]
	upper_limits = env.flexiv_dof_upper_limits[:DOF]
	lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
	upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])
	
	# dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=4)
	# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006000.pt"   # hidden_dim=256, step=50
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/4wrbxope/step_00006000.pt"   #horizon=12, hidden_dim=512, step=100, beta_max=1e-2
	
	diffusion = DiffusionModel(
		diffusion_core=DiffusionCore(),
		num_diffusion_steps=Args.num_diffusion_steps,
		input_size=(Args.copilot_obs_size + Args.act_size),
		beta_schedule=Args.beta_schedule,
		beta_min=Args.beta_min,
		beta_max=Args.beta_max,
		cond_dim=Args.copilot_obs_size
	)
	
	diffusion.load_ckpt(model_pt)
	diffusion_actor = DiffusionAssistedActor(-1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio)
	# ======================= End of Diffusion Actor ==================
	
	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))

	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	delta_pos_history = []
	start_sample_time = time.time()
	
	griv = 0.09
	
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		
		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints = task.detector.detect_right_joints()
		hand_detected = (joints is not None) and (len(joints) > 0)
		
		if (not recording_data) or (not hand_detected): 
			continue
		
		if time.time() - start_sample_time > dt:
			get_obs(task, cam_agent, cam_wrist)
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_ee_state is None:
				task.init_ee_state = cur_state[8].clone().cpu()

			# === X, Y, Z === 
			if hand_detected:
				# use leap_lower_bound and leap_upper_bound to cast joints['wrist'] into [-1, 1] real number
				delta_pos = (joints['right_wrist'] - task.prev_motion_joint['right_wrist']) * scaling_factor
				delta_pos = leap_axis_correction(delta_pos)
					
				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
				
				cur_rb_idx = 8
				cur_pos = cur_state[cur_rb_idx][:3]
				cur_rot_q = cur_state[cur_rb_idx][3:7]
				cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
				z_axis = cur_rot_mat[:, 2]
				
				target_pos = task.init_ee_state[:3] + delta_pos
				
				dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
				mask[cur_rb_idx, :3]  = True
				
				palm_norm   = leap_axis_correction(joints['right_palm_normal'])
				palm_norm   /= np.linalg.norm(palm_norm)    # z-axis
				
				arm_dir     = leap_axis_correction(-joints['middle_pos'][0] + joints['right_wrist'])
				arm_dir     /= np.linalg.norm(arm_dir)      # x-axis
				
				target_rot_mat = np.stack([arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1)
				target_rot_q = R.from_matrix(target_rot_mat).as_quat()
				target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)
				# print(np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3]))
				
				finger_dis = np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3])
				
				target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))
				
				orn = orientation_error(target_rot_q, cur_rot_q)
				dpose[cur_rb_idx, 3:] = orn
				mask[cur_rb_idx, 3:]  = True
				
				tcp_pos = target_pos + z_axis * 0.15
				nrm = np.linalg.norm(dpose[cur_rb_idx] - last_delta_pos) # filter human hand noise
				
				if not is_validpos(tcp_pos):
					# task.prev_motion_joint = joints
					# task.init_ee_state = cur_state[8].clone().cpu()
					# continue
					
					tcp_pos = torch.clamp(tcp_pos, tcp_lower_bound, tcp_upper_bound)
					target_pos = tcp_pos - z_axis * 0.15
					dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
					# print('clamp: ', tcp_pos)
					
				task.draw_sphere(target_pos)
				task.draw_sphere(tcp_pos, color=(0, 0, 0))
				
				if nrm < 0.003:
					dpose[cur_rb_idx] = last_delta_pos

				last_delta_pos = dpose[cur_rb_idx]
				
				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)
				next_dof_pos = env.flexiv_dof_pos + action
				
				act_dof = 0.9 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.1 * next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :env.num_flexiv_dofs] = act_dof
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

				# =============== Diffusion Robot ============
				delta_action = act_dof - env.flexiv_dof_pos
				agent_action = env.flexiv_dof_pos + delta_action 
				
				finger_dis = np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3])
				
				griv_width = get_gripper_width(gripper)
				if finger_dis > 0.07 and griv_width < 0.03:
					griv = 0.09
				elif finger_dis < 0.04 and griv_width > 0.08: # close gripper
					griv = 0
				# print(griv, finger_dis, griv_width)
				
				# griv = get_gripper_width(gripper)
				
				agent_action = torch.cat([agent_action[..., :DOF].squeeze(), torch.tensor([griv]).to(device)]).unsqueeze(0)
				
				cur_flange_pose = get_flange_pose(robot)
				
				if cur_flange_pose[1] < 1000: # TODO: this is hard-coded, you can directly remove the if statement
					cur_state = torch.squeeze(env.rigid_body_states[:, :env.num_flexiv_bodies])
					cur_rb_idx = 8 # flange
					tq = cur_state[cur_rb_idx][:7]
					tq[3:] /= torch.norm(tq[3:])
					tq[3:] = trans_quat(tq[3:])
					
					if test_on_real_robot:
						tq = torch.from_numpy(np.array(get_flange_pose(robot))).to(torch.float32)
						tq[3:] /= torch.norm(tq[3:])
					
					# batch[0][..., :DOF] = tq   # change it into current xyz + quat 
					# batch[0][..., DOF:] = griv # change it into current grivWidth
					
					agent_state = torch.cat([tq, torch.tensor([griv]).to(device)]).unsqueeze(0)
					
					agent_img, _ = cam_agent.get_data()
					agent_img = cv2.resize(agent_img, (320, 240))
					agent_img = agent_img.astype(np.float32).transpose((2, 0, 1)) / 255
					agent_img = torch.from_numpy(agent_img).unsqueeze(0).to(device)
					
					wrist_img, _ = cam_wrist.get_data()
					wrist_img = cv2.resize(wrist_img, (320, 240))
					wrist_img = wrist_img.astype(np.float32).transpose((2, 0, 1)) / 255
					wrist_img = torch.from_numpy(wrist_img).unsqueeze(0).to(device)
					
					batch = (agent_state, wrist_img, agent_img)
					
					agent_action = torch.cat([agent_action[..., :DOF].squeeze(), torch.tensor([griv]).to(device)]) # add griv dimension
					agent_action = unscale(agent_action.unsqueeze(0), lower_limits, upper_limits)
					agent_action = agent_action.repeat(1, Args.horizon).view(-1, 8 * Args.horizon) # repeat horizon times
					# agent_action = torch.zeros_like(agent_action) # TODO: autonomous agent's input

					agent_action = diffusion_actor.assisted_act(agent_action, batch).to(device).squeeze()
					agent_action = tensor_clamp(agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action))
					agent_action = scale(agent_action.view(-1, 8), lower_limits, upper_limits)
					# print("execute agent_action", agent_action)
					# print("act_dof: ", act_dof)
					# print("")
					# _ = input()
					start_sample_time = time.time() # TODO: neglect network inference time
				# ==================== End Diffusion Robot ========================
					
				# agent_action = agent_action[:4].mean(dim=0).unsqueeze(0)
				for act_next in agent_action[:6]:
					env.cur_targets[:, :DOF] = 0.9 * env.cur_targets[:, :DOF] + 0.1 * act_next[:DOF]
					# env.cur_targets[:, :DOF] = 0.9 * env.cur_targets[:, :DOF] + 0.1 * act_dof[..., :DOF] # TODO: blend with original human action
					
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

					task.step(clear_lines=False)
					handle_viewer_events(task, viewer)
					draw_bounding_box()
		
					if not recording_data: # TODO: turn on when testing on real robot
						break
					
					if time.time() - start_sample_time > dt:
						get_obs(task, cam_agent, cam_wrist)
						start_sample_time = time.time()
						
					griv_act = act_next[-1]
					# print('grivWidth: ', griv)
					
					if test_on_real_robot:
						try:
							# action = env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF]
							action = env.cur_targets[:, :DOF]
							action = action[0][:DOF].tolist()
							
							robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
							# open gripper
							if griv_act > 0.06 and get_gripper_width(gripper) < 0.03 and not is_gripper_moving(gripper):  
								gripper.move(0.09, 0.1, 20)
							# close gripper
							if griv_act < 0.075 and get_gripper_width(gripper) > 0.08 and not is_gripper_moving(gripper):
								# gripper.grasp(5) 
								gripper.move(0.0, 0.5, 10)
								
							safe_move(robot, dt)
							
						except Exception as e:
							log.error(str(e)) 
							exit(0)  
				
				# ===== sync joint state =====
				if test_on_real_robot:     
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose, please try to stay still...\033[0m")
				time.sleep(0.5)
				joints = task.detector.detect_right_joints()
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints
				griv = 0.09 if get_gripper_width(gripper) > 0.07 else 0

def flexiv_leap_diffusion_teleop_dexretarget(task: TeleopPlayer):
	OPERATOR2MANO_RIGHT = np.array(
	[
		[0, 0, 1],
		[0, 1, 0],
		[1, 0,0],
	]
)
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from leap.LeapDualJointDetector import Detector
	from .calibration.camera import CameraD400
	
	task.detector = Detector(up_axis='y',scale=0.001)
	ik  = IKController(damping=0.05)
	cam_bird = CameraD400(1)
	cam_left = CameraD400(2)
	cam_right = CameraD400(0)
	
	task.init_ee_state = None
	last_delta_pos = np.zeros(6)
	last_wrist_pos = np.zeros(3)
	last_detect_time = time.time()
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot :
		self_exam(log)
		# self_exam_leaphand(log)

		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		# global gripper

		init_joints = get_robot_joints(robot)
		DOF = 28
		Flexiv_DOF = len(init_joints)

		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 27
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	if test_on_real_hand:
		leap_hand = LeapNode()
	dt = 1.0 / frequency
	print("Sending command to robot at ",frequency," Hz")

	# ================== Diffusion Actor ====================
	from torch.utils.data import DataLoader
	from .diffusha_flexiv.diffusion.train import Trajectory
	from .diffusha_flexiv.config.default_args import Args
	from .diffusha_flexiv.diffusion.ddpm import DiffusionModel, DiffusionCore
	from .diffusha_flexiv.actor.assistive import DiffusionAssistedActor
	
	print(env.flexiv_dof_lower_limits.shape)
	device = env.device
	lower_limits = env.flexiv_dof_lower_limits[:DOF]
	upper_limits = env.flexiv_dof_upper_limits[:DOF]
	lower_limits = torch.cat([lower_limits, torch.tensor([0]).to(device)])
	upper_limits = torch.cat([upper_limits, torch.tensor([0.10]).to(device)])
	
	# dataset = Trajectory(lower_limits, upper_limits, device=device, traj_idx=4)
	# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/tao92m79/step_00006000.pt"   # hidden_dim=256, step=50
	model_pt = "/home/rhos/Desktop/flexiv/logs/diffusha-flexiv/4wrbxope/step_00006000.pt"   #horizon=12, hidden_dim=512, step=100, beta_max=1e-2
	
	diffusion = DiffusionModel(
		diffusion_core=DiffusionCore(),
		num_diffusion_steps=Args.num_diffusion_steps,
		input_size=(Args.copilot_obs_size + Args.act_size),
		beta_schedule=Args.beta_schedule,
		beta_min=Args.beta_min,
		beta_max=Args.beta_max,
		cond_dim=Args.copilot_obs_size
	)
	
	diffusion.load_ckpt(model_pt)
	diffusion_actor = DiffusionAssistedActor(-1, Args.act_size, diffusion, fwd_diff_ratio=Args.fwd_diff_ratio)
	# ======================= End of Diffusion Actor ==================
	
	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))

	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	delta_pos_history = []
	start_sample_time = time.time()
	cfg=None
	with open('/home/rhos_pub/LeapTele/DexCopilot/isaacgymenvs/isaacgymenvs/cfg/retarget/leap_hand_right_.yaml') as f:
		cfg=yaml.load(f, Loader=yaml.FullLoader) 

	retargeter=RetargetingConfig.from_dict(cfg).build()

	
	retargeting_joint_names = retargeter.joint_names
	isaac_joint_names=gym.get_actor_dof_names(env.envs[0],task.env.flexivs[0])
	isaac_joint_indices={}
	for name in retargeting_joint_names:
		if name in isaac_joint_names:
			isaac_joint_indices[retargeting_joint_names.index(name)]=isaac_joint_names.index(name)
	isaac_joints=np.array(list(isaac_joint_indices.values()))
	retargeting_joints=np.array(list(isaac_joint_indices.keys()))
	
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		
		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints = task.detector.detect_right_joints()
		hand_detected = (joints is not None) and (len(joints) > 0)
		
		if (not recording_data) or (not hand_detected): 
			continue
		
		if time.time() - start_sample_time > dt:
			get_obs(task,cam_bird=cam_bird,cam_left=cam_left,cam_right=cam_right)
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_hand_wrist_joint is None:
				state = cur_state[...,:3]
				tip_state = state[env.nail_handles]
				wrist_state = state[8]
				task.init_tip_state = tip_state.clone().cpu()
				task.init_hand_wrist_joint = wrist_state.clone().cpu()
			
			task.draw_sphere(wrist_state, color=(0, 0, 0), radius=0.02)

			# === X, Y, Z === 
			if hand_detected:
				rate = (joints['right_wrist'] - task.prev_motion_joint['right_wrist']) / (leap_upper_bound - leap_lower_bound)
				delta_pos = scale(torch.from_numpy(rate), isaacgym_lower_bound, isaacgym_upper_bound)
				delta_pos = leap_axis_correction(delta_pos)
				target_wrist_pos = delta_pos + task.init_hand_wrist_joint[:3]
				target_wrist_pos=np.clip(target_wrist_pos,tcp_lower_bound,tcp_upper_bound)

				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)

				dpose[8, :3] = target_wrist_pos.to(env.device) - wrist_state.to(env.device)
				mask[8, :3] = True
				task.draw_sphere(target_wrist_pos.cpu(), color=(0, 0, 0))

				cur_rot_q = cur_state[8][3:7]
				cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
				
				palm_norm=-leap_axis_correction(joints['right_palm_normal'])
				palm_norm   /= np.linalg.norm(palm_norm)    # z-axis
				
				arm_dir     = leap_axis_correction(joints['middle_pos'][0] - joints['right_wrist'] )
				arm_dir     /= np.linalg.norm(arm_dir)      # x-axis
				
				target_rot_mat = np.stack([arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1)
				target_rot_q = R.from_matrix(target_rot_mat).as_quat()
				target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)

				target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))      
				
				orn = orientation_error(target_rot_q, cur_rot_q)
				dpose[cur_rb_idx, 3:] = orn
				mask[cur_rb_idx, 3:]  = True
				
				transformed_keypoints=np.stack([np.array(leap_axis_correction(joints['right_wrist'])),np.array(leap_axis_correction(joints['thumb_pos'][-1])),np.array(leap_axis_correction(joints['index_pos'][-1])),np.array(leap_axis_correction(joints['middle_pos'][-1])),np.array(leap_axis_correction(joints['ring_pos'][-1]))])
				transformed_keypoints-=transformed_keypoints[0]


				transform_mat= [arm_dir,np.cross(arm_dir, -palm_norm), -palm_norm]

				
				transformed_keypoints=transformed_keypoints@np.linalg.solve(transform_mat, np.eye(3))@OPERATOR2MANO_RIGHT

				indices=(retargeter.optimizer.target_link_human_indices/4).astype(int)
				ref_pos=[]
				ref_pos=transformed_keypoints[indices[1,:]]-transformed_keypoints[indices[0,:]]

				ref_pos=np.stack(ref_pos)

				qpos=retargeter.retarget(ref_pos)
				
				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)
				next_dof_pos = env.flexiv_dof_pos + action

				next_dof_pos[:,isaac_joints]=torch.tensor(qpos[retargeting_joints],dtype=torch.float)
				
				act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05* next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)

				env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

				# =============== Diffusion Robot ============
				delta_action = act_dof - env.flexiv_dof_pos
				agent_action = env.flexiv_dof_pos + delta_action 
				
				
				agent_action = torch.cat([agent_action[..., :DOF].squeeze()]).unsqueeze(0)
				
				cur_flange_pose = get_flange_pose(robot)
				
				if cur_flange_pose[1] < 1000: # TODO: this is hard-coded, you can directly remove the if statement
					cur_state = torch.squeeze(env.rigid_body_states[:, :env.num_flexiv_bodies])
					cur_rb_idx = 8 # flange
					tq = cur_state[cur_rb_idx][:7]
					tq[3:] /= torch.norm(tq[3:])
					tq[3:] = trans_quat(tq[3:])
					
					if test_on_real_robot:
						tq = torch.from_numpy(np.array(get_flange_pose(robot))).to(torch.float32)
						tq[3:] /= torch.norm(tq[3:])
					
					# batch[0][..., :DOF] = tq   # change it into current xyz + quat 
					# batch[0][..., DOF:] = griv # change it into current grivWidth
					
					agent_state = torch.cat([tq, torch.tensor([griv]).to(device)]).unsqueeze(0)
					
					bird_img, _ = cam_bird.get_data()
					bird_img = cv2.resize(bird_img, (320, 240))
					bird_img = bird_img.astype(np.float32).transpose((2, 0, 1)) / 255
					bird_img = torch.from_numpy(bird_img).unsqueeze(0).to(device)

					left_img, _ = cam_left.get_data()
					left_img = cv2.resize(left_img, (320, 240))
					left_img = left_img.astype(np.float32).transpose((2, 0, 1)) / 255
					left_img = torch.from_numpy(left_img).unsqueeze(0).to(device)

					right_img, _ = cam_right.get_data()
					right_img = cv2.resize(right_img, (320, 240))
					right_img = right_img.astype(np.float32).transpose((2, 0, 1)) / 255
					right_img = torch.from_numpy(right_img).unsqueeze(0).to(device)
					
					batch = (agent_state, bird_img, left_img, right_img)
					
					agent_action = torch.cat([agent_action[..., :DOF].squeeze()]) # add griv dimension
					agent_action = unscale(agent_action.unsqueeze(0), lower_limits, upper_limits)
					agent_action = agent_action.repeat(1, Args.horizon).view(-1, 8 * Args.horizon) # repeat horizon times
					# agent_action = torch.zeros_like(agent_action) # TODO: autonomous agent's input

					agent_action = diffusion_actor.assisted_act(agent_action, batch).to(device).squeeze()
					agent_action = tensor_clamp(agent_action, -torch.ones_like(agent_action), torch.ones_like(agent_action))
					agent_action = scale(agent_action.view(-1, 8), lower_limits, upper_limits)
					# print("execute agent_action", agent_action)
					# print("act_dof: ", act_dof)
					# print("")
					# _ = input()
					start_sample_time = time.time() # TODO: neglect network inference time
				# ==================== End Diffusion Robot ========================
					
				# agent_action = agent_action[:4].mean(dim=0).unsqueeze(0)
				for act_next in agent_action[:6]:
					env.cur_targets[:, :DOF] = 0.9 * env.cur_targets[:, :DOF] + 0.1 * act_next[:DOF]
					# env.cur_targets[:, :DOF] = 0.9 * env.cur_targets[:, :DOF] + 0.1 * act_dof[..., :DOF] # TODO: blend with original human action
					
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

					task.step(clear_lines=False)
					handle_viewer_events(task, viewer)
					draw_bounding_box()
		
					if not recording_data: # TODO: turn on when testing on real robot
						break
					
					if time.time() - start_sample_time > dt:
						get_obs(task,cam_bird=cam_bird,cam_left=cam_left,cam_right=cam_right)
						start_sample_time = time.time()
						
					# print('grivWidth: ', griv)
					
					if test_on_real_robot:
						try:
							# action = env.cur_targets[:, env.num_flexiv_dofs:env.num_flexiv_dofs+DOF]
							flexiv_targets = env.cur_targets[..., :Flexiv_DOF]
							action = flexiv_targets[0][:Flexiv_DOF].tolist()
							
							robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
						   
							safe_move(robot, dt)
							
						except Exception as e:
							log.error(str(e)) 
							exit(0)  
					
					if test_on_real_hand:
						try:
							action = env.cur_targets[0].tolist()
							set_leap_hand_pos(leap_hand,np.asarray(action),env.nail_handles)
						except Exception as e:
							log.error(str(e))
				
				
				# ===== sync joint state =====
				if test_on_real_robot:     
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose, please try to stay still...\033[0m")
				time.sleep(0.5)
				joints = task.detector.detect_right_joints()
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints
				griv = 0.09 if get_gripper_width(gripper) > 0.07 else 0

def sim_move_gripper(env, width: float):
	
	env.cur_targets[:, 7] = width
	"""
	8: left_inner_knuckle_joint
	9: left_outer_knuckle_joint
	10: left_inner_finger_joint
	11: right_inner_knuckle_joint
	12: right_outer_knuckle_joint
	13: right_inner_finger_joint
	""" 
	
	env.cur_targets[:, 9] = 9.404 * width -0.155
	env.cur_targets[:, 8] = env.cur_targets[:, 9]
	env.cur_targets[:, 10] = -env.cur_targets[:, 9]
	
	env.cur_targets[:, 12] = env.cur_targets[:, 9]
	env.cur_targets[:, 11] = env.cur_targets[:, 12]
	env.cur_targets[:, 13] = -env.cur_targets[:, 12]

def leap_teleop(task: TeleopPlayer):
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from leap.LeapDualJointDetector import Detector
	from .calibration.camera import CameraD400
	
	task.detector = Detector(up_axis='y',scale=0.001)
	ik  = IKController(damping=0.05)
	cam_agent = CameraD400(0)
	cam_wrist = CameraD400(1)
	
	task.init_ee_state = None
	last_delta_pos = np.zeros(6)
	last_wrist_pos = np.zeros(3)
	last_detect_time = time.time()
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		try:
			gripper.move(0.09, 0.1, 20)
		except Exception as e:
			log.error(str(e))
			
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval",
	)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))

	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	delta_pos_history = []
	start_sample_time = time.time()
	sim_gripper_width = 0
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		
		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints = task.detector.detect_right_joints()
		hand_detected = (joints is not None) and (len(joints) > 0)
		
		# if (not recording_data) or (not hand_detected): 
		#     continue
		
		if time.time() - start_sample_time > dt:
			get_obs(task,cam_wrist=cam_wrist,cam_agent=cam_agent)
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_ee_state is None:
				task.init_ee_state = cur_state[8].clone().cpu()

			# === X, Y, Z === 
			if hand_detected:
				# use leap_lower_bound and leap_upper_bound to cast joints['wrist'] into [-1, 1] real number
				delta_pos = (joints['right_wrist'] - task.prev_motion_joint['right_wrist']) * scaling_factor
				delta_pos = leap_axis_correction(delta_pos)
				
				# ==== avoid jitter ====
				# delta_pos_history.append(delta_pos)
				# delta_pos_history = delta_pos_history[-20:]
				# dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
				# if dis <= 0.02:
				#     delta_pos_history = [delta_pos_history[0]] * 20
				#     delta_pos = delta_pos_history[-1]
				# ======================
					
				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
				
				cur_rb_idx = 8
				cur_pos = cur_state[cur_rb_idx][:3]
				cur_rot_q = cur_state[cur_rb_idx][3:7]
				cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
				z_axis = cur_rot_mat[:, 2]
				
				target_pos = task.init_ee_state[:3] + delta_pos
				
				dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
				mask[cur_rb_idx, :3]  = True
				
				palm_norm   = leap_axis_correction(joints['right_palm_normal'])
				palm_norm   /= np.linalg.norm(palm_norm)    # z-axis
				
				arm_dir     = leap_axis_correction(-joints['middle_pos'][0] + joints['right_wrist'])
				arm_dir     /= np.linalg.norm(arm_dir)      # x-axis
				
				target_rot_mat = np.stack([arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1)
				target_rot_q = R.from_matrix(target_rot_mat).as_quat()
				target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)
				# print(np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3]))
				
				finger_dis = np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3])
				
				target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))
				
				orn = orientation_error(target_rot_q, cur_rot_q)
				dpose[cur_rb_idx, 3:] = orn
				mask[cur_rb_idx, 3:]  = True
				
				tcp_pos = target_pos + z_axis * 0.15
				nrm = np.linalg.norm(dpose[cur_rb_idx] - last_delta_pos) # filter human hand noise
				
				# if time.time() - last_detect_time > 1:
				#     task.prev_motion_joint = joints
				#     task.init_ee_state = cur_state[8].clone().cpu()
				#     last_detect_time = time.time()
					
				# if not is_validpos(tcp_pos):
				#     task.prev_motion_joint = joints
				#     task.init_ee_state = cur_state[8].clone().cpu()
				#     continue
				
				if not is_validpos(tcp_pos):
					tcp_pos = torch.clamp(tcp_pos, tcp_lower_bound, tcp_upper_bound)
					target_pos = tcp_pos - z_axis * 0.15
					dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
				
				task.draw_sphere(target_pos)
				task.draw_sphere(tcp_pos, color=(0, 0, 0))
				
				if nrm < 0.003:
					dpose[cur_rb_idx] = last_delta_pos

				last_delta_pos = dpose[cur_rb_idx]
				
				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)
				next_dof_pos = env.flexiv_dof_pos + action
				
				act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				
				if finger_dis > 0.07 and env.cur_targets[:, DOF] < 0.03:
					sim_gripper_width = 0.09
				elif finger_dis < 0.04 and env.cur_targets[:, DOF] > 0.08:
					sim_gripper_width = 0
				
				print("@@@ ", env.cur_targets[:, :14])
				sim_move_gripper(env, sim_gripper_width)
				print(sim_gripper_width, env.cur_targets[:, :14])
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				if test_on_real_robot:
					try:
						action = act_dof[0][:DOF].tolist()
						robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
						
						target_width=np.clip(finger_dis,0.09,0)
						gripper.move(target_width, 0.1, 20)



						# open gripper
						# if finger_dis > 0.07 and get_gripper_width(gripper) < 0.05 and not is_gripper_moving(gripper):  
						# 	gripper.move(0.09, 0.1, 20)
						# close gripper
						# if finger_dis < 0.03 and get_gripper_width(gripper) > 0.06 and not is_gripper_moving(gripper):
						# 	gripper.grasp(20)
							
					except Exception as e:
						log.error(str(e))
				
				# ===== sync joint state =====
				if test_on_real_robot:
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))   
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose, please try to stay still...\033[0m")
				time.sleep(1)
				joints = task.detector.detect_right_joints()
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints

def leap_umi_gopro_teleop(task):
	import pathlib
	
	def handle_umi_viewer_events(task, viewer):
		env = task.env
		gym = env.gym
		for evt in gym.query_viewer_action_events(viewer):
			# Reset the DOF states and actor root state to their initial states
			global recording_data
			
			if (evt.action == "reset") and evt.value > 0:
				print("Reset Env: Home()")
				reset_home(task)    
				recording_data               = False
				task.initialized             = False
				task.init_ee_state           = None
				task.prev_motion_joint       = None
				task.obs_dict                = []
			
			elif (evt.action == "flexiv_data_collection") and evt.value > 0:
				if recording_data:
					recording_data = False
					print("End recording data")
				else:
					recording_data = True
					reset_home(task)
					task.initialized             = False
					task.prev_motion_joint       = None
					task.obs_dict                = []
					cam_agent_view.clear()
					cam_wrist_view.clear()
					cam_bird_view.clear()
					cam_left_view.clear()
					cam_right_view.clear()
					print("Start recording data")
			
			elif (evt.action == "space_shoot") and evt.value > 0:
				if save_realsense:
					dirname = f"/home/rhos/tactile_umi_grav/data/teleop/{task.timestamp}"
					os.makedirs(dirname, exist_ok=True)
					
					cnt = len(glob.glob(os.path.join(dirname, "*/")))
					dirname = os.path.join(dirname, f"{cnt:03}")
					os.makedirs(dirname, exist_ok=True)
					for f in os.listdir(dirname):
						os.remove(os.path.join(dirname, f))
						
					filename = os.path.join(dirname, "data.pkl")
					with open(filename, 'wb') as f:
						pkl.dump(task.obs_dict, f)
					
					# save wrist view
					assert len(cam_wrist_view)>0
					with h5py.File(os.path.join(dirname, "wrist_img.hdf5"), 'w') as hf:
						images = np.array([view["image"] for view in cam_wrist_view])
						timestamp = np.array([view["depth"] for view in cam_wrist_view]) # xuyue: hacked!
						hf.create_dataset("images", data=images)
						hf.create_dataset("timestamp", data=depths)

					print("GoPro Camera hdf5 Saved!")

	class GoPro(object):
		def __init__(self, camera_id=0):
			v4l_path = self.get_v4l_path()
			self.cap = cv2.VideoCapture(v4l_path, cv2.CAP_V4L2)
			if not self.cap.isOpened():
				print('Error: Cannot open camera')
				exit(1)
			
			w, h = (1920, 1080)
			fps = 60
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
			self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
			# set fps
			self.cap.set(cv2.CAP_PROP_FPS, fps)

			print('cam init done.')

		def __del__(self):
			self.cap.release()

		def get_v4l_path(self, prefix="usb-MACROSILICON"):
			dirname = 'by-id'
			v4l_dir = pathlib.Path('/dev/v4l').joinpath(dirname)
			valid_paths = list()
			for dev_path in sorted(v4l_dir.glob("*video*")):
				name = dev_path.name
				# only keep devices ends with "index0"
				index_str = name.split('-')[-1]
				assert index_str.startswith('index')
				index = int(index_str[5:])
				if index == 0 and prefix in str(dev_path.absolute()):
					valid_paths.append(str(dev_path.absolute()))
			assert len(valid_paths) == 1, str(valid_paths)
			return valid_paths[0]

		def get_data(self, hole_filling=False):
			"""image, timestamp"""
			ret, frame = False, None
			while not ret:
				ret, frame = self.cap.read(frame)
			t_recv = time.time()
			return frame, np.array([t_recv])

	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from leap.LeapDualJointDetector import Detector
	from .calibration.camera import CameraD400
	
	task.detector = Detector(up_axis='y',scale=0.001)
	ik  = IKController(damping=0.05)
	cam_wrist = GoPro()
	
	task.init_ee_state = None
	last_delta_pos = np.zeros(6)
	last_wrist_pos = np.zeros(3)
	last_detect_time = time.time()
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		try:
			gripper.move(0.09, 0.1, 20)
		except Exception as e:
			log.error(str(e))
			
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print("Sending command to robot at", frequency, "Hz, or", dt, "seconds interval",)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))
	

	delta_pos_history = []
	start_sample_time = time.time()
	sim_gripper_width = 0
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		
		handle_viewer_events(task, viewer)
		draw_bounding_box_global(task, tcp_lower_bound, tcp_lower_bound)
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints = task.detector.detect_right_joints()
		hand_detected = (joints is not None) and (len(joints) > 0)
		
		# if (not recording_data) or (not hand_detected): 
		#     continue
		
		if time.time() - start_sample_time > dt:
			get_obs(task, cam_wrist=cam_wrist)
			cv2.imshow("GoPro", cam_wrist_view[-1]["image"])
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_ee_state is None:
				task.init_ee_state = cur_state[8].clone().cpu()

			# === X, Y, Z === 
			if hand_detected:
				# use leap_lower_bound and leap_upper_bound to cast joints['wrist'] into [-1, 1] real number
				delta_pos = (joints['right_wrist'] - task.prev_motion_joint['right_wrist']) * scaling_factor
				delta_pos = leap_axis_correction(delta_pos)
				
				# ==== avoid jitter ====
				# delta_pos_history.append(delta_pos)
				# delta_pos_history = delta_pos_history[-20:]
				# dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
				# if dis <= 0.02:
				#     delta_pos_history = [delta_pos_history[0]] * 20
				#     delta_pos = delta_pos_history[-1]
				# ======================
					
				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
				
				cur_rb_idx = 8
				cur_pos = cur_state[cur_rb_idx][:3]
				cur_rot_q = cur_state[cur_rb_idx][3:7]
				cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
				z_axis = cur_rot_mat[:, 2]
				
				target_pos = task.init_ee_state[:3] + delta_pos
				
				dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
				mask[cur_rb_idx, :3]  = True
				
				palm_norm   = leap_axis_correction(joints['right_palm_normal'])
				palm_norm   /= np.linalg.norm(palm_norm)    # z-axis
				
				arm_dir     = leap_axis_correction(-joints['middle_pos'][0] + joints['right_wrist'])
				arm_dir     /= np.linalg.norm(arm_dir)      # x-axis
				
				target_rot_mat = np.stack([arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1)
				target_rot_q = R.from_matrix(target_rot_mat).as_quat()
				target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)
				# print(np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3]))
				
				finger_dis = np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3])
				
				target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))
				
				orn = orientation_error(target_rot_q, cur_rot_q)
				dpose[cur_rb_idx, 3:] = orn
				mask[cur_rb_idx, 3:]  = True
				
				tcp_pos = target_pos + z_axis * 0.15
				nrm = np.linalg.norm(dpose[cur_rb_idx] - last_delta_pos) # filter human hand noise
				
				# if time.time() - last_detect_time > 1:
				#     task.prev_motion_joint = joints
				#     task.init_ee_state = cur_state[8].clone().cpu()
				#     last_detect_time = time.time()
					
				# if not is_validpos(tcp_pos):
				#     task.prev_motion_joint = joints
				#     task.init_ee_state = cur_state[8].clone().cpu()
				#     continue
				
				if not is_validpos(tcp_pos):
					tcp_pos = torch.clamp(tcp_pos, tcp_lower_bound, tcp_upper_bound)
					target_pos = tcp_pos - z_axis * 0.15
					dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
				
				task.draw_sphere(target_pos)
				task.draw_sphere(tcp_pos, color=(0, 0, 0))
				
				if nrm < 0.003:
					dpose[cur_rb_idx] = last_delta_pos

				last_delta_pos = dpose[cur_rb_idx]
				
				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)
				next_dof_pos = env.flexiv_dof_pos + action
				
				act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				
				if finger_dis > 0.04: # and env.cur_targets[:, DOF] < 0.03:
					sim_gripper_width = 0.10
				elif finger_dis < 0.04: # and env.cur_targets[:, DOF] > 0.08:
					sim_gripper_width = 0.03
				
				print("@@@ ", env.cur_targets[:, :14])
				sim_move_gripper(env, sim_gripper_width)
				print(sim_gripper_width, env.cur_targets[:, :14])
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				if test_on_real_robot:
					try:
						action = act_dof[0][:DOF].tolist()
						robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
						
						# target_width=np.clip(sim_gripper_width, 0.09, 0)
						gripper.move(sim_gripper_width, 0.1, 20)


						# open gripper
						# if finger_dis > 0.07 and get_gripper_width(gripper) < 0.05 and not is_gripper_moving(gripper):  
						# 	gripper.move(0.09, 0.1, 20)
						# close gripper
						# if finger_dis < 0.03 and get_gripper_width(gripper) > 0.06 and not is_gripper_moving(gripper):
						# 	gripper.grasp(20)
							
					except Exception as e:
						log.error(str(e))
				
				# ===== sync joint state =====
				if test_on_real_robot:
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))   
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose, please try to stay still...\033[0m")
				time.sleep(1)
				joints = task.detector.detect_right_joints()
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints
			
def flexiv_leap_teleop(task: TeleopPlayer):
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from leap.LeapDualJointDetector import Detector
	from .calibration.camera import CameraD400

	sys.path.append('/home/rhos_pub/LeapTele/python/')
	from leaphand import LeapNode

	task.detector = Detector(up_axis='y',scale=0.001)
	ik  = IKController(damping=0.05)
	# cam_agent = CameraD400(0)
	# cam_wrist = CameraD400(1)
	
	task.init_ee_state = None
	last_delta_pos = np.zeros(6)
	last_wrist_pos = np.zeros(3)
	last_detect_time = time.time()
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1

	if test_on_real_robot :
		self_exam(log)
		# self_exam_leaphand(log)

		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		# global gripper

		init_joints = get_robot_joints(robot)
		DOF = 28
		Flexiv_DOF = len(init_joints)

		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 27
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	if test_on_real_hand:
		leap_hand = LeapNode()
	dt = 1.0 / frequency
	print("Sending command to robot at ",frequency," Hz")

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))

	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	delta_pos_history = []
	start_sample_time = time.time()
	sim_gripper_width = 0
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		
		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints = task.detector.detect_right_joints()
		hand_detected = (joints is not None) and (len(joints) > 0)
		
		# if (not recording_data) or (not hand_detected): 
		#     continue
		
		if time.time() - start_sample_time > dt:
			get_obs(task)
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		

		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies,:3])
			if task.init_hand_wrist_joint is None:
				state = cur_state
				tip_state = state[env.nail_handles]
				wrist_state = state[8]
				task.init_tip_state = tip_state.clone().cpu()
				task.init_hand_wrist_joint = wrist_state.clone().cpu()
			task.draw_sphere(wrist_state, color=(0, 0, 0), radius=0.02)

			# # === X, Y, Z ===
			if hand_detected:
				rate = (joints['right_wrist'] - task.prev_motion_joint['right_wrist']) / (leap_upper_bound - leap_lower_bound)
				delta_pos = scale(torch.from_numpy(rate), isaacgym_lower_bound, isaacgym_upper_bound)
				delta_pos = leap_axis_correction(delta_pos)
				target_wrist_pos = delta_pos + task.init_hand_wrist_joint[:3]


				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
				# print(target_wrist_pos.shape,wrist_state.shape)

				dpose[8, :3] = target_wrist_pos.to(env.device) - wrist_state.to(env.device)
				mask[8, :3] = True
				task.draw_sphere(target_wrist_pos.cpu(), color=(0, 0, 0))

				for i, key in enumerate(['index','thumb','middle','ring']):     
					# for i, key in enumerate(['thumb']):   
					k = key + '_pos'
					cur_rb_tip_idx           = env.nail_handles[i] #nail_handles:[10,14,18,22]
					cur_pos                  = cur_state[cur_rb_tip_idx]
					real_finger_len = np.linalg.norm(joints[k][0] - joints[k][1]) + np.linalg.norm(joints[k][2] - joints[k][1]) \
											+ np.linalg.norm(joints[k][3] - joints[k][2]) + np.linalg.norm(joints[k][0] - joints['right_wrist'])
					# leap_hand_finger_len = 0.298 + 0.295
					leap_hand_finger_len = 0.23
					if key == 'ring' :
						leap_hand_finger_len *= 1.05
					if key == 'index':
						leap_hand_finger_len *= 1.1

					if key == 'thumb':
						leap_hand_finger_len *= 0.8
					# leap_hand_finger_len /= 2.0
					target_pos = target_wrist_pos + leap_axis_correction(joints[k][-1] - joints['right_wrist']) * leap_hand_finger_len / real_finger_len
					# target_pos = target_wrist_pos + (joints[k][-1] - joints['right_wrist'])
					dpose[cur_rb_tip_idx, :3] = (target_pos.to(env.device)) - cur_pos.to(env.device)
					mask[cur_rb_tip_idx, :3]  = True 
					
					colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
					task.draw_sphere(target_pos, color=colors[i])

				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)
				next_dof_pos = env.flexiv_dof_pos + action

				act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)

				# env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				if test_on_real_robot:
					try:
						flexiv_targets = act_dof[..., :Flexiv_DOF]
						action = flexiv_targets[0][:Flexiv_DOF].tolist()
						robot.sendJointPosition(action, target_vel[:Flexiv_DOF], target_acc[:Flexiv_DOF], MAX_VEL[:Flexiv_DOF], MAX_ACC[:Flexiv_DOF])
						# print('action',np.asarray(act_dof).shape)
					except Exception as e:
						log.error(str(e))
				if test_on_real_hand:
					try:
						action = act_dof[0].tolist()
						set_leap_hand_pos(leap_hand,np.asarray(action),env.nail_handles)
					except Exception as e:
						log.error(str(e))
				# ===== sync joint state =====
				#if test_on_real_robot:
					#joints = get_robot_joints(robot)
					#env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
					#gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose\033[0m")
				time.sleep(1)
				joints = task.detector.detect_right_joints()
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints

def flexiv_leap_teleop_dexretarget(task: TeleopPlayer):
	print(task.env.flexiv_dof_upper_limits)
	print(task.env.flexiv_dof_lower_limits)
	OPERATOR2MANO_RIGHT = np.array(
	[
		[0, 0, 1],
		[0, 1, 0],
		[1, 0,0],
	]
)
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from leap.LeapDualJointDetector import Detector
	from .calibration.camera import CameraD400

	sys.path.append('/home/rhos_pub/LeapTele/python/')
	from leaphand import LeapNode

	task.detector = Detector(up_axis='y',scale=0.001)
	ik  = IKController(damping=0.05)
	cam_bird = CameraD400(1)
	# cam_left = CameraD400(2)
	cam_left=None
	cam_right = CameraD400(0)
	

	task.init_ee_state = None
	last_delta_pos = np.zeros(6)
	last_wrist_pos = np.zeros(3)
	last_detect_time = time.time()
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1

	if test_on_real_robot :
		self_exam(log)
		# self_exam_leaphand(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		# global gripper

		init_joints = get_robot_joints(robot)
		DOF = 28
		Flexiv_DOF = len(init_joints)

		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 27
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	if test_on_real_hand:
		global leap_hand
		leap_hand = LeapNode()
		set_leap_hand_pos(leap_hand,np.zeros(28),env.nail_handles)

	dt = 1.0 / frequency
	print("Sending command to robot at ",frequency," Hz")

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))

	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	delta_pos_history = []
	start_sample_time = time.time()
	cfg=None
	with open('/home/rhos_pub/LeapTele/DexCopilot/isaacgymenvs/isaacgymenvs/cfg/retarget/leap_hand_right_.yaml') as f:
		cfg=yaml.load(f, Loader=yaml.FullLoader) 

	retargeter=RetargetingConfig.from_dict(cfg).build()

	
	retargeting_joint_names = retargeter.joint_names
	isaac_joint_names=gym.get_actor_dof_names(env.envs[0],task.env.flexivs[0])
	isaac_joint_indices={}
	for name in retargeting_joint_names:
		if name in isaac_joint_names:
			isaac_joint_indices[retargeting_joint_names.index(name)]=isaac_joint_names.index(name)
	isaac_joints=np.array(list(isaac_joint_indices.values()))
	retargeting_joints=np.array(list(isaac_joint_indices.keys()))


	while not env.gym.query_viewer_has_closed(env.viewer):    
		begin_time=time.time()

		task.step()
						
		handle_viewer_events(task, viewer)
		draw_bounding_box()
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		
		joints = task.detector.detect_right_joints()
		hand_detected = (joints is not None) and (len(joints) > 0)
		
		# if (not recording_data) or (not hand_detected): 
		#     continue



		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		

		if task.initialized:

			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_hand_wrist_joint is None:
				state = cur_state[...,:3]
				tip_state = state[env.nail_handles]
				wrist_state = state[8]
				task.init_tip_state = tip_state.clone().cpu()
				task.init_hand_wrist_joint = wrist_state.clone().cpu()
			
			task.draw_sphere(wrist_state, color=(0, 0, 0), radius=0.02)
			
			# # === X, Y, Z ===
			if hand_detected:
				rate = (joints['right_wrist'] - task.prev_motion_joint['right_wrist']) / (leap_upper_bound - leap_lower_bound)
				delta_pos = scale(torch.from_numpy(rate), isaacgym_lower_bound, isaacgym_upper_bound)
				delta_pos = leap_axis_correction(delta_pos)
				target_wrist_pos = delta_pos + task.init_hand_wrist_joint[:3]
				target_wrist_pos=np.clip(target_wrist_pos,tcp_lower_bound,tcp_upper_bound)

				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
				# print(target_wrist_pos.shape,wrist_state.shape)

				dpose[8, :3] = target_wrist_pos.to(env.device) - wrist_state.to(env.device)
				mask[8, :3] = True
				task.draw_sphere(target_wrist_pos.cpu(), color=(0, 0, 0))

				cur_rot_q = cur_state[8][3:7]
				cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
				
				palm_norm=-leap_axis_correction(joints['right_palm_normal'])
				palm_norm   /= np.linalg.norm(palm_norm)    # z-axis
				
				arm_dir     = leap_axis_correction(joints['middle_pos'][0] - joints['right_wrist'] )
				arm_dir     /= np.linalg.norm(arm_dir)      # x-axis
				
				target_rot_mat = np.stack([arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1)
				target_rot_q = R.from_matrix(target_rot_mat).as_quat()
				target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)

				target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))

				orn = orientation_error(target_rot_q, cur_rot_q)
				dpose[8, 3:] = orn
				mask[8, 3:]  = True

				transformed_keypoints=np.stack([np.array(leap_axis_correction(joints['right_wrist'])),np.array(leap_axis_correction(joints['thumb_pos'][-1])),np.array(leap_axis_correction(joints['index_pos'][-1])),np.array(leap_axis_correction(joints['middle_pos'][-1])),np.array(leap_axis_correction(joints['ring_pos'][-1]))])
				transformed_keypoints-=transformed_keypoints[0]


				transform_mat= [arm_dir,np.cross(arm_dir, -palm_norm), -palm_norm]

				
				start_time=time.time()
				transformed_keypoints=transformed_keypoints@np.linalg.solve(transform_mat, np.eye(3))@OPERATOR2MANO_RIGHT


				indices=(retargeter.optimizer.target_link_human_indices/4).astype(int)
				ref_pos=[]
				ref_pos=transformed_keypoints[indices[1,:]]-transformed_keypoints[indices[0,:]]

				#for i, key in enumerate(['thumb','index','middle','ring']): 
					#ref_pos.append(np.array(transformed_keypoints[key][-1])@OPERATOR2MANO_RIGHT)
				ref_pos=np.stack(ref_pos)
				
				qpos=retargeter.retarget(ref_pos)
				print(time.time()-start_time)
				# print('qpos:',qpos)

				# ===== sync joint state =====
				if test_on_real_robot:
					joints = get_robot_joints(robot)
					hand_joints = leap_hand.read_pos()-3.14

					set_pose=np.zeros(28)
					for i, key in enumerate([12,22,27]):
						tip = key
						tip = tip-1
						finger_tip = tip-1
						dip = tip - 2
						pip = tip - 3

						set_pose[dip] = hand_joints[i*4]
						set_pose[pip] = hand_joints[i*4+1] 
						set_pose[finger_tip] = hand_joints[i*4+2] 
						set_pose[tip] = hand_joints[i*4+3] 

					tip = 16
					finger_tip = tip - 1
					dip = tip - 2
					pip = tip - 3
					set_pose[pip] = hand_joints[12]
					set_pose[dip] = hand_joints[13]
					set_pose[finger_tip] = hand_joints[14]*-1
					set_pose[tip] = hand_joints[15]*-1

					env.cur_targets[:,:]=torch.tensor(set_pose)
					#print('cur:',env.cur_targets[:,isaac_joints],hand_joints/180*np.pi)
					env.cur_targets[:, :7] = torch.from_numpy(np.array(joints)).to(device)
					#env.cur_targets[:,isaac_joints] = torch.from_numpy(np.array(hand_joints)[retargeting_joints]).to(device)
					
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					task.step()

				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)
				next_dof_pos = env.flexiv_dof_pos + action

				next_dof_pos[:,isaac_joints]=torch.tensor(qpos[retargeting_joints],dtype=torch.float)


				act_dof = 0.9 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.1 * next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)

				env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

				if test_on_real_robot:
					try:
						flexiv_targets = act_dof[..., :Flexiv_DOF]
						action = flexiv_targets[0][:Flexiv_DOF].tolist()
						robot.sendJointPosition(action, target_vel[:Flexiv_DOF], target_acc[:Flexiv_DOF], MAX_VEL[:Flexiv_DOF], MAX_ACC[:Flexiv_DOF])
						# print('action',np.asarray(act_dof).shape)
					except Exception as e:
						log.error(str(e))
				
				get_obs(task,leap_hand,cam_bird=cam_bird,cam_left=cam_left,cam_right=cam_right)


				if test_on_real_hand:
					try:
						action = act_dof[0].tolist()
						set_leap_hand_pos(leap_hand,np.asarray(action),env.nail_handles)
					except Exception as e:
						log.error(str(e))
				
				

				print('loop:',time.time()-begin_time)

		if not task.initialized:
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose\033[0m")
				time.sleep(1)
				joints = task.detector.detect_right_joints()
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints  
def vr_axis_correction(vec):
	""" -yzx coordinate -> xyz coordinate"""
	vec = vec[..., [2, 0, 1]]
	vec[..., 1] = -vec[..., 1]
	return vec

###main调用函数
#新增修改 边界速度缩放
def scale_velocity_near_boundary(joint_positions, tcp_pos, 
                                  target_vel, target_acc, 
                                  MAX_VEL, MAX_ACC, 
                                  margin=0.1, min_scale=0.3):
	"""
	若 TCP 位置靠近边界，则缩放速度/加速度。
	:param tcp_pos: 当前 TCP 空间位置 (x, y, z)
	:param margin: 与边界小于该值时触发缩放
	:param min_scale: 最小缩放比例（越接近边界越小）
	"""
	scale = 1.0
	# if tcp_pos[2]<0.2 or tcp_pos[2]>0.6:
	# 	scale = 0.3
	# for i in range(3):  # x, y, z
	dist_to_low = tcp_pos[2] - 0.1
	dist_to_up = 1.0 - tcp_pos[2]
	min_dist = min(dist_to_low, dist_to_up)

	if min_dist < margin:
		scale = min(scale, max(min_scale, min_dist / margin))

	# 缩放速度和加速度
	# import pdb; pdb.set_trace()
	print("MAX_VEL, MAX_ACC", MAX_VEL, MAX_ACC)
	# target_vel = [v * scale for v in target_vel]
	# target_acc = [a * scale for a in target_acc]
	MAX_VEL = [v * scale for v in MAX_VEL]
	MAX_ACC = [a * scale for a in MAX_ACC]

	return MAX_VEL, MAX_ACC


def vr_gripper_teleop(task: TeleopPlayer):
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from .vr_detector import AllegroHandDetector
	from .calibration.camera import CameraD400
	from .oculus_streamer import OculusStreamer

	task.detector = AllegroHandDetector()
	streamer= OculusStreamer()
	ik  = IKController(damping=0.05)
	# cam_agent = CameraD400(0)
	# cam_wrist = CameraD400(1)
	
	task.init_ee_state = None
	last_delta_pos = np.zeros(6) ##变化量
	last_wrist_pos = np.zeros(3) ##绝对值
	last_detect_time = time.time()
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1

	# initialization 、、、team14修改3
	task.resistance_factor = 0 ###可修改参数调整阻抗系数
	#####
	
	# import pdb; pdb.set_trace()
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		import pdb; pdb.set_trace()
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		try:
			gripper.move(0.09, 0.1, 20)
		except Exception as e:
			log.error(str(e))
			
		init_joints = get_robot_joints(robot)

		###fff
		fixed_joints = init_joints

		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval"
	)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))

	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	delta_pos_history = []
	start_sample_time = time.time()
	sim_gripper_width = 0
	###主循环
	flag = True

	# print(env.gym.query_viewer_has_closed(env.viewer))
	while not env.gym.query_viewer_has_closed(env.viewer): 
		# print("in the loop")
		task.step()
		
		color_image = None #TODO: replace with task ...
		while color_image is None:
			color_image =env.gym.get_camera_image_gpu_tensor(sim,env.envs[0], env.camera_handle,gymapi.IMAGE_COLOR)
			color_image=gymtorch.wrap_tensor(color_image)
			color_image=color_image.cpu().numpy()
			color_image=color_image[:,:,[2,1,0]]
		
		streamer.publish(color_image)

		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		# print("joints")
		joints = task.detector.detect_right_joints()
		# print("joints_detected")
		hand_detected = (joints is not None) and (len(joints) > 0)
		# print("hand_detected")
		
		# if (not recording_data) or (not hand_detected): 
		#     continue
		
		if time.time() - start_sample_time > dt:
			# get_obs(task, cam_agent, cam_wrist)
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_ee_state is None:
				task.init_ee_state = cur_state[8].clone().cpu()
				task.init_ee_state[:3]+=torch.tensor([0.2,0,0])

			# === X, Y, Z === 
			if hand_detected:

				##team14修改1急停
				vertical = np.array([0, 1, 0])  # 假设Y轴垂直向上
				palm_norm=np.cross(vr_axis_correction(joints['index'][1]-joints['thumb'][0]),vr_axis_correction(joints['ring'][1]-joints['thumb'][0]))
				palm_norm   /= np.linalg.norm(palm_norm)
				palm_up_cos = np.dot(palm_norm, vertical)
				
				# 急停条件：掌心朝上（夹角<30度）且手指张开超过阈值
				EMERGENCY_ANGLE = 0.866  # cos(30°)≈0.866
				EMERGENCY_DIST = 0.07    # 手指张开阈值

				# 计算拇指到多个指尖的距离
				dis_thumb_index = np.linalg.norm(joints['thumb'][-1] - joints['index'][-1])
				dis_thumb_middle = np.linalg.norm(joints['thumb'][-1] - joints['middle'][-1])
				dis_thumb_ring = np.linalg.norm(joints['thumb'][-1] - joints['ring'][-1])

				# 加权平均（侧重食指）
				finger_dis = 0.7*dis_thumb_index + 0.2*dis_thumb_middle + 0.1*dis_thumb_ring
				
				if palm_up_cos > EMERGENCY_ANGLE and finger_dis > EMERGENCY_DIST:
						print("\033[91m[EMERGENCY STOP!] Palm up and fingers spread detected\033[0m")
						
						# # 仿真环境急停
						# env.cur_targets[:, :DOF] = env.flexiv_dof_pos[0, :DOF].clone()
						# gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
						
						# 真实机器人急停
						if test_on_real_robot:
								try:
										robot.Stop()  # 调用机器人急停函数
										# gripper.move(0, 0.1, 20)  # 闭合夹爪
								except Exception as e:
										log.error(f"Emergency stop failed: {str(e)}")
						
						# 跳过后续控制
						break
				###修改1急停

				# use leap_lower_bound and leap_upper_bound to cast joints['wrist'] into [-1, 1] real number
				delta_pos = (joints['thumb'][0] - task.prev_motion_joint['thumb'][0])
				delta_pos = vr_axis_correction(delta_pos) ##(-y)zx -> xyz
				
				delta_pos = scale(torch.from_numpy(delta_pos), isaacgym_lower_bound, isaacgym_upper_bound)

				# ==== avoid jitter ====
				# delta_pos_history.append(delta_pos)
				# delta_pos_history = delta_pos_history[-20:]
				# dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
				# if dis <= 0.02:
				#     delta_pos_history = [delta_pos_history[0]] * 20
				#     delta_pos = delta_pos_history[-1]
				# ======================
					
				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device) ##要移动的距离
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
				
				cur_rb_idx = 8
				cur_pos = cur_state[cur_rb_idx][:3]
				cur_rot_q = cur_state[cur_rb_idx][3:7]
				cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
				z_axis = cur_rot_mat[:, 2]
				
				target_pos = task.init_ee_state[:3] + delta_pos
				
				dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
				mask[cur_rb_idx, :3]  = True
				
				palm_norm=np.cross(vr_axis_correction(joints['index'][1]-joints['thumb'][0]),vr_axis_correction(joints['ring'][1]-joints['thumb'][0]))
				palm_norm   /= np.linalg.norm(palm_norm)    # z-axis
				
				arm_dir     = vr_axis_correction(-joints['middle'][1] + joints['thumb'][0])
				arm_dir     /= np.linalg.norm(arm_dir)      # x-axis


				target_rot_mat = np.stack([arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1)
				target_rot_q = R.from_matrix(target_rot_mat).as_quat()
				target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)
				# print(np.linalg.norm(joints['thumb_pos'][3] - joints['index_pos'][3]))
				
				
				###修改0
				计算拇指到多个指尖的距离
				dis_thumb_index = np.linalg.norm(joints['thumb'][-1] - joints['index'][-1])
				dis_thumb_middle = np.linalg.norm(joints['thumb'][-1] - joints['middle'][-1])
				dis_thumb_ring = np.linalg.norm(joints['thumb'][-1] - joints['ring'][-1])

				# 加权平均（侧重食指）
				finger_dis = 0.7*dis_thumb_index + 0.2*dis_thumb_middle + 0.1*dis_thumb_ring
				
				target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
				task.draw_line(target_pos, target_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))
				
				orn = orientation_error(target_rot_q, cur_rot_q)
				dpose[cur_rb_idx, 3:] = orn
				mask[cur_rb_idx, 3:]  = True
				
				tcp_pos = target_pos + z_axis * 0.15
				nrm = np.linalg.norm(dpose[cur_rb_idx] - last_delta_pos) # filter human hand noise
				
				# if time.time() - last_detect_time > 1:
				#     task.prev_motion_joint = joints
				#     task.init_ee_state = cur_state[8].clone().cpu()
				#     last_detect_time = time.time()
					
				# if not is_validpos(tcp_pos):
				#     task.prev_motion_joint = joints
				#     task.init_ee_state = cur_state[8].clone().cpu()
				#     continue
				
				if not is_validpos(tcp_pos):
					tcp_pos = torch.clamp(tcp_pos, tcp_lower_bound, tcp_upper_bound)
					target_pos = tcp_pos - z_axis * 0.15
					dpose[cur_rb_idx, :3] = target_pos.to(env.device) - cur_pos.to(env.device)
				
				task.draw_sphere(target_pos)
				task.draw_sphere(tcp_pos, color=(0, 0, 0))
				
				if nrm < 0.003:
					dpose[cur_rb_idx] = last_delta_pos

				last_delta_pos = dpose[cur_rb_idx]
				
				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)
				next_dof_pos = env.flexiv_dof_pos + action
				
				act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]

				
				if finger_dis > 0.07 and env.cur_targets[:, DOF] < 0.03:
					sim_gripper_width = 0.09
				elif finger_dis < 0.04 and env.cur_targets[:, DOF] > 0.08:
					sim_gripper_width = 0
				
				# print("@@@666 ", env.cur_targets[:, :14])
				sim_move_gripper(env, sim_gripper_width)
				# print(sim_gripper_width, env.cur_targets[:, :14])
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				if test_on_real_robot:
					try:
						action = act_dof[0][:DOF].tolist()
						# import pdb; pdb.set_trace()
						#修改
						tcp_pos = get_flange_pose(robot)[:3]  # 获取当前 TCP 位姿的前3维 (x, y, z)
						# print("tcp_pos", tcp_pos)
						MAX_VEL, MAX_ACC = scale_velocity_near_boundary(
							action, tcp_pos, target_vel, target_acc, MAX_VEL, MAX_ACC
						)
		
						#修改结束

						# mofy v 、、、
						resisted_vel = [
							v * (1 - task.resistance_factor) 
							for v in MAX_VEL
						]
      
						robot.sendJointPosition(action, target_vel, target_acc, resisted_vel, MAX_ACC)
						# # print('action',np.asarray(act_dof).shape)
				
						print("Sent to Robot - Pos:", action, "Vel:", resisted_vel, "Resistance:", task.resistance_factor)
						#####
						# robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)##todo1控制机械臂移动可以修改targetvel和targteacc的计算方式
						
						
						# open gripper
						if finger_dis > 0.07 and get_gripper_width(gripper) < 0.07 and not is_gripper_moving(gripper):  
							gripper.move(0.09, 0.1, 20)
							# print("666fff [info]: open")
							###fff need to fix the condiction 0.05 to big one // and can set a global condiction to label it
						# close gripper
						if finger_dis < 0.03 and get_gripper_width(gripper) > 0.06 and not is_gripper_moving(gripper):
							gripper.grasp(5)
							# print("666fff [info]: close")
							
					except Exception as e:
						log.error(str(e))
				
				# ===== sync joint state =====
				if test_on_real_robot:
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))   
			
		if not task.initialized:
			print("fff666 [info]: No Task!")
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose, please try to stay still...\033[0m")
				time.sleep(1)
				joints = task.detector.detect_right_joints()
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints


def vr_teleop(task: TeleopPlayer):
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from .vr_detector import AllegroHandDetector
	from .calibration.camera import CameraD400
	from .oculus_streamer import OculusStreamer

	sys.path.append('/home/rhos_pub/LeapTele/python/')
	from leaphand import LeapNode

	task.detector = AllegroHandDetector()
	streamer= OculusStreamer()
	ik  = IKController(damping=0.05)
	#cam_agent = CameraD400(0)
	#cam_wrist = CameraD400(1)
	
	task.init_ee_state = None

	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
    
    # initialization 、、、
	task.resistance_factor = 0.5 # [0, 0.9]

	if test_on_real_robot:
		self_exam(log)
		move_home(robot)    
		# reset_home(task)
        
		robot.setMode(mode.NRT_JOINT_POSITION)
		# global gripper
		# gripper = flexivrdk.Gripper(robot)
			
		init_joints = get_robot_joints(robot)
		DOF = 28
		Flexiv_DOF = len(init_joints)

		target_vel = [0.0] * DOF 
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 27
		target_vel = [0.0] * DOF 
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	if test_on_real_hand:
		leap_hand = LeapNode()
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval",
	)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))

	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	start_sample_time = time.time()

	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		color_image = None #TODO: replace with task ...
		while color_image is None:
			color_image =env.gym.get_camera_image_gpu_tensor(sim,env.envs[0], env.camera_handle,gymapi.IMAGE_COLOR)
			color_image=gymtorch.wrap_tensor(color_image)
			color_image=color_image.cpu().numpy()
			color_image=color_image[:,:,[2,1,0]]
		
		streamer.publish(color_image)
	

		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints_hand= task.detector.detect_right_joints()
		transformed_keypoints=task.detector.detect_transformed_joints()
		print(joints_hand)
		joints_arm=joints_hand['thumb'][0]
		hand_detected = (joints_hand is not None) and (len(joints_hand) > 0)
		
		# if (not recording_data) or (not hand_detected): 
		#     continue
		
		if time.time() - start_sample_time > dt:
			get_obs(task)
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_hand_wrist_joint is None:
					state                           = cur_state[...,:3]
					tip_state                       = state[env.nail_handles]
					wrist_state                     = state[8]
					task.init_tip_joint        = tip_state.clone().cpu()
					task.init_hand_wrist_joint      = wrist_state.clone().cpu()+torch.tensor([0.5,0,-1])
			task.draw_sphere(wrist_state, color=(0, 0, 0), radius=0.02)
			print('wrist_state:',wrist_state)
	
			# === X, Y, Z === 
			if hand_detected:
				rate = (joints_arm[:3] - task.prev_motion_joint[:3]) / (vr_upper_bound - vr_lower_bound)
				print('init:',task.init_hand_wrist_joint)
				delta_pos = scale(torch.from_numpy(rate), isaacgym_lower_bound, isaacgym_upper_bound)
				delta_pos = vr_axis_correction(delta_pos)
				target_wrist_pos = delta_pos + task.init_hand_wrist_joint[:3]  
				target_wrist_pos=np.clip(target_wrist_pos,tcp_lower_bound,tcp_upper_bound)

				# ==== avoid jitter ====
				# delta_pos_history.append(delta_pos)
				# delta_pos_history = delta_pos_history[-20:]
				# dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
				# if dis <= 0.02:
				#     delta_pos_history = [delta_pos_history[0]] * 20
				#     delta_pos = delta_pos_history[-1]
				# ======================
					
				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
				
				dpose[8, :3] = target_wrist_pos.to(env.device) - wrist_state.to(env.device)
				mask[8, :3] = True
				task.draw_sphere(target_wrist_pos.cpu(), color=(0, 0, 0))      

				


				for i, key in enumerate(['index','thumb','middle','ring']):     
						cur_rb_tip_idx           = env.nail_handles[i]
						cur_pos                  = cur_state[cur_rb_tip_idx][:3]
						real_finger_len = np.linalg.norm(joints_hand[key][0] - joints_hand[key][1]) + np.linalg.norm(joints_hand[key][2] - joints_hand[key][1]) \
												+ np.linalg.norm(joints_hand[key][3] - joints_hand[key][2]) + np.linalg.norm(joints_hand[key][4] - joints_hand[key][3])
						if key=='thumb':
							real_finger_len += np.linalg.norm(joints_hand[key][5] - joints_hand[key][4])
							
						leap_hand_finger_len = 0.23
						if key == 'ring' :
							leap_hand_finger_len *= 1.05
						if key == 'index':
							leap_hand_finger_len *= 1.1
						if key == 'thumb':
							leap_hand_finger_len *= 0.8

						target_finger_pos = target_wrist_pos + vr_axis_correction(joints_hand[key][-1] - joints_arm) * leap_hand_finger_len / real_finger_len

						dpose[cur_rb_tip_idx, :3] = target_finger_pos.to(env.device) - cur_pos.to(env.device)
						mask[cur_rb_tip_idx, :3]  = True 
						
						colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
						task.draw_sphere(target_finger_pos, color=colors[i])
				
				
				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)
				next_dof_pos = env.flexiv_dof_pos + action
				
				act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]                
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				if test_on_real_robot:
					try:
						flexiv_targets = act_dof[..., :Flexiv_DOF]
						action = flexiv_targets[0][:Flexiv_DOF].tolist()
      
						# modify v 、、、
						resisted_vel = [
							v * (1 - task.resistance_factor) 
							for v in target_vel[:Flexiv_DOF]
						]
      
						robot.sendJointPosition(action, resisted_vel, target_acc[:Flexiv_DOF], MAX_VEL[:Flexiv_DOF], MAX_ACC[:Flexiv_DOF])
						# print('action',np.asarray(act_dof).shape)
				
						print("Sent to Robot - Pos:", action, "Vel:", resisted_vel, "Resistance:", task.resistance_factor)
      
					except Exception as e:
						log.error(str(e))
				if test_on_real_hand:
					try:
						action = act_dof[0].tolist()
						set_leap_hand_pos(leap_hand,np.asarray(action),env.nail_handles)
					except Exception as e:
						log.error(str(e))

				# ===== sync joint state =====
				# if test_on_real_robot:
				#     joints_arm = get_robot_joints(robot)
				#     env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints_arm)).to(device)
				#     gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))   
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose, please try to stay still...\033[0m")
				time.sleep(1)
				joints_hand = task.detector.detect_right_joints()
				joints_arm=joints_hand['thumb'][0]
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints_arm

def vr_teleop_dexretarget(task: TeleopPlayer):
	OPERATOR2MANO_RIGHT = np.array(
	[
		[0, 1, 0.],
		[0, 0, 1],
		[-1, 0, 0],
	]
)
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/'))
	from .vr_detector import AllegroHandDetector
	from .calibration.camera import CameraD400
	from .oculus_streamer import OculusStreamer

	# sys.path.append('/home/rhos_pub/LeapTele/python/')
	# from leaphand import LeapNode

	task.detector = AllegroHandDetector()
	streamer= OculusStreamer()
	ik  = IKController(damping=0.05)
	#cam_agent = CameraD400(0)
	#cam_wrist = CameraD400(1)
	
	task.init_ee_state = None

	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1

	if test_on_real_robot:
		self_exam(log)
		move_home(robot)    
		# reset_home(task)

		robot.setMode(mode.NRT_JOINT_POSITION)
		# global gripper
		# gripper = flexivrdk.Gripper(robot)
			
		init_joints = get_robot_joints(robot)
		DOF = 28
		Flexiv_DOF = len(init_joints)

		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 27
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	if test_on_real_hand:
		leap_hand = LeapNode()
	dt = 1.0 / frequency
	print(
		"Sending command to robot at",
		frequency,
		"Hz, or",
		dt,
		"seconds interval",
	)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))

	def draw_bounding_box():
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(tcp_lower_bound, torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(tcp_upper_bound, torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))
		task.draw_line(tcp_upper_bound, torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]))

		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_lower_bound[2]]))
		task.draw_line(torch.tensor([tcp_lower_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_lower_bound[0], tcp_upper_bound[1], tcp_upper_bound[2]]))
		task.draw_line(torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_upper_bound[2]]), torch.tensor([tcp_upper_bound[0], tcp_lower_bound[1], tcp_lower_bound[2]]))

	start_sample_time = time.time()
	cfg=None
	with open('/home/rhos_pub/LeapTele/DexCopilot/isaacgymenvs/isaacgymenvs/cfg/retarget/leap_hand_right_.yaml') as f:
		cfg=yaml.load(f, Loader=yaml.FullLoader) 

	retargeter=RetargetingConfig.from_dict(cfg).build()

	
	retargeting_joint_names = retargeter.joint_names
	isaac_joint_names=gym.get_actor_dof_names(env.envs[0],task.env.flexivs[0])
	isaac_joint_indices={}
	for name in retargeting_joint_names:
		if name in isaac_joint_names:
			isaac_joint_indices[retargeting_joint_names.index(name)]=isaac_joint_names.index(name)
	print(retargeting_joint_names,isaac_joint_names)
	isaac_joints=np.array(list(isaac_joint_indices.values()))
	retargeting_joints=np.array(list(isaac_joint_indices.keys()))

	print(retargeter.optimizer.target_link_human_indices)

	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		color_image = None #TODO: replace with task ...
		while color_image is None:
			color_image =env.gym.get_camera_image_gpu_tensor(sim,env.envs[0], env.camera_handle,gymapi.IMAGE_COLOR)
			color_image=gymtorch.wrap_tensor(color_image)
			color_image=color_image.cpu().numpy()
			color_image=color_image[:,:,[2,1,0]]
		
		streamer.publish(color_image)
	

		handle_viewer_events(task, viewer)
		draw_bounding_box()
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints_hand= task.detector.detect_right_joints()
		transformed_keypoints=task.detector.detect_transformed_joints()
		print(joints_hand)
		joints_arm=joints_hand['thumb'][0]
		hand_detected = (joints_hand is not None) and (len(joints_hand) > 0)
		
		# if (not recording_data) or (not hand_detected): 
		#     continue
		
		if time.time() - start_sample_time > dt:
			get_obs(task)
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_hand_wrist_joint is None:
					state                           = cur_state[...,:3]
					tip_state                       = state[env.nail_handles]
					wrist_state                     = state[8]
					task.init_tip_joint        = tip_state.clone().cpu()
					task.init_hand_wrist_joint      = wrist_state.clone().cpu()+torch.tensor([0.5,0,-1])
			task.draw_sphere(wrist_state, color=(0, 0, 0), radius=0.02)
			print('wrist_state:',wrist_state)
	
			# === X, Y, Z === 
			if hand_detected:
				rate = (joints_arm[:3] - task.prev_motion_joint[:3]) / (vr_upper_bound - vr_lower_bound)
				print('init:',task.init_hand_wrist_joint)
				delta_pos = scale(torch.from_numpy(rate), isaacgym_lower_bound, isaacgym_upper_bound)
				delta_pos = vr_axis_correction(delta_pos)
				target_wrist_pos = delta_pos + task.init_hand_wrist_joint[:3]  
				target_wrist_pos=np.clip(target_wrist_pos,tcp_lower_bound,tcp_upper_bound)

				# ==== avoid jitter ====
				# delta_pos_history.append(delta_pos)
				# delta_pos_history = delta_pos_history[-20:]
				# dis = np.linalg.norm(delta_pos_history[0] - delta_pos_history[-1])
				# if dis <= 0.02:
				#     delta_pos_history = [delta_pos_history[0]] * 20
				#     delta_pos = delta_pos_history[-1]
				# ======================
					
				dpose = torch.zeros((env.num_flexiv_bodies, 6), device=env.device)
				mask  = torch.zeros((env.num_flexiv_bodies, 6), dtype=bool, device=env.device)
				
				dpose[8, :3] = target_wrist_pos.to(env.device) - wrist_state.to(env.device)
				mask[8, :3] = True
				task.draw_sphere(target_wrist_pos.cpu(), color=(0, 0, 0))      

				cur_rot_q = cur_state[8][3:7]
				cur_rot_mat = R.from_quat(cur_rot_q.cpu().numpy()).as_matrix()
				z_axis = cur_rot_mat[:, 2]
				
				palm_norm=-np.cross(vr_axis_correction(joints_hand['index'][1]-joints_hand['thumb'][0]),vr_axis_correction(joints_hand['ring'][1]-joints_hand['thumb'][0]))
				palm_norm   /= np.linalg.norm(palm_norm)    # z-axis
				
				arm_dir     = vr_axis_correction(joints_hand['middle'][1] - joints_hand['thumb'][0])
				arm_dir     /= np.linalg.norm(arm_dir)      # x-axis
				
				target_rot_mat = np.stack([arm_dir, np.cross(palm_norm, arm_dir), palm_norm], axis=1)
				target_rot_q = R.from_matrix(target_rot_mat).as_quat()
				target_rot_q = torch.from_numpy(target_rot_q).type_as(cur_rot_q)

				target_rot_mat = R.from_quat(target_rot_q.cpu().numpy()).as_matrix()
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 0] * 0.1, color=(1, 0, 0))
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 1] * 0.1, color=(0, 1, 0))
				task.draw_line(target_wrist_pos, target_wrist_pos + target_rot_mat[:, 2] * 0.1, color=(0, 0, 1))

				orn = orientation_error(target_rot_q, cur_rot_q)
				dpose[8, 3:] = orn
				mask[8, 3:]  = True

				indices=(retargeter.optimizer.target_link_human_indices/4).astype(int)
				fingertip_pos=np.stack([np.array(transformed_keypoints['thumb'][0])@OPERATOR2MANO_RIGHT,np.array(transformed_keypoints['thumb'][-1])@OPERATOR2MANO_RIGHT,np.array(transformed_keypoints['index'][-1])@OPERATOR2MANO_RIGHT,np.array(transformed_keypoints['middle'][-1])@OPERATOR2MANO_RIGHT,np.array(transformed_keypoints['ring'][-1])@OPERATOR2MANO_RIGHT])
				ref_pos=[]
				ref_pos=fingertip_pos[indices[1,:]]-fingertip_pos[indices[0,:]]

				#for i, key in enumerate(['thumb','index','middle','ring']): 
					#ref_pos.append(np.array(transformed_keypoints[key][-1])@OPERATOR2MANO_RIGHT)
				print('transformed:',transformed_keypoints)
				ref_pos=np.stack(ref_pos)
				qpos=retargeter.retarget(ref_pos)
				print('qpos:',qpos)
				#for i, key in enumerate(['index','thumb','middle','ring']):     
				#        cur_rb_tip_idx           = env.nail_handles[i]
				#        cur_pos                  = cur_state[cur_rb_tip_idx][:3]
				#        real_finger_len = np.linalg.norm(joints_hand[key][0] - joints_hand[key][1]) + np.linalg.norm(joints_hand[key][2] - joints_hand[key][1]) \
				#                                + np.linalg.norm(joints_hand[key][3] - joints_hand[key][2]) + np.linalg.norm(joints_hand[key][4] - joints_hand[key][3])
				#        if key=='thumb':
				#            real_finger_len += np.linalg.norm(joints_hand[key][5] - joints_hand[key][4])
				#            
				#        leap_hand_finger_len = 0.23
				#        if key == 'ring' :
				#            leap_hand_finger_len *= 1.05
				#        if key == 'index':
				#          leap_hand_finger_len *= 1.1
				#        if key == 'thumb':
				#         leap_hand_finger_len *= 0.8

				#        target_finger_pos = target_wrist_pos + vr_axis_correction(joints_hand[key][-1] - joints_arm) * leap_hand_finger_len / real_finger_len

				#        dpose[cur_rb_tip_idx, :3] = target_finger_pos.to(env.device) - cur_pos.to(env.device)
				#        mask[cur_rb_tip_idx, :3]  = True 
				#        
				#        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
				#        task.draw_sphere(target_finger_pos, color=colors[i])
				
				
				fake_slice = torch.ones((1, 6, env.num_dofs), device=env.device)
				fake_jacobian = torch.cat((fake_slice, env.jacobian), dim=0)

				action = ik.control_ik(dpose, fake_jacobian, mask)


				
				next_dof_pos = env.flexiv_dof_pos + action
				
				next_dof_pos[:,isaac_joints]=torch.tensor(qpos[retargeting_joints],dtype=torch.float)


				act_dof = 0.95 * env.cur_targets[:, :env.num_flexiv_dofs] + 0.05 * next_dof_pos
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]                
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				if test_on_real_robot:
					try:
						flexiv_targets = act_dof[..., :Flexiv_DOF]
						action = flexiv_targets[0][:Flexiv_DOF].tolist()
						robot.sendJointPosition(action, target_vel[:Flexiv_DOF], target_acc[:Flexiv_DOF], MAX_VEL[:Flexiv_DOF], MAX_ACC[:Flexiv_DOF])
						# print('action',np.asarray(act_dof).shape)
					except Exception as e:
						log.error(str(e))
				if test_on_real_hand:
					try:
						action = act_dof[0].tolist()
						set_leap_hand_pos(leap_hand,np.asarray(action),env.nail_handles)
					except Exception as e:
						log.error(str(e))

				# ===== sync joint state =====
				# if test_on_real_robot:
				#     joints_arm = get_robot_joints(robot)
				#     env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints_arm)).to(device)
				#     gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))   
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True
				print("\033[91mSampling human pose, please try to stay still...\033[0m")
				time.sleep(1)
				joints_hand = task.detector.detect_right_joints()
				joints_arm=joints_hand['thumb'][0]
				print("\033[92mSampled\033[0m")
				task.prev_motion_joint = joints_arm


# @HongLi


import pysnooper

# Tactile teleop Needed. @xuyue
def koch_umi_gopro_tactile_teleop(task: TeleopPlayer):
	import pathlib
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ /'))
	from koch import LeadArmReader
	from .calibration.camera import CameraTactile
	calib_path = "/home/rhos/Desktop/LeapTele/DexCopilot/isaacgymenvs/isaacgymenvs/tasks/lerobot/.cache/calibration/koch/left_leader.json"
	task.detector = LeadArmReader("/dev/ttyUSB0", calib_path)
	ik  = IKController(damping=0.05)
	# cam_wrist = GoPro()
	cam_agent = CameraTactile("/dev/video10", "/dev/video10")
	from .calibration.camera import CameraD400
	cam_left = CameraD400("001622071104")
	cam_right = CameraD400("233722071807")
	
	task.init_ee_state = None
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		try:
			gripper.move(0.09, 0.1, 20)
		except Exception as e:
			log.error(str(e))
			
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print("Sending command to robot at", frequency, "Hz, or", dt, "seconds interval",)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))
	

	delta_pos_history = []
	start_sample_time = time.time()
	sim_gripper_width = 0
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		
		handle_viewer_events(task, viewer, save_folder="/Disk1/xuyue/teleop_data/raw")
		draw_bounding_box_global(task, tcp_lower_bound, tcp_lower_bound)
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints, gripper_angle = task.detector.get_follow_arm_joints()
		eef_position = task.detector.get_lead_arm_eef()
		# print("eef_position=", eef_position)
		lowcost_lower_bound = np.array([-0.09, 0.09, 0.02])
		lowcost_upper_bound = np.array([0.06, 0.24, 0.16])	
		eef_pos = eef_position[0]
		outofbound = np.any(eef_pos > lowcost_upper_bound) or np.any(eef_pos < lowcost_lower_bound)
		# print("outofbound=", outofbound)
		hand_detected = (joints is not None) and (gripper_angle is not None) and not outofbound
		# print("hand_detected=",hand_detected)

		if time.time() - start_sample_time > dt:
			# get_obs(task)
			get_obs(task, cam_agent = cam_agent, cam_left = cam_left, cam_right = cam_right)
			# cv2.imshow("GoPro", cam_wrist_view[-1]["image"])
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_ee_state is None:
				task.init_ee_state = cur_state[8].clone().cpu()

			# === X, Y, Z === 
			if hand_detected:
				act_dof = torch.tensor(joints)
				finger_dis = gripper_angle
				# act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				
				# print("finger_dis: ", finger_dis)
				if finger_dis > 0.1: # and env.cur_targets[:, DOF] < 0.03:
					sim_gripper_width = 0.1
				elif finger_dis < 0.1: # and env.cur_targets[:, DOF] > 0.08:
					sim_gripper_width = 0.01
				
				# print("@@@ ", env.cur_targets[:, :14])
				sim_move_gripper(env, sim_gripper_width)
				# print(sim_gripper_width, env.cur_targets[:, :14])
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				# print("act_dof: ", act_dof)
				if test_on_real_robot:
					try:
						# input("Press Enter to Continue!")
						action = act_dof[:DOF].tolist()
						robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
						
						# target_width=np.clip(sim_gripper_width, 0.09, 0)
						gripper.move(sim_gripper_width, 0.1, 20)
							
					except Exception as e:
						log.error(str(e))
				
				# ===== sync joint state =====
				if test_on_real_robot:
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))   
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True




# OCL4Rob Needed. @Yushun
def koch_umi_gopro_teleop(task: TeleopPlayer):
	import pathlib
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ /'))
	from koch import LeadArmReader
	from .calibration.camera import CameraD400
	calib_path = "/home/rhos/Desktop/LeapTele/DexCopilot/isaacgymenvs/isaacgymenvs/tasks/lerobot/.cache/calibration/koch/left_leader.json"
	task.detector = LeadArmReader("/dev/ttyUSB0", calib_path)
	ik  = IKController(damping=0.05)
	# cam_wrist = GoPro()
	# cam_agent = CameraD400("233722071467")
	# cam_wrist = CameraD400("233722071807")
	
	task.init_ee_state = None
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		try:
			gripper.move(0.09, 0.1, 20)
		except Exception as e:
			log.error(str(e))
			
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print("Sending command to robot at", frequency, "Hz, or", dt, "seconds interval",)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))
	

	delta_pos_history = []
	start_sample_time = time.time()
	sim_gripper_width = 0
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()
		
		handle_viewer_events(task, viewer, save_folder="/home/rhos/ziyu/data/")
		draw_bounding_box_global(task, tcp_lower_bound, tcp_lower_bound)
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints, gripper_angle = task.detector.get_follow_arm_joints()
		eef_position = task.detector.get_lead_arm_eef()
		# print("eef_position=", eef_position)
		lowcost_lower_bound = np.array([-0.09, 0.09, 0.02])
		lowcost_upper_bound = np.array([0.06, 0.24, 0.16])	
		eef_pos = eef_position[0]
		outofbound = np.any(eef_pos > lowcost_upper_bound) or np.any(eef_pos < lowcost_lower_bound)
		# print("outofbound=", outofbound)
		hand_detected = (joints is not None) and (gripper_angle is not None) and not outofbound
		# print("hand_detected=",hand_detected)

		if time.time() - start_sample_time > dt:
			# get_obs(task, cam_wrist=cam_wrist, cam_agent=cam_agent)
			# cv2.imshow("GoPro", cam_wrist_view[-1]["image"])
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_ee_state is None:
				task.init_ee_state = cur_state[8].clone().cpu()

			# === X, Y, Z === 
			if hand_detected:
				act_dof = torch.tensor(joints)
				finger_dis = gripper_angle
				print(act_dof.shape, env.flexiv_dof_lower_limits.shape)
				# act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				
				# print("finger_dis: ", finger_dis)
				if finger_dis > 0.1: # and env.cur_targets[:, DOF] < 0.03:
					sim_gripper_width = 0.1
				elif finger_dis < 0.1: # and env.cur_targets[:, DOF] > 0.08:
					sim_gripper_width = 0.01
				
				# print("@@@ ", env.cur_targets[:, :14])
				print(finger_dis)
				sim_move_gripper(env, sim_gripper_width)
				# print(sim_gripper_width, env.cur_targets[:, :14])
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				# print("act_dof: ", act_dof)
				if test_on_real_robot:
					try:
						# input("Press Enter to Continue!")
						action = act_dof[:DOF].tolist()
						robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
						
						# target_width=np.clip(sim_gripper_width, 0.09, 0)
						gripper.move(sim_gripper_width, 0.1, 20)
							
					except Exception as e:
						log.error(str(e))
				
				# ===== sync joint state =====
				if test_on_real_robot:
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))   
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True



def umi_trajectory_only_teleop(task):
	print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
	print("$   x     x  y    y                    XUYUE WARNING:                           $")
	print("$     x x      y y     ::    This is a trajectory only data collection task,    $")
	print("$     x x       y      ::        not the regular koch teleoperation.            $")
	print("$   x     x    y             Please edit the `main` function for your own good  $")
	print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

	import pathlib
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ /'))
	from koch import LeadArmReader
	calib_path = "/home/rhos/Desktop/LeapTele/DexCopilot/isaacgymenvs/isaacgymenvs/tasks/lerobot/.cache/calibration/koch/left_leader.json"
	task.detector = LeadArmReader("/dev/ttyUSB0", calib_path)
	ik  = IKController(damping=0.05)
	
	task.init_ee_state = None
	
	env = task.env
	gym = task.env.gym
	sim = task.env.sim
	device = env.device
	viewer = task.env.viewer
	task.subscribe_keyboard_events(gym ,viewer)

	# sync sim & real robot joints
	DOF = -1
	
	if test_on_real_robot:
		self_exam(log)
		move_home(robot)
		robot.setMode(mode.NRT_JOINT_POSITION)
		global gripper
		gripper = flexivrdk.Gripper(robot)
		try:
			gripper.move(0.09, 0.1, 20)
		except Exception as e:
			log.error(str(e))
			
		init_joints = get_robot_joints(robot)
		DOF = len(init_joints)
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [1.0] * DOF
		MAX_ACC = [0.5] * DOF
		
		init_joints = to_torch(init_joints)
		
		reset_joints(task, init_joints)
	
	else: 
		DOF = 7
		target_vel = [0.0] * DOF
		target_acc = [0.0] * DOF
		MAX_VEL = [0.10] * DOF
		MAX_ACC = [0.20] * DOF
		reset_joints(task, home_joint)    
	
	dt = 1.0 / frequency
	print("Sending command to robot at", frequency, "Hz, or", dt, "seconds interval",)

	assert DOF > 0, "DOF must be greater than 0"
	env.actions = torch.zeros((1, env.num_dofs))
	

	delta_pos_history = []
	start_sample_time = time.time()
	sim_gripper_width = 0
	while not env.gym.query_viewer_has_closed(env.viewer):    
		task.step()


		# add force
		
		
		handle_viewer_events(task, viewer, save_folder="/home/rhos/tactile_umi_grav/data/teleop_for_compare_trajectory/")
		draw_bounding_box_global(task, tcp_lower_bound, tcp_lower_bound)
		
		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		joints, gripper_angle = task.detector.get_follow_arm_joints()		
		hand_detected = (joints is not None) and (gripper_angle is not None)

		if time.time() - start_sample_time > dt:
			get_obs(task)
			# cv2.imshow("GoPro", cam_wrist_view[-1]["image"])
			start_sample_time = time.time()

		if test_on_real_robot and robot.isFault():
			raise Exception("Fault occurred on robot server, exiting ...")
		
		if task.initialized:
			cur_state = torch.squeeze(env.rigid_body_states[..., :env.num_flexiv_bodies])
			if task.init_ee_state is None:
				task.init_ee_state = cur_state[8].clone().cpu()

			# === X, Y, Z === 
			if hand_detected:
				act_dof = torch.tensor(joints)
				finger_dis = gripper_angle
				act_dof = tensor_clamp(act_dof, env.flexiv_dof_lower_limits, env.flexiv_dof_upper_limits)
				
				env.cur_targets[:, :DOF] = act_dof[..., :DOF]
				
								
				# print("finger_dis: ", finger_dis)
				if finger_dis > 0.4: # and env.cur_targets[:, DOF] < 0.03:
					sim_gripper_width = 0.10
				elif finger_dis < 0.4: # and env.cur_targets[:, DOF] > 0.08:
					sim_gripper_width = 0.03
				# print("@@@ ", env.cur_targets[:, :14])
				sim_move_gripper(env, sim_gripper_width)
				# print(sim_gripper_width, env.cur_targets[:, :14])
				gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
					
				# print("act_dof: ", act_dof)
				if test_on_real_robot:
					try:
						# input("Press Enter to Continue!")
						action = act_dof[:DOF].tolist()
						robot.sendJointPosition(action, target_vel, target_acc, MAX_VEL, MAX_ACC)
						
						# target_width=np.clip(sim_gripper_width, 0.09, 0)
						gripper.move(sim_gripper_width, 0.1, 20)
							
					except Exception as e:
						log.error(str(e))
				
				# ===== sync joint state =====
				if test_on_real_robot:
					joints = get_robot_joints(robot)
					env.cur_targets[:, :DOF] = torch.from_numpy(np.array(joints)).to(device)
					gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))   
			
		if not task.initialized:
			if hand_detected:
				task.initialized = True
