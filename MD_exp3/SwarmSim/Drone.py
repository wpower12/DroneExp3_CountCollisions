import numpy as np
# from simple_pid import
import simple_pid
from . import constants as C
  
class Drone():
	def __init__(self):
		# State Vectors
		self.pos = np.zeros((3)) # ACTUAL Location
		self.pos_initial = np.zeros((3)) # Saved positions
		self.pos_estimate = np.zeros((3)) # Estimated via deadreckoning
		self.pos_variance = np.zeros((3))
		self.pos_estimate_animate = np.zeros((3))
		
		self.vel = np.zeros((3)) # Velocity based on self.pos
		self.vel_estimate = np.zeros((3)) # Velocity based on self.pos_estimate
		
		self.acc = np.zeros((3)) # Acceleration based on self.pos
		self.acc_estimate = np.zeros((3)) # Acceleration based on self.pos_estimate

		# Target Vector - Tracking Next Target Position
		self.target       = np.zeros((3)) # Actual position target used for setpoints
		self.saved_target = np.zeros((3)) # Planned target - next waypoint. 'Goal'

		# Controllers
		self.PID_X = None
		self.PID_Y = None
		self.PID_Z = None
		
		self.PID_X_estimate = None
		self.PID_Y_estimate = None
		self.PID_Z_estimate = None

		# History Lists
		self.H_pos = []
		self.H_pos_estimate = []
		self.swarm_index = None
		# Models/External State
		# self.others_pos    = [] # last known pos of other drones
		# self.others_models = [] # model parameters for other drones
	
	
	def set_target(self, t):
		self.target = np.copy(t)
		self.init_PIDs()
	
	
	def init_PIDs(self):
		# Assuming we call this everytime we update the targets
		x, y, z = self.target
		self.PID_X = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=x, sample_time=None)
		self.PID_Y = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=y, sample_time=None)
		self.PID_Z = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=z, sample_time=None)
		
		self.PID_Y_estimate = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=y, sample_time=None)
		self.PID_X_estimate = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=x, sample_time=None)
		self.PID_Z_estimate = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=z, sample_time=None)

	
	def update_state_from_pos(self, pos):
		# Update output limits manually so that we wouldn't clamp the last integral and the last output value
		self.PID_X.output_limits = (-1.0 * C.MAX_ACC - self.acc[0], C.MAX_ACC - self.acc[0])
		self.PID_Y.output_limits = (-1.0 * C.MAX_ACC - self.acc[1], C.MAX_ACC - self.acc[1])
		self.PID_Z.output_limits = (-1.0 * C.MAX_ACC - self.acc[2], C.MAX_ACC - self.acc[2])

		# Changes in acc are the outputs of the PID controllers
		dAcc_x = self.PID_X(pos[0], dt=C.DT)
		dAcc_y = self.PID_Y(pos[1], dt=C.DT)
		dAcc_z = self.PID_Z(pos[2], dt=C.DT)

		# Update acc's by clamp adding differences in acceleration (previously clamped by setting output_limits)
		self.acc = np.asarray([self.acc[0] + dAcc_x, self.acc[1] + dAcc_y, self.acc[2] + dAcc_z])

		# Update velocities by clamp adding the velocity contributions obtained from accelerating DT 'seconds'.
		n_vel_x = clamp_add(self.vel[0], self.acc[0] * C.DT, C.MAX_VEL)
		n_vel_y = clamp_add(self.vel[1], self.acc[1] * C.DT, C.MAX_VEL)
		n_vel_z = clamp_add(self.vel[2], self.acc[2] * C.DT, C.MAX_VEL)

		# self.vel = np.asarray([n_vel_x, n_vel_y, n_vel_z])
		# self.vel /= np.linalg.norm(self.vel)

		if self.swarm_index == 0:
			self.vel = np.asarray([1.0, 1.0, 1.0])
		else:
			self.vel = -np.asarray([1.0, 1.0, 1.0])
	
	
	def update_state_from_pos_estimate(self, pos_estimate):
		# Update output limits manually so that we wouldn't clamp the last integral and the last output value
		self.PID_X_estimate.output_limits = (-1.0 * C.MAX_ACC - self.acc_estimate[0], C.MAX_ACC - self.acc_estimate[0])
		self.PID_Y_estimate.output_limits = (-1.0 * C.MAX_ACC - self.acc_estimate[1], C.MAX_ACC - self.acc_estimate[1])
		self.PID_Z_estimate.output_limits = (-1.0 * C.MAX_ACC - self.acc_estimate[2], C.MAX_ACC - self.acc_estimate[2])
		
		# Changes in acc are the outputs of the PID controllers
		dAcc_x = self.PID_X_estimate(pos_estimate[0], dt=C.DT)
		dAcc_y = self.PID_Y_estimate(pos_estimate[1], dt=C.DT)
		dAcc_z = self.PID_Z_estimate(pos_estimate[2], dt=C.DT)	
		
		# Update acc's by clamp adding differences in acceleration (previously clamped by setting output_limits)
		self.acc_estimate = np.asarray([self.acc_estimate[0] + dAcc_x, self.acc_estimate[1] + dAcc_y, self.acc_estimate[2] + dAcc_z])
		
		# Update velocities by clamp adding the velocity contributions obtained from accelerating DT 'seconds'.
		n_vel_x = clamp_add(self.vel_estimate[0], self.acc_estimate[0] * C.DT, C.MAX_VEL)
		n_vel_y = clamp_add(self.vel_estimate[1], self.acc_estimate[1] * C.DT, C.MAX_VEL)
		n_vel_z = clamp_add(self.vel_estimate[2], self.acc_estimate[2] * C.DT, C.MAX_VEL)

		# self.vel_estimate = np.asarray([n_vel_x, n_vel_y, n_vel_z])
		# self.vel_estimate /= np.linalg.norm(self.vel_estimate)

		if self.swarm_index == 0:
			self.vel = np.asarray([1.0, 1.0, 1.0])
		else:
			self.vel = -np.asarray([1.0, 1.0, 1.0])
	
	def update_training(self):
		'''
		# In training, we use the 'real' position
		self.update_state_from_pos(self.pos)

		# Update position.
		self.pos += self.vel*C.DT
		self.H_pos.append(np.copy(self.pos))

		# Save that in the estimated position
		self.pos_estimate = np.copy(self.pos)

		self.model_update()
		'''

	def update_inference(self, model, use_model, X, S, d_index, use_structure):
		'''
		if use_model:
			# Apply output of model to predict deviation from Wind
			# self.pos_estimate += model.predict(X, S, d_index) # I think? more on this later.
			self.pos_estimate = model.predict(X, S, d_index, use_structure) # I think? more on this later.

		# In inference, we use the ESTIMATE of pos to update
		self.update_state_from_pos(self.pos_estimate)

		# Finally, update current estimate of position
		self.H_pos_est.append(np.copy(self.pos_estimate))
		self.pos_estimate += self.vel*C.DT

		# We also update the 'real' position, because the real position
		# has already been moved by the wind, the effect of the PID's 
		# changes to the acceleration are still impacting the true
		# location ONTOP of the wind moving it. 
		self.H_pos.append(np.copy(self.pos))
		self.pos += self.vel*C.DT
		'''

	def model_update(self):
		# TODO
		pass

	def model_predict(self):
		# TODO 
		return np.zeros((3))

	def model_variance(self):
		# TODO
		return 1.0

	def has_reached_target(self, epsilon):
		# return true if distance to target is within epsilon
		return abs(np.linalg.norm(self.pos_estimate-self.target)) < epsilon

	def getVelocityVariance(self):
		n_steps = len(self.H_pos)
		n_inference = len(self.H_pos_estimate)
		if n_inference == 0:
			return np.zeros(3, dtype=float)
		n_train = n_steps - n_inference
		Hpos = np.asarray(self.H_pos)
		Hpos_estimate = np.asarray(self.H_pos_estimate)
		movement_diffs = np.zeros((n_steps - 1, 3), dtype=float)
		movement_diffs[:n_train-1, 0] = np.ediff1d(Hpos[:n_train, 0])
		movement_diffs[:n_train-1, 1] = np.ediff1d(Hpos[:n_train, 1])
		movement_diffs[:n_train-1, 2] = np.ediff1d(Hpos[:n_train, 2])

		# weighted std https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

		#border case
		movement_diffs[n_train-1,:]=Hpos_estimate[0, :] - Hpos[n_train-1, :]

		movement_diffs[n_train:, 0]= np.ediff1d(Hpos_estimate[:,0])
		movement_diffs[n_train:, 1] = np.ediff1d(Hpos_estimate[:, 1])
		movement_diffs[n_train:, 2] = np.ediff1d(Hpos_estimate[:, 2])
		return np.std(movement_diffs, axis=0)

# Returns a value that is max(abs(max_a), abs(a+b))
def clamp_add(a, b, max_a):
	if (a + b) > max_a:
		return max_a
	if (a + b) < -1.0*max_a:
		return -1.0*max_a
	return a + b