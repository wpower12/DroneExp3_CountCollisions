from . import Swarm as S
from . import Wind  as W
from . import MultiSwarmAnimator as A
from . import constants as C

import numpy as np
import time

class MultiSwarmSim():
    def __init__(self, swarms, rnd_seed, num_steps_train, num_steps_test, use_struct, figname, animate=True):
        # Assuming swarms holds a list of swarms:
        # [[num_drones, type, plot_color, inital_position, target],..,[...]]
        self.swarms = []
        self.prng = np.random.RandomState(rnd_seed)
        self.wind = W.Wind(self.prng)
        self.animating = animate

        self.did_colision_avoidance = False
        self.post_colision_avoidance_timer = C.WINDOW_SIZE

        lower = np.asarray(swarms[0][3])
        upper = np.asarray(swarms[0][4])

        for i, s in enumerate(swarms):
            num_drones, swarm_type, colors, pos, target = s
            new_sim = S.Swarm(num_drones, self.prng, num_steps_train, num_steps_test, use_struct, swarm_type)
            new_sim.color = colors
            new_sim.set_swarm_pos_relative(pos)
            new_sim.set_swarm_target_relative(target)
            for d in new_sim.drones:
                d.swarm_index = i
            new_sim.init_drone_PIDs()
            self.swarms.append(new_sim)

            lower = np.minimum(lower, np.asarray(pos))
            upper = np.maximum(upper, np.asarray(target))

        if self.animating:
            self.anm  = A.MultiSwarmAnimator(lower, upper, figname)

        self.saved_last_swarm_pos_estimates =[]
        self.delta_vec = None

    def animate(self):
        self.anm.plot_swarms(self.swarms)

    def set_seed(self, n):
        self.wind.set_seed(n)

    def tick(self):
        if self.animating:
            self.animate()
        self.wind.sample_wind()
        for s in self.swarms:
            s.tick(self.wind)


        if self.swarmsIntersect():
            for s in self.swarms:
                swarm_pos_estimates = np.zeros((s.N, 3), dtype=float)
                for i, d in enumerate(s.drones):
                    swarm_pos_estimates[i,:]=np.copy(d.pos_estimate)

                self.saved_last_swarm_pos_estimates.append(swarm_pos_estimates)
            while self.swarmsIntersect():
                deflection_vectors = self.getDeflectionVectors()
                self.singleStepCollisionAvoidance(deflection_vectors)
                self.anm.plot_swarms(self.swarms)
            self.did_colision_avoidance = True

        if self.delta_vec is None and self.did_colision_avoidance:
            self.delta_vec = []
            for si, s in enumerate(self.swarms):
                swarm_delta_vects= np.zeros((s.N, 3))
                for di, d in enumerate(s.drones):
                    swarm_delta_vects[di, :] =  d.pos_estimate_animate - self.saved_last_swarm_pos_estimates[si][di,:]
                self.delta_vec.append(swarm_delta_vects)
        if self.did_colision_avoidance:
            for si, s in enumerate(self.swarms):
                for di, d in enumerate(s.drones):
                    print("Usao za di ", di)
                    d.pos_estimate_animate = d.pos_estimate + self.delta_vec[si][di, :]
        else:
            for s in self.swarms:
                for d in s.drones:
                    d.pos_estimate_animate = d.pos_estimate
        for s in self.swarms:
            if not s.training:
                for d in s.drones:
                    d.H_pos_estimate.append(d.pos_estimate_animate)

        if self.swarms[0].timestep == (self.swarms[0].num_inference_steps + self.swarms[0].num_training_steps - 1):
            time.sleep(100)

    def collision_observed(self):
        s   = self.swarms[0]
        s_o = self.swarms[1]

        for d in s.drones:
            for d_o in s_o.drones:
                if np.abs(np.linalg.norm(d_o.pos - d.pos)) == 0:
                    print(d_o.pos, d.pos)
                    print(np.abs(np.linalg.norm(d_o.pos - d.pos)))
                    return True
        return False


    def start_inference(self, use_model=True):
        for s in self.swarms:
            s.training = False
            s.use_model = use_model

    def use_expansion(self, ph):
        for s in self.swarms:
            s.use_expansion(ph)

    def dump_swarms_location(self, return_actual_positions):
        positions=[]
        for s in self.swarms:
            positions.append(s.dump_locations(return_actual_positions))
        return positions

    def swarmsIntersect(self):
        if self.swarms[0].training:
            return False

        std_1 = self.swarms[0].swarm_variance
        std_2 = self.swarms[1].swarm_variance

        if std_1[0]>0 and std_2[0]>0:

            std_1 += C.DT
            std_2 += C.DT

            dist12 = np.linalg.norm(self.swarms[0].swarm_mean - self.swarms[1].swarm_mean)

            if np.any(std_1+std_2 > dist12):
                print('COLISION!')
                return True

        return False
            # if std_1[0] + std_2[0] > dist12 or \
            # 		std_1[1] + std_2[1] > dist12 or \
            # 		std_1[2] + std_2[2] > dist12:
            # 	# collision detected!
            # 	print('collision detected!')

    def getDeflectionVectors(self):
        deflection_vectors = np.zeros((2,3), dtype=float)
        if self.swarms[0].swarm_mean[2] > self.swarms[1].swarm_mean[2]:
            # 0th swarm goes up
            deflection_vectors[0,:] = np.asarray([0.0,0.0,1.0])
            deflection_vectors[1,:] = -np.asarray([0.0,0.0,1.0])
        else:
            # 0th swarms goes up
            deflection_vectors[0, :] = -np.asarray([0.0, 0.0, 1.0])
            deflection_vectors[1, :] = np.asarray([0.0, 0.0, 1.0])

        return deflection_vectors

    def singleStepCollisionAvoidance(self, deflectionVectors):
        deflectionVectors[0, :] +=  np.asarray([1.0, 1.0, 1.0])
        deflectionVectors[1, :] += -np.asarray([1.0, 1.0, 1.0])

        deflectionVectors[0, :] /= np.linalg.norm(deflectionVectors[0, :])
        deflectionVectors[1, :] /= np.linalg.norm(deflectionVectors[1, :])

        deflectionVectors *= C.DT

        for i, s in enumerate(self.swarms):
            for d in s.drones:
                #d.update_state_from_pos_estimate(d.pos_estimate_animate) # this is the first 4 lines
                d.pos_estimate_animate += deflectionVectors[i, :]
                d.H_pos_estimate.append(np.copy(d.pos_estimate_animate))


                s.updateSwarmMeanPostCollisionAvoidance()
