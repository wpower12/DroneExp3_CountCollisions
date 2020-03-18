from SwarmSim.SingleSwarmSim import *
from SwarmSim.ExpHelper import *
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")


NUM_RUNS = 1
NUM_TRAINING_STEPS = 80 #300 #100
NUM_INFERENCE_STEPS = 50 #100 #30

SWARM_SIZE = 8
SWARM_TYPE = "cube"
START = [0, 0, 0]
END = [10, 10, 4] #[14, 14, 5] 
#[10, 10, 4] #[50, 50, 20] #[20, 20, 20]  # [100,100,100]
ANIMATE = True

swarm_options = [SWARM_SIZE, SWARM_TYPE, "b", START, END]


print('rnd_seed,MSE_DR,,,MSE_UR,,,MSE_GCRF,,,RMSE_DR,,,RMSE_UR,,,RMSE_GCRF,,,R2_DR,,,R2_UR,,,R2_GCRF,,,alpha,beta')

rnd_seeds = np.loadtxt('rnd_seeds.csv', delimiter=',')


for run in range(NUM_RUNS):
	
	#while True:
	if True:
		
		try:

			# rnd_seed = 0 # when set to 0 we don't get the LINALG exception
			#rnd_seed = random.randint(0, 10000000)
			# rnd_seed = 1028791
			###rnd_seed = 3614146
			#rnd_seed = 212413
			
			#rnd_seed = 7501252 # good example
			#rnd_sedd = 3561402 # bad example (UR fails)
			
			rnd_seed = int(rnd_seeds[run])
			
			#print('rnd_seed:', rnd_seed)
			#print('\n\n')

			# =========================================================================
			'''
			# baseline
			np.random.seed = rnd_seed
			sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, None, run, 'Ground truth', ANIMATE)
			sim.set_seed(rnd_seed)
			for i in range(NUM_TRAINING_STEPS + NUM_INFERENCE_STEPS):
				sim.tick()

			print('Ground truth trajectories: generated!')
			print('\n\n')

			# =========================================================================
			'''

			# without Model
			np.random.seed = rnd_seed
			sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, None, run, 'Dead reckoning', ANIMATE)
			sim.set_seed(rnd_seed)
			for i in range(NUM_TRAINING_STEPS):
				sim.tick()

			# starting inference with FALSE means we DONT use the model
			# only dead reckoning
			sim.start_inference(False)
			dr_mse_accumulated = np.zeros((3,), dtype=float)
			dr_rmse_accumulated = np.zeros((3,), dtype=float)
			dr_r2_score_accumulated = np.zeros((3,), dtype=float)
			for i in range(NUM_INFERENCE_STEPS):
				sim.tick()

				target_locations = sim.dump_drone_locations(True)

				dr_locations = sim.dump_drone_locations(False)
				mse_current = calc_mse(target_locations, dr_locations)
				dr_mse_accumulated += mse_current
				dr_rmse_accumulated += np.sqrt(mse_current)
				dr_r2_score_accumulated += calc_r2_score(target_locations, dr_locations)
			dr_mse_avg = dr_mse_accumulated / NUM_INFERENCE_STEPS
			dr_rmse_avg = dr_rmse_accumulated / NUM_INFERENCE_STEPS
			dr_r2_score_avg = dr_r2_score_accumulated / NUM_INFERENCE_STEPS

			#print('DR:\nMSE AVG: {}; RMSE AVG: {}; R2 SCORE AVG: {}'.format(dr_mse_avg, dr_rmse_avg, dr_r2_score_avg))
			#print('\n\n')	

			# =========================================================================

			# with Model
			np.random.seed = rnd_seed
			sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, False, run, 'Unstructured regressor', ANIMATE)
			sim.set_seed(rnd_seed)
			for i in range(NUM_TRAINING_STEPS):
				sim.tick()
				#if i % 100 == 0:
				#    print('t:', i)

			# starting inference with TRUE means we use the model
			sim.start_inference(True)
			model_mse_accumulated = np.zeros((3,), dtype=float)
			model_rmse_accumulated = np.zeros((3,), dtype=float)
			model_r2_score_accumulated = np.zeros((3,), dtype=float)
			for i in range(NUM_INFERENCE_STEPS):
				sim.tick()

				target_locations = sim.dump_drone_locations(True)

				model_locations = sim.dump_drone_locations(False)
				mse_current = calc_mse(target_locations, model_locations)
				model_mse_accumulated += mse_current
				model_rmse_accumulated += np.sqrt(mse_current)
				model_r2_score_accumulated += calc_r2_score(target_locations, model_locations)
			model_unstructured_mse_avg = model_mse_accumulated / NUM_INFERENCE_STEPS
			model_unstructured_rmse_avg = model_rmse_accumulated / NUM_INFERENCE_STEPS
			model_unstructured_r2_score_avg = model_r2_score_accumulated / NUM_INFERENCE_STEPS

			#print('Unstructured model:\nMSE AVG: {}; RMSE AVG: {}; R2 SCORE AVG: {}'.format(model_unstructured_mse_avg,
			#                                                                                model_unstructured_rmse_avg,
			#                                                                                model_unstructured_r2_score_avg))
			#print('\n\n')

			# =========================================================================

			# with Model
			np.random.seed = rnd_seed
			sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, True, run, 'Temporal GCRF', ANIMATE)
			sim.set_seed(rnd_seed)
			for i in range(NUM_TRAINING_STEPS):
				sim.tick()
				#print('t:', i)

			#print('INFERENCE:\n')
			# starting inference with TRUE means we use the model
			sim.start_inference(True)
			model_mse_accumulated = np.zeros((3,), dtype=float)
			model_rmse_accumulated = np.zeros((3,), dtype=float)
			model_r2_score_accumulated = np.zeros((3,), dtype=float)
			for i in range(NUM_INFERENCE_STEPS):
				sim.tick()
				#print('t:', i)

				target_locations = sim.dump_drone_locations(True)

				model_locations = sim.dump_drone_locations(False)
				mse_current = calc_mse(target_locations, model_locations)
				model_mse_accumulated += mse_current
				model_rmse_accumulated += np.sqrt(mse_current)
				model_r2_score_accumulated += calc_r2_score(target_locations, model_locations)
			model_structured_mse_avg = model_mse_accumulated / NUM_INFERENCE_STEPS
			model_structured_rmse_avg = model_rmse_accumulated / NUM_INFERENCE_STEPS
			model_structured_r2_score_avg = model_r2_score_accumulated / NUM_INFERENCE_STEPS

			#print(
			#    'GCRF:\nMSE AVG: {}; RMSE AVG: {}; R2 SCORE AVG: {}'.format(model_structured_mse_avg, model_structured_rmse_avg,
			#                                                                model_structured_r2_score_avg))
			#print('\n\n')


			out_str = str(rnd_seed) + ','

			out_str += ','.join([str(dim_err) for dim_err in dr_mse_avg]) + ','
			out_str += ','.join([str(dim_err) for dim_err in model_unstructured_mse_avg]) + ','
			out_str += ','.join([str(dim_err) for dim_err in model_structured_mse_avg]) + ','

			out_str += ','.join([str(dim_err) for dim_err in dr_rmse_avg]) + ','
			out_str += ','.join([str(dim_err) for dim_err in model_unstructured_rmse_avg]) + ','
			out_str += ','.join([str(dim_err) for dim_err in model_structured_rmse_avg]) + ','


			out_str += ','.join([str(dim_err) for dim_err in dr_r2_score_avg]) + ','
			out_str += ','.join([str(dim_err) for dim_err in model_unstructured_r2_score_avg]) + ','
			out_str += ','.join([str(dim_err) for dim_err in model_structured_r2_score_avg]) + ','

			out_str += str(sim.sim.model.theta[0]) + ',' + str(sim.sim.model.theta[1])

			print(out_str)

			#break

		
		except:

			#print('\n\n')
			print('except')
			pass
		
	

print()
input("Wait ...")

print('Done!')
