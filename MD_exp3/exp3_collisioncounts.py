from SwarmSim.MultiSwarmSim import *
from SwarmSim.ExpHelper import *
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

NUM_RUNS = 100
NUM_TRAINING_STEPS = 100
NUM_INFERENCE_STEPS = 50
ANIMATE = False

swarm1 = [4, 'planar', ['b', 'r'], [0, 0, 0], [13, 13, 13]]
swarm2 = [4, 'planar',  ['g', 'xkcd:orange'], [13, 13, 13], [0, 0, 0]]
multiswarm_options = [swarm1, swarm2]

collision_count = 0
collision_seen  = False

for n in range(NUM_RUNS):
    print('run:', n + 1)

    # rnd_seed = 0 # when set to 0 we don't get the LINALG exception
    rnd_seed = random.randint(0, 10000000)
    # rnd_seed=2265069 # DEAD RECONING PATH ?!?!?!
    print('rnd_seed:', rnd_seed)
    print('\n')

    # with Model
    sim = MultiSwarmSim(multiswarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, True, 'Temporal GCRF', ANIMATE)

    for i in range(NUM_TRAINING_STEPS):
        sim.tick()

    # starting inference with TRUE means we use the model
    sim.start_inference(True)
    for i in range(NUM_INFERENCE_STEPS):
        sim.tick()

        if(sim.collision_observed() and not collision_seen):
            collision_seen = True
            collision_count += 1
            print("collision!")

    collision_seen = False

print("collision count: {}".format(collision_count))

