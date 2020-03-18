import numpy as np
from scipy import stats
import math


def calc_conf_int(v):
	sample_size = v.shape[0]
	conf_level = 0.1
	v_mean = np.mean(v)
	v_int_width = stats.t.ppf(1-conf_level/2, sample_size-1) * (np.std(v)/math.sqrt(sample_size))
	return [v_mean, v_int_width]


filenames = ["OUTPUT(9,planar,const_vel)",
			 "OUTPUT(8,cube,const_vel)",
			 "OUTPUT(9,planar,PID_vel)"]
			 #"OUTPUT(8,cube,PID_vel)"]

for filename in filenames:
	
	print(filename)
	
	errors = np.loadtxt('OUTPUT/' + filename + '.csv', skiprows=1, delimiter=',')

	rnd_seeds = errors[:,0] 

	errors = errors[:, 1:10]
	print(errors.shape)
	print('# of unique random seeds:', len(np.unique(rnd_seeds)))
	print()

	for coordinate in [0,1,2]:

		model_errors = {}
		model_errors['DR'] = errors[:, coordinate]
		model_errors['UR'] = errors[:, 3 + coordinate]
		model_errors['GCRF'] = errors[:, 6 + coordinate]

		model_names = ['DR', 'UR', 'GCRF']

		table = np.zeros((len(model_names),2), dtype=float)

		for k in range(0,len(model_names)):
			[avg_err, conf_int] = calc_conf_int(model_errors[model_names[k]])
			print( str(avg_err) + '\t' + str(conf_int) )
			table[k,0] = avg_err
			table[k,1] = conf_int

		print()

		print( table[0,0] - table[0,1] )
		print( table[1,0] - table[1,1] )
		print( table[2,0] + table[2,1] )

		print('=================================================')

	
	print('\n==================================================================================================\n')
	
	
	
print('Done!')