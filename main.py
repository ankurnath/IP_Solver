import os
import subprocess


# algorithms = ['Greedy','TS','EO','SG','Gurobi','Cplex']

# algorithms = ['Greedy','TS','EO','SG','Cplex']
# algorithms = ['SDP']
algorithms = ['Gurobi']

distributions = [
                'ER_800vertices_unweighted',
                'ER_1000vertices_unweighted',
                'planar_1000vertices_unweighted',
                'ER_2000vertices_unweighted',
                'planar_2000vertices_unweighted',
                'ER_2000vertices_weighted',
                'ER_5000vertices_unweighted',
                'planar_5000vertices_unweighted',
                'ER_5000vertices_weighted',
                'ER_7000vertices_unweighted',
                'planar_7000vertices_unweighted',
                'ER_10000vertices_unweighted',
                'ER_7000vertices_weighted',
                
                'ER_800vertices_weighted',
                
                
                'planar_2000vertices_weighted',
                
                'planar_5000vertices_weighted',
                
                'planar_7000vertices_weighted',
                'planar_800vertices_unweighted',
                'planar_800vertices_weighted',
                'torodial_10000vertices_weighted',
                'torodial_2000vertices_weighted',
                'torodial_3000vertices_unweighted',
                'torodial_5000vertices_weighted',
                'torodial_7000vertices_weighted',
                'torodial_8000vertices_weighted',
                'torodial_800vertices_weighted',
                'torodial_9000vertices_weighted',           
       

                 ]

for distribution in distributions:
    for algorithm in algorithms:
        if algorithm in ['Cplex','Gurobi','SG','Greedy','SDP']:
            command = f'python {algorithm}.py --distribution {distribution} '

        elif algorithm == 'TS':
            command = f'python {algorithm}.py --distribution {distribution} --gamma 100'

        elif algorithm =='EO':
            command = f'python {algorithm}.py --distribution {distribution} --tau 1.4'

        subprocess.run(command, shell=True, check=True)