import subprocess

distributions = []

for dist in ['ER','BA']:
    for n in [20,40,60,100,200,500]:
        distributions.append(f'{dist}_{n}vertices_weighted')


for algorithm in [
                #   'Greedy',
                #   'SDP',
                  'Gurobi',
                #   'TS',
                #   'EO'
                  ]:

    for distribution in distributions:

        if algorithm == 'Greedy':

            command=f'python {algorithm}.py --distribution {distribution}'
            subprocess.run(command,shell=True)

        elif algorithm == 'TS':

            command=f'python {algorithm}.py --distribution {distribution} --gamma 20'
            subprocess.run(command,shell=True)

        elif algorithm == 'EO':

            command=f'python {algorithm}.py --distribution {distribution} --tau 1.4'
            subprocess.run(command,shell=True)

        elif algorithm == 'Gurobi':

            command=f'python {algorithm}.py --distribution {distribution} --time_limit 10 --threads 10'
            subprocess.run(command,shell=True)

        # break