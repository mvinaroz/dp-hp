import os

def params_to_strings(base_string, params, add_index=False):
  # if no other function is specified, take the powerset of parameter settings and add a setup for each
  setups = [[]]
  for param_values in params:
    new_setups = []
    for s in setups:
      s_extended = [s + [p] for p in param_values]
      new_setups.extend(s_extended)
    setups = new_setups

  if add_index:
    setups = [[i] + s for i, s in enumerate(setups)]
  return [base_string.format(*p) for p in setups]


def gen_cluster_scripts(experiment_name, save_dir, base_string, params, runs_per_job=3, params_to_string_fun=None):

  if params_to_string_fun is None:
    params_to_string_fun = params_to_strings

  run_strings = params_to_string_fun(base_string, params)

  run_script = ['#!/bin/bash', 'if [ $1 -eq 0 ] ; then']
  jobs_count = 1
  runs_at_job = 0
  for idx, run in enumerate(run_strings):
    run_script.append(run)
    runs_at_job +=1
    if runs_at_job == runs_per_job and not idx + 1 == len(run_strings):
      run_script.append(f'fi ; if [ $1 -eq {jobs_count} ] ; then')
      jobs_count += 1
      runs_at_job = 0

  run_script.append('fi')

  condor_script = [f'executable = {experiment_name}.sh',
                   'arguments = $(Process)',
                   f'error = joblogs/{experiment_name}_$(Process).err.txt',
                   f'output = joblogs/{experiment_name}_$(Process).out.txt',
                   f'log = joblogs/{experiment_name}_$(Process).log.txt',
                   'getenv = True',
                   'request_cpus = 1',
                   'request_gpus = 2',
                   'request_memory = 8GB',
                   '+MaxRunningPrice = 500',
                   '+RunningPriceExceededAction = "restart"',
                   f'queue {jobs_count}']

  with open(os.path.join(save_dir, f'{experiment_name}.sh'), 'w') as f:
    f.writelines([l + '\n' for l in run_script])
    for line in run_script:
      print(line.rstrip())
  print('-------------------------------------------')

  with open(os.path.join(save_dir, f'{experiment_name}_sub.sb'), 'w') as f:
    f.writelines([l + '\n' for l in condor_script])
    for line in condor_script:
      print(line.rstrip())
  print('-------------------------------------------')


if __name__ == '__main__':
  experiment_name = 'cgtest'
  save_dir = '.'
  base_string = 'python3 dostuff.py --flag1 {} --flag2 {}'
  params = [[k for k in range(50)], ['a', 'b', 'c']]
  gen_cluster_scripts(experiment_name, save_dir, base_string, params, runs_per_job=3, params_to_string_fun=None)