import os


def params_to_strings(base_cmd, flag_val_list, exp_id_flag=None, id_offset=0):
  # if no other function is specified, take the powerset of parameter settings and add a setup for each
  setups = [[]]
  for flag, param_values in flag_val_list:
    new_setups = []

    if '{}' not in flag:
      flag = str(flag) + ' {}'

    if not isinstance(param_values[0], tuple):
      param_values = [(k,) for k in param_values]

    for s in setups:
      print(flag, param_values)
      s_extended = [s + [flag.format(*p)] for p in param_values]
      new_setups.extend(s_extended)
    setups = new_setups

  if exp_id_flag is not None:
    exp_id_strings = [exp_id_flag.format(i) + ' ' for i in range(id_offset + len(setups))]
  else:
    exp_id_strings = ['']*len(setups)

  return [f'{base_cmd} {exp_id_strings[i]}' + ' '.join(p) for i, p in enumerate(setups)]


def gen_cluster_scripts(experiment_name, save_dir, base_cmd, flag_val_list, exp_id_flag, id_offset=0, runs_per_job=3):

  # if params_to_string_fun is None:
  #   params_to_string_fun = params_to_strings
  assert (exp_id_flag is None) or ('{}' in exp_id_flag)

  run_strings = params_to_strings(base_cmd, flag_val_list, exp_id_flag, id_offset)

  run_script = ['#!/bin/bash', 'if [ $1 -eq 0 ] ; then']
  jobs_count = 1
  runs_at_job = 0
  for idx, run in enumerate(run_strings):
    run_script.append(run)
    runs_at_job += 1
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
                   'request_cpus = 2',
                   'request_gpus = 2',
                   'request_memory = 8GB',
                   '+MaxRunningPrice = 500',
                   '+RunningPriceExceededAction = "restart"',
                   # 'requirements = CUDAGlobalMemoryMb > 5000',
                   'requirements = CUDACapability >= 3.7',
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


def running_eval_script():
  experiment_name = 'apr27_me_eval_small_models_fashion_just_svc'
  save_dir = 'cluster_scripts'
  base_string = 'python3.6 downstream_test.py'
  params = [('--data', ['fashion']),
            # ('', ['--skip-slow-models', '--only-slow-models']),
            ('--custom-keys', [  # 'logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb',
              'linear_svc']),
            ('--log-name {} --seed {}',
             list(zip([f'apr23_me_training_{k}' for k in range(10)],
                      [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))),
            ('-rate', [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
            ]
  exp_id_flag = None
  gen_cluster_scripts(experiment_name, save_dir, base_string, params, exp_id_flag, runs_per_job=15)


def running_train_script():
  experiment_name = 'apr27_me_train_dmnist'
  save_dir = 'cluster_scripts'
  base_string = 'python3.6 Me_sum_kernel_args.py'
  params = [('--data', ['digits']),
            ('-bs', [200]),
            ('-ebs', [2000]),
            ('-ep', [10]),
            ('-lr', [0.01]),
            ('--is-private', ['True', 'False']),
            ('--order-hermite', [20, 50, 100, 200, 500, 1000]),
            ('', ['--skip-downstream-model']),
            ('--kernel-length', [0.005]),
            ]
  exp_id_flag = '--log-name apr27_mehp_dmnist_{}'
  gen_cluster_scripts(experiment_name, save_dir, base_string, params, exp_id_flag, runs_per_job=1)


if __name__ == '__main__':
  running_train_script()
