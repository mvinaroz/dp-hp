from all_aux_files import test_gen_data, log_final_score
import argparse
import numpy as np
import faulthandler
faulthandler.enable()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits, fashion and 2d')
  parser.add_argument('--subsampling-rate', '-rate', type=float, default=0.1, help='')
  parser.add_argument('--custom-keys', type=str, default=None, help='')
  parser.add_argument('--skip-slow-models', action='store_true', default=False, help='skip models that take longer')
  parser.add_argument('--only-slow-models', action='store_true', default=False, help='only do slower the models')

  ar = parser.parse_args()

  if ar.log_dir is None:
    assert ar.log_name is not None
    ar.log_dir = ar.base_log_dir + ar.log_name + '/'

  convoluted_log_name = f'{ar.log_name}/{ar.data}'

  np.random.seed(ar.seed)
  # test_results_subsampling_rate(ar.data, ar.log_name, ar.log_dir, False, ar.subsampling_rate)
  final_score = test_gen_data(data_log_name=convoluted_log_name, data_key=ar.data, subsample=ar.subsampling_rate,
                              custom_keys=ar.custom_keys,
                              skip_slow_models=ar.skip_slow_models, only_slow_models=ar.only_slow_models)
  log_final_score(ar.log_dir, final_score)
