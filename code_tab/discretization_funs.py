import numpy as np


def discretize_adult_data(gen_data):
  # data, categorical_columns, ordinal_columns = load_dataset('adult')
  data = np.round(gen_data).astype(np.int)

  # cut down features 2, 10, 11, 12 to usable ranges
  def bound_and_squeeze(vals, bound_max=None, squeeze_factor=None):
    vals = vals if bound_max is None else np.minimum(bound_max, vals)
    vals = vals if squeeze_factor is None else vals // squeeze_factor
    return vals

  data[:, 2] = bound_and_squeeze(data[:, 2], 500_000, 50_000)
  data[:, 10] = bound_and_squeeze(data[:, 10], 15_000, 1_500)
  data[:, 11] = bound_and_squeeze(data[:, 11], 2_500, 250)
  data[:, 12] = bound_and_squeeze(data[:, 12], None, 10)

  bounded_domain = [91, 9, 11, 16, 16, 7, 15, 6, 5, 2, 11, 11, 10, 41, 2]
  bound_by_domain(data, bounded_domain)
  np.save('../data/generated/gen_then_disc_bounded_adult.npy', data)

  # bounded_domain = '{"age": 91, "workclass": 9, "fnlwgt": 11, "education-type": 16, "education-num": 16, ' \
  #                  '"marital-status": 7, "occupation": 15, "relationship": 6, "race": 5, "sex": 2, ' \
  #                  '"capital-gain": 11, "capital-loss": 11, "hours-per-week": 10, "native-country": 41, ' \
  #                  '"income>50K": 2}'
  data[:, 0] = bound_and_squeeze(data[:, 0], 70, 10)
  data[:, 8] = bound_and_squeeze(data[:, 8], 1, None)
  data[:, 10] = bound_and_squeeze(data[:, 10], 1, None)
  data[:, 11] = bound_and_squeeze(data[:, 11], 1, None)
  data[:, 12] = bound_and_squeeze(data[:, 12], 6, 2)
  data[:, 13] = bound_and_squeeze(data[:, 13], 1, None)

  simple_domain = [8, 9, 11, 16, 16, 7, 15, 6, 2, 2, 2, 2, 4, 2, 2]
  bound_by_domain(data, simple_domain)
  np.save('../data/generated/gen_then_disc_simple_adult.npy', data)

  # simple_domain = '{"age": 8, "workclass": 9, "fnlwgt": 11, "education-type": 16, "education-num": 16, ' \
  #                 '"marital-status": 7, "occupation": 15, "relationship": 6, "race": 2, "sex": 2, ' \
  #                 '"capital-gain": 2, "capital-loss": 2, "hours-per-week": 4, "native-country": 2, ' \
  #                 '"income>50K": 2}'


def bound_by_domain(data, domain):
  assert data.shape[1] == len(domain)
  for idx in range(len(domain)):
    d_i = data[:, idx]
    data[:, idx] = np.minimum(domain[idx] - np.min(d_i), d_i)
  return data


def discretize_census_data(gen_data):

  data = np.round(gen_data).astype(np.int)
  n_samples, n_features = data.shape

  def bound_and_squeeze(vals, bound_max=None, squeeze_factor=None):
    vals = vals if bound_max is None else np.minimum(bound_max, vals)
    vals = vals if squeeze_factor is None else vals // squeeze_factor
    return vals

  for idx in range(n_features):
    data_i = data[:, idx]
    print(f'{idx}: min {np.min(data_i)} max {np.max(data_i)}')
    if np.max(data_i) > 50:
      data_i = bound_and_squeeze(data_i, None, squeeze_factor=np.max(data_i) // 10)
      data[:, idx] = data_i

  bounded_domain = [10, 8, 10, 46, 16, 10, 2, 6, 23, 14, 4, 9, 1, 2, 5, 7, 10, 10, 10, 5,
                    5, 50, 37, 7, 9, 8, 9, 2, 3, 6, 4, 42, 42, 42, 4, 2, 2, 2, 10, 1, 1]
  data = bound_by_domain(data, bounded_domain)
  np.save('../data/generated/gen_then_disc_bounded_census.npy', data)

  for idx in range(n_features):
    data_i = data[:, idx]
    print(f'{idx}: min {np.min(data_i)} max {np.max(data_i)}')
    if np.max(data_i) > 18:
      data_i = bound_and_squeeze(data_i, None, squeeze_factor=np.max(data_i) // 10)
      data[:, idx] = data_i
      print(f'--> {idx}: min {np.min(data_i)} max {np.max(data_i)}')

  simple_domain = [10, 8, 10, 11, 16, 10, 2, 6, 11, 14, 4, 9, 1, 2, 5, 7, 10, 10, 10, 5,
                   5, 10, 12, 7, 9, 8, 9, 2, 3, 6, 4, 10, 10, 10, 4, 2, 2, 2, 10, 1, 1]

  data = bound_by_domain(data, simple_domain)
  np.save('../data/generated/gen_then_disc_simple_census.npy', data)


def gen_adult_discrete():
  data_path = '../data/generated/adult/adult_generated_privatized_1_eps_1.0_epochs_8000_features_1000_samples_11077_features_14.npz'
  mat = np.load(data_path)
  x, y = mat['x'], mat['y']
  xy = np.concatenate([x, y], axis=1)
  discretize_adult_data(xy)


if __name__ == '__main__':
  gen_adult_discrete()