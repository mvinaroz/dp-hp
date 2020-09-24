import numpy as np
from collections import namedtuple


cube_tup = namedtuple('cube_tuple', ['min', 'max', 'center', 'data'])


def alg1_private_partition(epsilon, delta, initial_cube):
  depth = 0
  active_cubes = [initial_cube]
  grid_set = []

  epsilon = 1.  # TODO remove debug

  n_data = initial_cube.data.shape[0]
  dp_add_select = _get_dp_selector(n_data, epsilon, delta)

  print(f'starting alg1 loop at depth 0')
  while depth <= np.log(n_data) and active_cubes:
    print(f'depth: {depth}, active cubes: {len(active_cubes)}, grid_set: {len(grid_set)}')
    depth += 1
    grid_set.extend([k.center for k in active_cubes])
    next_active_cubes = []
    for cube in active_cubes:
      sub_cubes = _split_cube(cube)
      next_active_cubes.extend([k for k in sub_cubes if dp_add_select(len(k.data))])
    active_cubes = next_active_cubes
  return np.stack(grid_set)


def _split_cube(cube, axis=0):
  if axis == len(cube.min):  # base case: cube has been split over all axes
    center = cube.min + 0.5 * (cube.max - cube.min)
    if len(cube.data) > 0:
      print(f'ndata {cube.data.shape[0]}')
    return [cube_tup(cube.min, cube.max, center, cube.data)]
  else:  # split cube over current axis, then do the later ones
    axis_split = cube.min[axis] + 0.5 * (cube.max[axis] - cube.min[axis])
    max_a = cube.max.copy()
    max_a[axis] = axis_split
    min_b = cube.min.copy()
    min_b[axis] = axis_split

    if len(cube.data) == 0:
      data_a, data_b = cube.data, cube.data
    else:
      data_a = cube.data[cube.data[:, axis] <= axis_split]
      data_b = cube.data[cube.data[:, axis] > axis_split]
      # print(f'ndata ({cube.data.shape[0]}) split a: {data_a.shape[0]}, b: {data_b.shape[0]}')
    # compute centers only for final cubes
    cube_a = cube_tup(cube.min, max_a, None, data_a)
    cube_b = cube_tup(min_b, cube.max, None, data_b)

    return _split_cube(cube_a, axis+1) + _split_cube(cube_b, axis+1)


def _get_dp_selector(n_data, epsilon, delta):
  epsilon_prime = epsilon / (2 * np.log(n_data))
  gamma = 20 / epsilon_prime * np.log(n_data / delta)
  print(f'eps prime: {epsilon_prime}, gamma {gamma}')

  def dp_add_select(m):
    if m <= gamma:
      threshold = 0.5 * np.exp(-epsilon_prime * (gamma - m))
    else:
      threshold = 1 - 0.5 * np.exp(epsilon_prime * (gamma - m))
    if m > 0:
      print(f'm: {m}, threshold: {threshold}')
    return np.random.rand() < threshold
  return dp_add_select


def alg2_candidate(dataset, epsilon, delta, n_centers, data_radius, max_iter=None):
  # n_data = dataset.shape[0]
  n_feat = dataset.shape[1]
  grid_set = []
  # max_iter = int(np.round(27 * n_centers * np.log(n_data / delta)))
  if max_iter is None:
    max_iter = n_centers  # according to note in section 7

  print(f'starting alg2 loop with {max_iter} iterations')
  for idx in range(max_iter):
    print(f'alg2 iter {idx}')
    shift_vec = np.random.uniform(-data_radius, data_radius, n_feat)
    min_shift = shift_vec - data_radius
    max_shift = shift_vec + data_radius
    init_cube = cube_tup(min_shift, max_shift, shift_vec, dataset)
    grid_set.append(alg1_private_partition(epsilon / max_iter, delta / max_iter, init_cube))
  return np.concatenate(grid_set)


def alg3_localswap(dataset, grid_set, epsilon, delta, n_centers, data_radius, max_iter=None):

  centers_z = np.random.permutation(len(grid_set))[:n_centers]  # choose a random subset of centers to start
  centers_z = np.sort(centers_z)  # sort in ascending order

  # pre-compute the pairwise distances between datapoints and grid points
  dist_mat = _pairwise_data_center_distances(dataset, grid_set)

  candidate_loss = np.sum(np.min(dist_mat[:, centers_z], axis=1))  # also, we compute the initial loss L(Z^(0))

  max_iter = 1000  # TODO remove debug
  if max_iter is None:  # if not set, we fix number of iterations
    max_iter = int(np.round(100 * n_centers * np.log(dataset.shape[0] / delta)))

  eps_term = -epsilon / (8 * data_radius**2 * (max_iter+1))  # and pre-define the factor for the exponential mechanism

  candidates, candidate_losses = [], []

  print(f'starting alg3 loop with {max_iter} iterations')
  for idx in range(max_iter):
    # rint(f'alg3 iter {idx}')
    centers_z, candidate_loss = get_candidate_set(centers_z, dist_mat, n_centers, candidate_loss,
                                                  eps_term, grid_set)
    candidates.append(centers_z)
    candidate_losses.append(candidate_loss)

  candidate_losses = np.stack(candidate_losses)
  candidate_selection_weights = np.exp(eps_term * candidate_losses)
  candidate_selection_index, _, _ = select_idx_by_relative_weights(candidate_selection_weights)

  chosen_candidate = candidates[candidate_selection_index]
  return grid_set[chosen_candidate], chosen_candidate


def _pairwise_data_center_distances(dataset, grid_set):
  diff_mat = np.expand_dims(dataset.T, axis=2) @ np.expand_dims(grid_set.T, axis=1)  # (d,n,1) x (d,1,m) = (d,n,m)
  dist_mat = np.sqrt(np.sum(diff_mat**2, axis=0))  # (n, m); d_ij = dist(x_i, c_j)
  return dist_mat


def _center_switch_losses(centers, centers_to_add, dist_mat):
  # for the current set of centers, we compute the distances to all datapoints
  subset_min_dist = np.min(dist_mat[:, centers], axis=1)  # (n,)
  # for all other centers, we compute the distances to all datapoints if they were to be addded to the set
  switch_minima = np.minimum(np.expand_dims(subset_min_dist, axis=1), centers_to_add)  # (n, m-|Z|+1)
  switch_losses = np.sum(switch_minima, axis=0)  # (m-|Z|+1,)  we then compute the loss as sum of minimum distances
  return switch_losses


def select_idx_by_relative_weights(weights, random_draw=None):
  increasing_weights = np.cumsum(weights)
  if random_draw is None:
    random_draw = np.random.rand() * increasing_weights[-1]
  selection_index = np.where(random_draw < increasing_weights)[0][0]
  return selection_index, random_draw, increasing_weights


def get_candidate_set(centers_z, dist_mat, n_centers, old_loss, eps_term, grid_set):
  # in each iteration we define the current set of centers C \ Z, from which to swap in.
  aux_list = [-1] + list(centers_z) + [len(grid_set)]
  indices_not_in_z = np.concatenate(
    [np.arange(aux_list[k] + 1, aux_list[k + 1]) for k in range(n_centers + 1)])
  # rint(centers_z)
  # rint(indices_not_in_z)
  centers_not_in_z = dist_mat[:, indices_not_in_z]  # (n, m-|Z|+1)

  # we go though all center-subsets omitting one center at a time
  losses_per_idx = []
  selection_weights_per_idx = []
  total_weights_per_idx = np.zeros(n_centers)
  for z_idx in range(n_centers):
    centers_z_subset = np.concatenate([centers_z[:z_idx], centers_z[z_idx + 1:]])
    switch_losses = _center_switch_losses(centers_z_subset, centers_not_in_z, dist_mat)
    selection_weights = np.exp(eps_term * (switch_losses - old_loss))
    total_weight = np.sum(selection_weights)

    losses_per_idx.append(switch_losses)
    selection_weights_per_idx.append(selection_weights)
    total_weights_per_idx[z_idx] = total_weight

  index_out, random_choice, increasing_total_weights = select_idx_by_relative_weights(total_weights_per_idx)

  if index_out > 0:
    random_choice -= increasing_total_weights[index_out - 1]

  # find which center is swapped in
  index_in_offset, _, _ = select_idx_by_relative_weights(selection_weights_per_idx[index_out], random_choice)

  # index_out is relative to centers_z, while index_in indexes sub_excl_centers.
  # what we need is the index of center c_in = sub_excl_centers[index_in] in grid_set. since Z is sorted,
  # we can reverse the offset by going through Z and adding 1 to index_in each time it is greater than Z_i.
  index_in = index_in_offset
  for c_idx in centers_z:
    if index_in >= c_idx:
      index_in += 1
    else:
      break

  # ensure that we have recovered the right index: (this doesn't work because the latter contains distances not points..
  # assert np.linalg.norm(grid_set[index_in] - centers_not_in_z[index_in_offset]) < 1e-5

  #  rint(f'idx_out: {index_out}, z[idx_out]: {centers_z[index_out]}, idx_in: {index_in}')
  centers_z = centers_z.copy()
  centers_z[index_out] = index_in
  centers_z = np.sort(centers_z)

  candidate_loss = losses_per_idx[index_out][index_in_offset]
  return centers_z, candidate_loss


def alg4_private_clustering(dataset, epsilon, delta, n_centers, data_radius,
                            encoding_dim=None, alg2_max_iter=None, alg3_max_iter=None, alg4_max_iter=None):

  n_data, n_feat = dataset.shape

  if encoding_dim is None:
    encoding_dim = int(np.round(8 * np.log(n_data)))

  if alg4_max_iter is None:
    alg4_max_iter = int(np.round(2 * np.log(1/delta)))

  candidates = []

  print(f'starting alg4 loop with {alg4_max_iter} iterations')
  for idx in range(alg4_max_iter):
    print(f'alg4 iter {idx}')
    projection_mat = np.random.randn(n_feat, encoding_dim)
    projected_data = dataset @ projection_mat / np.sqrt(n_feat)

    grid_set = alg2_candidate(projected_data, epsilon / (6 * alg4_max_iter), delta, n_centers, data_radius,
                              alg2_max_iter)
    print(f'alg2 done. grid set size {grid_set.shape[0]}')

    chosen_candidate, _ = alg3_localswap(projected_data, grid_set, epsilon / (6 * alg4_max_iter), delta,
                                         n_centers, data_radius, alg3_max_iter)
    print(f'alg3 done.')

    candidate_centers = _release_candidate_centers(projected_data, chosen_candidate, dataset, alg4_max_iter, epsilon,
                                                   data_radius)
    candidates.append(candidate_centers)

  print(f'choosing final candidate')
  chosen_centers, l2_loss = _choose_final_release(dataset, candidates, epsilon, data_radius)
  return chosen_centers, l2_loss


def _release_candidate_centers(projected_dataset, centers, real_dataset, alg4_max_iter, epsilon, data_radius):
  # assign data to centers and compute noisy cluster sizes
  print(projected_dataset.shape, centers.shape)
  dist_mat = _pairwise_data_center_distances(projected_dataset, centers)  # (n_data, n_centers)
  closest_centers = np.argmin(dist_mat, axis=1)
  data_by_centers = [real_dataset[closest_centers == k] for k in range(centers.shape[0])]
  noisy_cluster_sizes = [k.shape[0] + np.random.laplace(scale=24*alg4_max_iter / epsilon) for k in data_by_centers]
  noisy_cluster_sizes = np.maximum(np.asarray(noisy_cluster_sizes), 1.)

  # compute noisy centroids
  noisy_centroids = []
  for idx in range(centers.shape[0]):
    data_average = np.sum(data_by_centers[idx], axis=0) / noisy_cluster_sizes[idx]
    release_noise = np.random.laplace(scale=24 * alg4_max_iter * data_radius / (epsilon * noisy_cluster_sizes[idx]))
    noisy_centroids.append(data_average + release_noise)

  return np.stack(noisy_centroids)


def _choose_final_release(dataset, candidates, epsilon, data_radius):
  relative_losses = []
  for candidate in candidates:
    dist_mat = _pairwise_data_center_distances(dataset, candidate)
    candidate_loss = np.sum(np.min(dist_mat, axis=1))
    relative_losses.append(candidate_loss)

  relative_weights = np.exp(-epsilon / (24 * data_radius**2) * np.asarray(relative_losses))
  selection_index, _, _ = select_idx_by_relative_weights(relative_weights)
  return candidates[selection_index], relative_losses[selection_index]


def test_clustering():
  pass
