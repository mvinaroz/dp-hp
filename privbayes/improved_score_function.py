import numpy as np


def get_parent_set_probs(x_data, parent_set_data):
  # x_data is a (n,) binary vector
  # parent_set_data is a (n x k) matrix of attributes for all data samples
  n_data, k_attr = parent_set_data.shape
  parent_set_data = parent_set_data.copy()
  parent_set_counts = np.zeros((2**k_attr, 2))
  for idx in range(k_attr):
    parent_set_data[:, idx] *= 2**idx  # in ascending order, each attribute contributes a power of two to the binary id

  binary_ids = np.sum(parent_set_data, axis=1)

  for idx in range(n_data):
    parent_set_counts[binary_ids[idx], x_data[idx]] += 1

  return parent_set_counts / n_data


def compute_f(parent_set_probs):
  # parent_set_probs is a (2^k x 2) Matrix where index [i, 0] denotes the relative frequency of setting of
  # pi = binary(i), given that X=0, and [i, 0] analogously
  n_assignments, _ = parent_set_probs.shape
  # c_set is the set of non-dominated C(i,a,b).
  # We denote this with (K_0,K_1) tuples, since parent_set_probs already has the 1/n factor.
  # It starts with (0,0) as the only item
  c_set_i = np.zeros((1, 2))

  for idx in range(n_assignments):
    # using recursive definition in eq (10), find all possible sets for i+1
    c_set_0 = c_set_i.copy()
    c_set_1 = c_set_i.copy()
    c_set_0[:, 0] += parent_set_probs[idx, 0]
    c_set_1[:, 1] += parent_set_probs[idx, 1]

    # given that c_set_i contained no dominated C, neither of the sets 0,1 can contain pairs where C dominates C'
    # however C_0 in c_set_0 can dominate C_1 in c_set_1 and vice versa.
    dom_0 = np.zeros(c_set_i.shape[0])  # compute vector that is 1 at idx=j if C_j is dominated and 0 otherwise
    dom_1 = np.zeros(c_set_i.shape[0])

    for jdx in range(c_set_i.shape[0]):
      dom_0 += np.prod((c_set_0 - c_set_1[jdx, :]) <= 0, axis=1)   # prod is 1 where both entries are <= C_1_j
      dom_1 += np.prod((c_set_1 - c_set_0[jdx, :]) <= 0, axis=1)

    c_set_0 = c_set_0[np.where(dom_0 == 0)]  # shrink c_set_0 to entries that are not dominated
    c_set_1 = c_set_1[np.where(dom_1 == 0)]

    c_set_i = np.concatenate([c_set_0, c_set_1])
    print(f'len c:{c_set_i.shape[0]}')

  # now compute the final score as - min_C relu(0.5 - K_0) + relu(0.5 - K_1)
  positive_diffs = np.maximum(0, 0.5 - c_set_i)
  print(f'pos diffs:{positive_diffs}')
  final_score = - np.min(np.sum(positive_diffs, axis=1))
  return final_score


def compute_r(parent_set_probs):
  p_x = np.sum(parent_set_probs, axis=0, keepdims=True)
  p_parent = np.sum(parent_set_probs, axis=1, keepdims=True)
  p_bar = p_parent @ p_x
  score = 0.5 * np.sum(np.abs(parent_set_probs - p_bar))
  return score


def compute_r_fast(x_data, parent_set_data):
  # there is probably some room for optimization here
  n_data, k_attr = parent_set_data.shape

  parent_set_counts = np.zeros((2**k_attr, 2))
  bin_idx_mat = np.asarray([[2**idx for idx in range(k_attr)]])

  binary_ids = np.sum(parent_set_data * bin_idx_mat, axis=1)

  for idx in range(n_data):
    parent_set_counts[binary_ids[idx], x_data[idx]] += 1

  parent_set_probs = parent_set_counts / n_data
  p_x = np.sum(parent_set_probs, axis=0, keepdims=True)
  p_parent = np.sum(parent_set_probs, axis=1, keepdims=True)
  p_bar = p_parent @ p_x
  score = 0.5 * np.sum(np.abs(parent_set_probs - p_bar))
  return score

def basic_test():
  x_data = np.asarray([1, 1, 1, 1,
                       0, 0, 0, 0, 0, 0])
  p_data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1],
                       [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

  p_probs = get_parent_set_probs(x_data, p_data)
  print(p_probs)
  score_f = compute_f(p_probs)
  print(score_f)
  score_r = compute_r(p_probs)
  print(score_r)
  score_r = compute_r_fast(x_data, p_data)
  print(score_r)


if __name__ == '__main__':
  basic_test()
