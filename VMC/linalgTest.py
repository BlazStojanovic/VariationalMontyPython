"""
Disclaimer, this is not my code. It was authored by 
Dominic Jack, who posted it on his blog, https://jackd.github.io/posts/generalized-eig-jvp/

"""


from Utilities.Linalg.eigh import symmetrize, eigh, standardize_angle
import jax.numpy as jnp
import numpy as np

import jax.test_util as jtu
import scipy.linalg

jnp.set_printoptions(3)
rng = np.random.default_rng(0)

n = 5
is_complex = False


def make_spd(x):
    n = x.shape[0]
    return symmetrize(x) + n * jnp.eye(n)


def get_random_square(rng, size, is_complex=True):
    real = rng.uniform(size=size).astype(np.float32)
    if is_complex:
        return real + rng.uniform(size=size).astype(np.float32) * 1j
    return real


a = make_spd(get_random_square(rng, (n, n), is_complex=is_complex))
b = make_spd(get_random_square(rng, (n, n), is_complex=is_complex))

vals, vecs = eigh(a, b)
# ensure solution satisfies the problem
np.testing.assert_allclose(a @ vecs, b @ vecs @ jnp.diag(vals), atol=1e-5)
# ensure vectors are orthogonal w.r.t b
np.testing.assert_allclose(vecs.T.conj() @ b @ vecs, jnp.eye(n), atol=1e-5, rtol=1e-5)
# ensure eigenvalues are ascending
np.testing.assert_array_less(vals[:-1], vals[1:])
jtu.check_grads(eigh, (a, b), 2, modes=["fwd"])

# ensure values consistent with scipy
vals_sp, vecs_sp = scipy.linalg.eigh(a, b)
print("scipy")
print(vecs_sp)
print("this work")
print(vecs)
np.testing.assert_allclose(vals, vals_sp, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(vecs, standardize_angle(vecs_sp, b), rtol=1e-4, atol=1e-5)
print("success")