import numpy as np

np.set_printoptions(precision=3, suppress=True)

def make_near_singular_spd(eigs):
    """
    Create a symmetric (intended SPD) matrix with prescribed eigenvalues.
    If some eigenvalues are extremely tiny, finite precision may make it fail Cholesky.
    """
    n = len(eigs)
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)          # random orthogonal
    S = (Q * eigs) @ Q.T            # Q diag(eigs) Q^T
    S = (S + S.T) / 2.0             # symmetrize
    return S

# Example 1: A matrix with one *very* small eigenvalue
eigs = np.array([1.0, 0.5, 1e-14])      # ill-conditioned
S = make_near_singular_spd(eigs)

print("=== Example 1: Near-singular SPD (before jitter) ===")
print("Eigenvalues (target):", eigs)
print("Eigenvalues (numerical):", np.linalg.eigvalsh(S))

# Try Cholesky without jitter
try:
    L = np.linalg.cholesky(S)
    print("Cholesky succeeded WITHOUT jitter. (Sometimes it does, but it's risky.)")
except np.linalg.LinAlgError as e:
    print("Cholesky FAILED without jitter:", str(e))

# Add jitter
jitter = 1e-6
S_jit = S + jitter * np.eye(3)

print("\n=== After adding jitter ({} * I) ===".format(jitter))
print("Eigenvalues (numerical):", np.linalg.eigvalsh(S_jit))

# Try Cholesky with jitter
try:
    L_jit = np.linalg.cholesky(S_jit)
    print("Cholesky SUCCEEDED with jitter. L lower-triangular:")
    print(L_jit)
except np.linalg.LinAlgError as e:
    print("Cholesky FAILED even with jitter:", str(e))
