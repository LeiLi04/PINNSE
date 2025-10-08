import sympy as sp
import numpy as np
#-------1.1 Hessian Matrix_symbols---------
x,y = sp.symbols('x,y')
f = x**2 + y**2


f_hessain = sp.hessian(f, (x,y))
print(f'f_hessian Matrix is {f_hessain}')

from jax import jacobian, grad , hessian
import jax.numpy as jnp
#-------1.2 Hessian Matrix_jax---------
def f(x):
  return x[0]**2+x[1]**2

J = jacobian(f)(jnp.array([1.0, 2.0]))
print(f'J_jacobian is {J}')

H = hessian(f)(jnp.array([1.0, 2.0]))
print(f'H_hessian is {H}')
#-------2.1 Hessian Matrix_newton's method---------
# 01 
x, y = sp.symbols('x y')
f = (x-1)**2+sp.Rational(1,2)*(y-2)**2+x*y+ sp.sin(y) # (x-1)^2+0.5(y-2)^2+xy+sin(y)
vars_vect = [x,y]

#02
# 2.1
gra_f_jacobian = sp.Matrix([f]).jacobian(vars_vect).T
#2.2 
gra_f_diff = sp.Matrix([sp.diff(f,v) for v in vars_vect])
#2.3
gra_f_array = sp.Matrix(sp.derive_by_array(f, vars_vect))
#2.4
hessian_f = sp.hessian(f, vars_vect)

print(f'gra_f_jacobian is \n{gra_f_jacobian}')
print(f'gra_f_diff is \n{gra_f_diff}')
print(f'gra_f_array is \n{gra_f_array}')
print(f'hessian_f is \n{hessian_f}')

#03
f_fn = sp.lambdify(vars_vect, f, 'numpy')
gra_f_jacobian_fn = sp.lambdify(vars_vect, gra_f_jacobian, 'numpy')
gra_f_diff_fn = sp.lambdify(vars_vect, gra_f_diff, 'numpy')
gra_f_array_fn = sp.lambdify(vars_vect, gra_f_array, 'numpy')
hessian_f_fn = sp.lambdify(vars_vect, hessian_f, 'numpy')

#04 单步newton's method
def newton_step(x):
  g = np.array(gra_f_jacobian_fn(x[0], x[1]), dtype = float).reshape(-1)
  H = np.array(hessian_f_fn(x[0], x[1]), dtype = float)
  # method A ;
  step_via_inv = np.linalg.inv(H)@g
  # method B ;
  step_via_solve = np.linalg.solve(H, g)

  x_next = x-step_via_solve
  return x_next, g, H, step_via_inv, step_via_solve, np.linalg.inv(H)
x0 = np.array([0.0, 0.0])
x_next_0, g_0, H_0, step_via_inv_0, step_via_solve_0, H_inverse_0 = newton_step(x0)
print(f'x_next_0 is \n {x_next_0}')
print(f'g_0 is \n {g_0}')
print(f'H_0 is \n {H_0}')
print(f'step_via_inv_0 is \n {step_via_inv_0}')
print(f'step_via_solve_0 is \n {step_via_solve_0}')
print(f'f_hessian_inverse_0 is \n {H_inverse_0}')


# 05 多步newton's Method
def newton_method(x_init, tol = 1e-8, max_iter = 50):
  x = x_init.astype(float)
  history = [x.copy()]
  for i in range(max_iter):
    g = np.array(gra_f_jacobian_fn(x[0], x[1]), dtype = float).reshape(-1)
    H = np.array(hessian_f_fn(x[0], x[1]), dtype = float)
    step = np.linalg.solve(H, g)
    x = x-step
    history.append(x.copy())
    if np.linalg.norm(step, ord =2) < tol:
      break
  return x, i+1, np.array(history)


x_star, iter_num, history = newton_method(x0)
eigvals = np.linalg.eigvals(hessian_f_fn(x_star[0], x_star[1]))

print('\n-- Newton iterations--')
print(f'x_star is \n {x_star}')
print(f'iter_num is \n {iter_num}')
print(f'history is \n {history}')
print(f'f(x*) = {f_fn(*x_star)}')
print(f'Eigenvalues of H()x: {eigvals}')
if np.all(eigvals>0):
  print('H在x*处为正定矩阵 -> x*是局部最小点值')
elif np.all(eigvals<0):
  print('H在x*处为负定矩阵 -> x*是局部最大点值')
else:
  print('H在X*既有正特征值， 又有负特征值， 所以是saddle point')