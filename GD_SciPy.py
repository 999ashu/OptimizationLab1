#!/usr/bin/env python
# coding: utf-8

# # Gradient descent implementations in SciPy Optimize.

# ## Needed Libraries import

# In[32]:


from helpers import *
import numpy as np
import scipy.optimize as opt


# ## Gradient Descent implementations with SciPy Optimize

# ### Minimize function provides interface to optimization algorithms, which are using dynamic learning rate.

# In[33]:


def gd_dlr_opt(func, grad, init, max_iter=1000, tol=1e-6):
    x_history = [np.array(init, dtype=float)]

    def callback(xk):
        x_history.append(np.array(xk, dtype=float))

    result = opt.minimize(func, x0=init, jac=grad, method='BFGS', tol=tol, callback=callback,
                          options={'maxiter': max_iter})

    result.x_history = np.array(x_history)

    return result


# ### SciPy optimize supplies minimize_scalar function for line search, which can use different algorithms heuristics. We can use it to implement gradient descent with line search.

# In[34]:


def gd_line_search_opt(func, grad, init, max_iter=1000, tol=1e-6):
    x = np.array(init, dtype=float)

    x_history = [x.copy()]
    nit = 0
    nfev = 0

    for i in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        def line_search_func(alpha):
            nonlocal nfev
            nfev += 1
            return func(x - alpha * g)

        res = opt.minimize_scalar(line_search_func, method='golden', tol=tol)

        x = x - res.x * g

        x_history.append(x.copy())
        nit += 1
        nfev += res.nfev + 1

        if np.linalg.norm(x_history[-1] - x_history[-2]) < tol:
            break

    return opt.OptimizeResult(
        x=x,
        fun=func(x),
        nit=nit + 1,
        nfev=nfev,
        njev=nit + 1,
        x_history=np.array(x_history),
    )


# ### Extraction for reuse

# In[35]:


# !jupyter nbconvert --to python GD_SciPy.ipynb


# ## Results

# ### Symmetrical parabola: $(x - 3)^2 + (y + 2)^2$

# In[36]:


print_output([-5, 3], gd_dlr_opt, func_sp, grad_sp, [3, -2])
print_output([-5, 3], gd_line_search_opt, func_sp, grad_sp, [3, -2])


# ### Rotated elliptical function: $2(x + 2)^2 + 4xy + 3(y - 4)^2$

# In[37]:


print_output([-3, -25], gd_dlr_opt, func_re, grad_re, [-18, 16], grid=[-30, 30])
print_output([-3, -25], gd_line_search_opt, func_re, grad_re, [-18, 16], grid=[-30, 30])


# ### Elliptical function with scale: $8(x - 3)^2 + (y + 1)^2$

# In[38]:


print_output([-5, 3], gd_dlr_opt, func_es, grad_es, [3, -1])
print_output([-5, 3], gd_line_search_opt, func_es, grad_es, [3, -1])

