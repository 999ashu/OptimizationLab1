#!/usr/bin/env python
# coding: utf-8

# # Gradient descent implementation based on golden-section search

# ## Needed Libraries import

# In[21]:


from helpers import *
import numpy as np
from scipy.optimize import OptimizeResult


# ## Gradient Descent implementations with golden-section search

# ### Implementation of golden-section search

# In[22]:


def golden_section_search(f, a, b, tol=1e-6, max_iter=1000):
    golden_ratio = (np.sqrt(5) - 1) / 2

    nfev = 0

    x1 = b - golden_ratio * (b - a)
    x2 = a + golden_ratio * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    nfev += 2

    nit = 0
    for nit in range(max_iter):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + golden_ratio * (b - a)
            f2 = f(x2)
            nfev += 1
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - golden_ratio * (b - a)
            f1 = f(x1)
            nfev += 1

        if abs(b - a) < tol:
            break

    x_min = (a + b) / 2
    f_min = f(x_min)
    nfev += 1

    result = OptimizeResult(
        x=x_min,
        fun=f_min,
        nfev=nfev,
        nit=nit + 1,
        success=abs(b - a) < tol,
        status=0 if abs(b - a) < tol else 1,
    )

    return result


# ### Gradient descent based of previously implemented line search

# In[23]:


def gd_gs(f, grad_f, x0, tol=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)

    nit = 0
    nfev = 1
    ngev = 0
    x_history = [x.copy()]

    for nit in range(max_iter):
        g = grad_f(x)
        ngev += 1

        if np.linalg.norm(g) < tol:
            break

        d = -g

        def f_along_line(alpha):
            return f(x + alpha * d)

        alpha_result = golden_section_search(f_along_line, 0, 1.0)
        alpha = alpha_result.x
        nfev += alpha_result.nfev

        x_new = x + alpha * d
        x_history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            nfev += 1
            break

        x = x_new
        nfev += 1

    fun = f(x)

    result = OptimizeResult(
        x=x,
        fun=fun,
        jac=g,
        nfev=nfev,
        njev=ngev,
        nit=nit + 1,
        success=nit < max_iter - 1 or np.linalg.norm(g) < tol,
        x_history=np.array(x_history)
    )

    return result


# ### Extraction for reuse

# In[24]:


get_ipython().system('jupyter nbconvert --to python GD_GSS.ipynb')


# ## Results

# ### Symmetrical parabola: $(x - 3)^2 + (y + 2)^2$

# In[25]:


print_output([-5, 3], gd_gs, func_sp, grad_sp, [3, -2])


# ### Rotated elliptical function: $2(x + 2)^2 + 4xy + 3(y - 4)^2$

# In[26]:


print_output([-3, -25], gd_gs, func_re, grad_re, [-18, 16], [-30, 30])


# ### Elliptical function with scale: $8(x - 3)^2 + (y + 1)^2$

# In[27]:


print_output([-5, 3], gd_gs, func_es, grad_es, [3, -1])

