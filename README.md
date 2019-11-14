<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

### 0. Function

$$
y=exp(ax^2+bx+c)
$$



### 1. Auto Derivatives



### 2. Numeric Derivatives

> In some cases, its not possible to define a templated cost functor, for example when the evaluation of the residual involves a call to a library function that you do not have control over.  In such a situation, numerical differentiation can be used
>
> Generally speaking we recommend automatic differentiation instead of numeric differentiation. The use of C++ templates makes automatic differentiation efficient, whereas numeric differentiation is **expensive**, **prone to numeric errors**, and **leads to slower convergence**.



### 3. Analytic Derivatives

> In some cases, using automatic differentiation is not possible. For example, it may be the case that it is more **efficient** to compute the derivatives in closed form instead of relying on the chain rule used by the automatic differentiation code.

