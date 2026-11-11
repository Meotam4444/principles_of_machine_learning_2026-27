# Lecture 02 — Least squares regression
*Prof. Christian Bauckhage, Principles of Machine Learning (79 slides)*

## Overview
1. Least squares for regression (setting the stage: predicting weight from height)
2. Univariate linear regression
3. Univariate non-linear regression (polynomial, RBF)
4. Multivariate linear regression
5. Multivariate non-linear regression
6. Remarks on numerics / notation / terminology
7. Summary

**Historical note:** least squares is the "grandfather" of all model fitting. It goes back to Gauss (1795, at age 18; used in 1801 to predict where the dwarf planet Ceres would reappear), Legendre (published 1805) and Adrain (1808).
**Didactic note:** LSQ is a *non-iterative* learning task. The parameters come from a single direct calculation (a closed form), which is *exceptional* in ML.

Regression settings covered: {univariate, multivariate} × {linear, non-linear}.

---

## 1. Three aspects of every model fitting task
- **Modeling**: which function $f$? Which inputs, outputs and parameters?
- **Representation**: how do we represent inputs, outputs and parameters so that computation is easy and efficient?
- **Parametrization**: how do we estimate the optimal parameters, i.e. which learning algorithm?

These are intertwined and must be considered jointly.

## 2. Univariate linear regression
- Data $D=\{(x_j,y_j)\}_{j=1}^n\subset\mathbb R^2$, model $f(x\mid w_0,w_1)=w_0+w_1x$ (intercept and slope).
- Abstract parameter notation: $\theta=\{w_0,w_1\}$, $f(x\mid\theta)\equiv f_\theta(x)$.
- **Residuals**: $\epsilon_j=f_\theta(x_j)-y_j$. The goal is $\forall j:\epsilon_j\approx0$, which we turn into "the average of $\epsilon_j^2$ is $\approx0$".
- **Mean squared error**:
$$E(\theta\mid f,D)=\frac1n\sum_{j=1}^n\big(f_\theta(x_j)-y_j\big)^2$$
- The factor $1/n$ changes the *value* of the minimum but not the *minimizer*, so we may minimize the **least squares loss**
$$L(w_0,w_1)=\sum_{j=1}^n(w_0+w_1x_j-y_j)^2 .$$
- $E$ is continuous and convex (a sum of squares), so it has a **unique minimizer** (strictly, this needs a non-degenerate $\Phi$).
- **Loss functions** $L(\theta\mid f_\theta,D)$ are central to ML: the smaller the loss, the better the fit.

### Vectorization
- $w_0\cdot1+w_1x_j=\mathbf w^\top\boldsymbol\varphi_j$ with $\mathbf w=[w_0,w_1]^\top$ and $\boldsymbol\varphi_j=[1,x_j]^\top$.
- Then $L(\mathbf w)=\sum_j(\boldsymbol\varphi_j^\top\mathbf w-y_j)^2=\|\mathbf v-\mathbf y\|^2$, a **squared Euclidean distance**.
- **Feature matrix** $\Phi=[\boldsymbol\varphi_1\cdots\boldsymbol\varphi_n]\in\mathbb R^{M\times n}$ (feature vectors as *columns*), so
$$\hat{\mathbf w}=\arg\min_{\mathbf w}\|\Phi^\top\mathbf w-\mathbf y\|^2 .$$
- **Feature map** $\varphi:\mathbb R\to\mathbb R^2$, $x\mapsto[1,x]^\top$, *lifts* the data into a higher-dimensional space.

### Derivation ("derive and equate to zero")
$$L=(\Phi^\top\mathbf w-\mathbf y)^\top(\Phi^\top\mathbf w-\mathbf y)=\mathbf w^\top\Phi\Phi^\top\mathbf w-2\mathbf w^\top\Phi\mathbf y+\mathbf y^\top\mathbf y$$
$$\nabla L=2\Phi\Phi^\top\mathbf w-2\Phi\mathbf y\overset!=0\ \Rightarrow\ \Phi\Phi^\top\mathbf w=\Phi\mathbf y\quad\text{(normal equations)}$$
$$\boxed{\hat{\mathbf w}=(\Phi\Phi^\top)^{-1}\Phi\,\mathbf y}$$
- Optimality: $\|\Phi^\top\hat{\mathbf w}-\mathbf y\|^2\le\|\Phi^\top\mathbf w-\mathbf y\|^2$ for all $\mathbf w$.
- **Prediction**: $y=\varphi(x)^\top\hat{\mathbf w}$. Example: a 200 cm person is predicted at about 94.5 kg.

### General principle (slide 37)
Data $D\subset\mathbb R^m\times\mathbb R^t$, model class $\{f:\mathbb R^m\to\mathbb R^t\}$ with $p$ parameters, loss $L:\mathbb R^p\to\mathbb R$, feature map $\varphi:\mathbb R^m\to\mathbb R^M$. The loss comes from statistics; the optimization uses linear algebra and calculus. In ML language: *train* on training data, *apply* to application data.

## 3. Univariate non-linear regression
### Polynomial regression
- $f(x\mid\mathbf w)=\sum_{i=0}^d w_ix^i$ with feature map $\varphi(x)=[x^0,x^1,\dots,x^d]^\top\in\mathbb R^{d+1}$.
- The model is **non-linear in $x$ but linear in $\mathbf w$**, so the problem is *algebraically identical* to linear LSQ and has the same solution $\hat{\mathbf w}=(\Phi\Phi^\top)^{-1}\Phi\mathbf y$.
- **Vandermonde matrix**: $V_{rc}=v_r^{c-1}$. The design matrix $\Phi^\top$ of polynomial regression is a Vandermonde matrix. (These are notoriously ill-conditioned for large $d$.)
- In general: fit a **linear combination of basis functions**, $f(x)=\sum_i w_if_i(x)$.

### RBF regression
- **Radial basis function**: $\phi_c(\mathbf x)=\phi(d(\mathbf x,\mathbf c))$ depends only on the distance to a center $\mathbf c$.
- Common choices ($\beta>0$ is the scale): $e^{-\|\mathbf x-\mathbf c\|/\beta}$, $e^{-\|\mathbf x-\mathbf c\|^2/\beta}$ (**Gaussian, by far the most common**), $1-\frac{\|\mathbf x-\mathbf c\|^2}{\|\mathbf x-\mathbf c\|^2+\beta}$.
- Center one RBF at each sample: $f(\mathbf x\mid\mathbf w)=\sum_{j=1}^n\phi(\mathbf x,\mathbf x_j)w_j$.
- Requiring $y_i=f(x_i)$ gives $\mathbf y=\Phi\mathbf w$ with $\Phi_{ij}=\phi(x_i,x_j)$, which is **square ($n\times n$) and symmetric**.
- So $\hat{\mathbf w}=\Phi^{-1}\mathbf y$, but often it is better to use LSQ, $(\Phi\Phi^\top)^{-1}\Phi\mathbf y$. (The plotted result used a "modified version", i.e. regularization, covered later.)

### Principles
- **Different model classes give different results and predictions.** At $x=200$: linear 94.5 kg, polynomial 61.3 kg, RBF 62.7 kg. Some predictions are more plausible than others.
- **LSQ is model-agnostic**: it fits almost any model and never checks whether the model is appropriate (e.g. a line fitted to parabolic data). Choosing the model is the job of a human expert (or an AI agent).

## 4. Multivariate linear regression
- $\mathbf x_j\in\mathbb R^m$, $y_j\in\mathbb R$, $f(\mathbf x\mid\mathbf w)=w_0+\sum_{i=1}^m w_ix_i$ (for $m=2$ a plane).
- $\varphi(\mathbf x)=[1,x_1,\dots,x_m]^\top\in\mathbb R^{m+1}$, $\Phi\in\mathbb R^{(m+1)\times n}$, and the same solution formula.

## 5. Multivariate non-linear regression
- Example: the bivariate polynomial $f(\mathbf x)=\sum_{i=0}^2\sum_{k=0}^2 w_{ik}x_1^ix_2^k$ with $\varphi:\mathbb R^2\to\mathbb R^9$ (all monomials $x_1^ix_2^k$). Same recipe.

## 6. Numerics, notation, terminology
- **Warning:** the pen-and-paper solution assumes $\Phi\Phi^\top$ is invertible. It may be rank-deficient or singular.
- The closed form is mostly of "theoretical interest". LAPACK-based libraries compute **minimum-norm LSQ solutions via QR or SVD** (→ Exercise 1.1).
- **Notation conventions:**
  - $n$ = number of data vectors, $m$ = data dimension, $M$ = feature dimension
  - data matrix $X\in\mathbb R^{m\times n}$ (columns $\mathbf x_j$), design matrix $X^\top\in\mathbb R^{n\times m}$ (rows $\mathbf x_j^\top$)
  - feature matrix $\Phi\in\mathbb R^{M\times n}$, feature design matrix $\Phi^\top\equiv\Psi\in\mathbb R^{n\times M}$
- The literature writes $M\mathbf x=\mathbf v$ with $\mathbf x=(M^\top M)^{-1}M^\top\mathbf v$. With $\Psi=\Phi^\top$ we get $\mathbf w=(\Psi^\top\Psi)^{-1}\Psi^\top\mathbf y=(\Phi\Phi^\top)^{-1}\Phi\mathbf y$, which is identical. The course prefers the $\Phi$ form for later links to statistics.
- Shorthand: $\varphi(\mathbf x_j)\equiv\boldsymbol\varphi_j$, $\varphi(\mathbf x)\equiv\boldsymbol\varphi_{\mathbf x}$.

## 7. Summary
- LSQ regression is supervised ML: minimize a quadratic loss to fit $y=f(\mathbf x\mid\mathbf w)=\sum_{i=0}^{M-1}w_if_i(\mathbf x)=\mathbf w^\top\boldsymbol\varphi_{\mathbf x}$.
- $\hat{\mathbf w}=\arg\min\|\Phi^\top\mathbf w-\mathbf y\|^2=(\Phi\Phi^\top)^{-1}\Phi\mathbf y$. Different models mean different feature maps.
- Prediction: $y=\boldsymbol\varphi_{\mathbf x}^\top\hat{\mathbf w}$.
- Pseudocode: `TRAIN(X, y): Φ = φ(X); return (ΦΦᵀ)⁻¹Φy` and `APPLY(X, ŵ): Φ = φ(X); return Φᵀŵ`.
- Neural network view: $y=h(a)$ with $a=\boldsymbol\varphi_{\mathbf x}^\top\hat{\mathbf w}$ and $h=\mathrm{id}$. **LSQ regression trains a very simple neural network** (input → fixed basis functions → weighted sum → identity activation).
