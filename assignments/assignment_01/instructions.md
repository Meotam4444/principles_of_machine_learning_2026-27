# Assignment 01 — Fun with Least Squares Optimization
Source: `lectures/lecture_01/poml-ex-01.pdf` (20 pages). Relevant lectures: **01, 02**.

## How to approach the tasks (from the sheet)
- Work in teams (communication skills).
- Be ambitious and try to solve the problems yourself. With an AI, you don't have to code everything, but **you must figure out what needs to be coded** so you can give your AI companion specific instructions.

**Questions while working on a task:**
- How could we approach this task?
- Why should we consider this approach?
- Are there any other or better approaches?

**Questions after solving a task:**
- What are the main takeaways from this task?
- Why might the instructor have set this task?
- What could he have wanted us to learn by solving it?

---

## Problems — overview and teaching notes

### 1.1 Beware of numerical instabilities (everyone does this alone!)
- OLS with $X^\top=\begin{bmatrix}1&-1\\0&10^{-5}\\0&0\end{bmatrix}$ and $\mathbf y=[0,10^{-5},0]^\top$. The exact solution is $\mathbf w^\star=[1,1]^\top$ with zero residual.
- 1.1.1: run the pen-and-paper code `inv(X @ X.T) @ X @ y` and marvel at the output.
- Background: thin QR, $X^\top=Q_1R_1$, so $\mathbf w^\star=R_1^{-1}Q_1^\top\mathbf y$. `np.linalg.qr` returns $Q_1,R_1$ by default.
- 1.1.2: implement the QR solution. 1.1.3: use `la.lstsq`. Does it use eq. (4)?
- 1.1.4: reflect on floats vs. reals, machine epsilon, very large or very small numbers, $XX^\top$ by hand vs. on a computer, and blind trust in code from the internet.
- *Teaching focus:* $1+10^{-10}$ rounds to $1$ in float64 because $10^{-10}$ is smaller than the spacing of floats near 1. Forming $XX^\top$ **squares the condition number** ($\kappa(XX^\top)=\kappa(X)^2$). QR and SVD avoid forming $XX^\top$. Link to lecture 02, slide 67.

### 1.2 Learning cellular automaton updates
- Elementary CA: $x_j[t+1]=f(x_{j-1}[t],x_j[t],x_{j+1}[t])$ with $f:\{0,1\}^3\to\{0,1\}$, giving 256 rules (Wolfram numbering: the target vector read bottom to top as binary).
- 1.2.1: simulate and plot any rule $r\in\{0..255\}$ ($n_c=127$ cells, $n_r=64$ rows, random init); run rules 60 and 102. (Boundary handling must be decided, e.g. periodic.)
- Bipolar substitution $x\mapsto(-1)^x$, i.e. $0\to+1$, $1\to-1$.
- 1.2.2: LSQ with the raw $8\times3$ design matrix (no bias, linear in $\mathbf x$) for rules 110 and 126; print $\mathbf y$ and $\hat{\mathbf y}$. *Why does this hardly make sense?* A linear model cannot represent non-linear Boolean functions; there is not even a bias term.
- Boolean Fourier expansion: $f(\mathbf x)=\sum_{S\in2^{\mathcal I}}w_S\prod_{j\in S}x_j$ with parity basis functions; $\varphi:\{\pm1\}^n\to\{\pm1\}^{2^n}$.
- 1.2.3: implement $\varphi$ for general $n$ (hint: `itertools.combinations`), ordered as $[1,x_1,x_2,x_3,x_1x_2,x_1x_3,x_2x_3,x_1x_2x_3]$.
- 1.2.4: build the $8\times8$ matrix $\Phi^\top$, solve LSQ, compare $\hat{\mathbf y}$ with 1.2.2. *What's the price?* Feature dimension $2^n$ grows exponentially, as many parameters as data points (pure interpolation / memorization, nothing left to generalize to).
- *Teaching focus:* feature maps that lift data to higher dimensions make non-linear problems linear (link to the lecture 02 feature-map idea). $\Phi^\top$ here is a Hadamard (Walsh) matrix with orthogonal columns.

### 1.3 Estimating fractal dimensions in images (box counting)
- Binarize (use the **given** `binarize()` snippet exactly, so results are deterministic), use scales $s_l=2^{-l}$ for $l=1..L-2$, count the boxes $n_l$ that contain foreground, then fit $D\log(1/s_l)+b=\log n_l$ by LSQ. The slope $D$ is the fractal dimension.
- Images: `imageTree.png`, `imageBolt.png` (**not yet in the repo**). Which has the higher $D$?
- Imports: `imageio.v3 as iio`, `scipy.ndimage as img`. Read with `iio.imread(f, mode='L').astype(float)`.
- *Teaching focus:* a real application of plain univariate linear regression after a log-log transform. Box counting is efficient via reshaping.

### 1.4 Fitting exponential functions to data (METR data)
- Model $y=A e^{Bx}$. Take logs: $\log y=a+Bx$ with $a=\log A$.
- 1.4.1: naive LSQ in log space. 1.4.2: weighted LSQ $\sum_j y_j(\log y_j-a-Bx_j)^2$. Compare.
- Data: `dataMETR202606.csv` (columns `day`, `avg`; **not yet in the repo**), read with pandas.
- *Teaching focus:* linearization by a transform, and the fact that the transform changes what the loss emphasizes (bias). Weighted LSQ: $\hat{\mathbf w}=(\Phi W\Phi^\top)^{-1}\Phi W\mathbf y$ with $W=\mathrm{diag}(y_j)$. Also a doubling-time interpretation, $\ln 2/B$.

### 1.5 Working with barycentric coordinates
- Triangle $\Delta(\mathbf v_1,\mathbf v_2,\mathbf v_3)\subset\mathbb R^2$ with points $\mathbf p$ (inside) and $\mathbf q$ (outside). Coordinates must be read off the figure on p. 17/19 (to be confirmed with the student).
- Constraints: $\mathbf w\ge0$, $\mathbf 1^\top\mathbf w=1$. Augmented system $X=\begin{bmatrix}V\\\mathbf 1^\top\end{bmatrix}\in\mathbb R^{3\times3}$, $\mathbf y(\mathbf p)=[\mathbf p;1]$.
- 1.5.1: solve LSQ for $\mathbf p$ and for $\mathbf q$. For $\mathbf q$ the sum-to-one constraint holds but **non-negativity is violated**; $V\mathbf w^\star$ still reproduces $\mathbf q$ exactly (an affine, not convex, combination).
- 1.5.2: closest point $\hat{\mathbf q}=\arg\min_{\mathbf x\in\Delta}\|\mathbf q-\mathbf x\|^2$, a *constrained* problem (projection onto the simplex, NNLS, edge projections, or `scipy.optimize`). Find the barycentric coordinates of $\hat{\mathbf q}$ (expect at least one zero, since $\hat{\mathbf q}$ lies on an edge or vertex).
- *Teaching focus:* how to encode equality constraints in LSQ by augmenting the matrix, and why inequality constraints need more (preparation for convexity constraints later in the course, e.g. archetypal analysis).

## Notes for me (tutor)
- The assignment PDF is stored in `lectures/lecture_01/`, not in `assignments/`.
- Data files for 1.3 and 1.4 are needed before coding.
- Code goes in `assignments/assignment_01/code/`. Takeaways go in `knowledge_base/01/takeaways_01.md`.
- Don't hand out solutions; use the guiding questions above to probe the student.
