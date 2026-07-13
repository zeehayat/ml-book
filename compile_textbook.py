import re
from pathlib import Path
import markdown
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source_material"
OUTPUT_FILE = ROOT / "regression_guide.html"

# Group chapters into Sections for structural hierarchy
# Section I: Chapters 0 & 1
# Section II: Chapters 2 & 3
# Section III: Chapters 4 & 5
# Section IV: Chapters 6 & 7
# Section V: Chapters 8 & 9

SECTION_INTRODUCTIONS = {
    1: {
        "assumptions": "Assumes absolute beginner status in mathematical notation and programming, requiring only basic arithmetic comfort.",
        "deliverable": "Build a predictive linear and logistic regression model from scratch in pure Python, analyze residuals, and map linear decision boundaries.",
        "continuation": "Acts as the foundation of the Zero-to-Research sequence, establishing arithmetic symbols and basic coding execution prior to advanced multi-dimensional matrix operations."
    },
    2: {
        "assumptions": "Requires comfort with OLS residual diagnostics, basic derivative chains, and 1D vector memory addressing.",
        "deliverable": "Implement a Poisson count GLM with exposure offsets, and optimize it using a custom strided tensor class with custom memory layouts.",
        "continuation": "Continues from Section I's simple linear models, expanding to non-normal error distributions and hardware-level memory representation."
    },
    3: {
        "assumptions": "Requires familiarity with strided tensors, basic partial derivatives, and matrix multiplication shapes.",
        "deliverable": "Build a modular, tape-based multivariable reverse-mode automatic differentiation engine and train a continuous parameter optimizer using Newton's method.",
        "continuation": "Continues from Section II's strided layout and Poisson IRLS updates, scaling to general computational graph gradients and multidimensional loss surface optimization."
    },
    4: {
        "assumptions": "Requires comfort with gradients, linear algebra, and basic probability.",
        "deliverable": "Build a dual SVM with Mercer kernels and a recursive decision tree classifier, and analyze high-dimensional data projected onto maximum variance components via SVD.",
        "continuation": "Continues from Section III's continuous optimization algorithms, moving to non-linear spaces, decision trees, and unsupervised dimension reduction."
    },
    5: {
        "assumptions": "Requires understanding of PCA/SVD projections, optimization, and chain-rule derivatives.",
        "deliverable": "Implement spectral clustering on a graph Laplacian, build a modular neural network backpropagation engine (Linear, ReLU, Sigmoid, CrossEntropyLoss) from scratch in NumPy, and verify gradient correctness using numerical finite differences.",
        "continuation": "Continues from Section IV's linear PCA embeddings, expanding to non-Euclidean graph topologies and end-to-end parametric representation learning."
    }
}

SECTION_RESOURCES = {
    1: """<li><a href="https://www.statlearning.com/" target="_blank">An Introduction to Statistical Learning</a> (James et al.) — Seminal textbook for regression and classification.</li>
<li><a href="https://wesmckinney.com/book/" target="_blank">Python for Data Analysis</a> (Wes McKinney) — Practical guide to data manipulation in Python.</li>""",
    2: """<li><a href="https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606" target="_blank">Generalized Linear Models</a> (McCullagh & Nelder) — The definitive mathematical text on GLMs.</li>
<li><a href="https://lwn.net/Articles/250967/" target="_blank">What Every Programmer Should Know About Memory</a> (Ulrich Drepper) — Crucial reference for computer memory and strides.</li>""",
    3: """<li><a href="https://arxiv.org/abs/1502.05767" target="_blank">Automatic Differentiation in Machine Learning: A Survey</a> (Baydin et al.) — Comprehensive overview of autograd mechanics.</li>
<li><a href="https://web.stanford.edu/~boyd/cvxbook/" target="_blank">Convex Optimization</a> (Stephen Boyd) — Seminal reference text for convex optimization.</li>""",
    4: """<li><a href="https://www.statlearning.com/" target="_blank">The Elements of Statistical Learning</a> (Hastie et al.) — Detailed coverage of SVMs, kernels, and decision trees.</li>
<li><a href="https://arxiv.org/abs/1404.1100" target="_blank">Singular Value Decomposition Tutorial</a> (Shlens) — Intuitive guide to SVD and PCA.</li>""",
    5: """<li><a href="https://arxiv.org/abs/0711.0189" target="_blank">A Tutorial on Spectral Clustering</a> (von Luxburg) — The definitive reference on graph Laplacians and clustering.</li>
<li><a href="https://www.deeplearningbook.org/" target="_blank">Deep Learning</a> (Goodfellow et al.) — Standard textbook on deep neural architectures.</li>"""
}

SECTION_TESTS = {
    1: {
        "content": """<p><strong>1. Multiple Choice:</strong> What does R-squared measure in OLS?</p>
<ul>
  <li>A) The slope of the regression line.</li>
  <li>B) The proportion of variance explained by the model.</li>
  <li>C) The absolute sum of squared residuals.</li>
  <li>D) The probability that the slope is non-zero.</li>
</ul>
<p><strong>2. Coding Challenge:</strong> Write a python function to compute the mean squared error (MSE) of predictions <code>y_pred</code> against true labels <code>y_true</code> using only Python loops.</p>
<p><strong>3. Conceptual Essay:</strong> Explain the key differences between covariance and correlation.</p>""",
        "answers": """<p><strong>1. Multiple Choice Answer:</strong> B</p>
<p><em>Explanation:</em> R-squared is defined as 1 - (SS_res / SS_tot), which represents the proportion of total variance in the dependent variable that is explained by the independent variables in the model.</p>
<p><strong>2. Coding Challenge Answer:</strong></p>
<pre><code class="language-python">def compute_mse(y_true, y_pred):
    n = len(y_true)
    squared_errors = [(y_true[i] - y_pred[i]) ** 2 for i in range(n)]
    return sum(squared_errors) / n</code></pre>
<p><strong>3. Conceptual Essay Answer:</strong></p>
<p>Covariance measures the directional relationship between two variables (positive, negative, or near-zero), but its scale is unbounded and depends on the units of the variables. Correlation scales covariance to a dimensionless range between -1.0 and 1.0 by dividing the covariance by the product of the standard deviations of the two variables, representing the strength of the linear relationship.</p>"""
    },
    2: {
        "content": """<p><strong>1. Multiple Choice:</strong> Which exponential family distribution is canonical for count regression data?</p>
<ul>
  <li>A) Normal</li>
  <li>B) Binomial</li>
  <li>C) Poisson</li>
  <li>D) Gamma</li>
</ul>
<p><strong>2. Coding Challenge:</strong> For a tensor of shape (2, 3, 4) in F-contiguous layout, calculate its strides vector using a short Python snippet.</p>
<p><strong>3. Conceptual Essay:</strong> Explain why transposing a strided tensor is a zero-copy operation.</p>""",
        "answers": """<p><strong>1. Multiple Choice Answer:</strong> C</p>
<p><em>Explanation:</em> Poisson distribution is the standard exponential family distribution used to model counts or frequencies of events occurring in a fixed interval.</p>
<p><strong>2. Coding Challenge Answer:</strong></p>
<pre><code class="language-python">shape = (2, 3, 4)
# For F-contiguous (column-major), strides increase from left to right:
strides = []
current_stride = 1
for dim in shape:
    strides.append(current_stride)
    current_stride *= dim
print("Strides:", strides)  # [1, 2, 6]</code></pre>
<p><strong>3. Conceptual Essay Answer:</strong></p>
<p>A strided tensor stores its elements in a flat, contiguous 1D memory buffer. The multidimensional structure is represented entirely by metadata (shape, strides, and base offset). Transposing simply swaps the strides and dimensions of the selected axes in the metadata, requiring no elements in the underlying data buffer to be copied, moved, or reordered.</p>"""
    },
    3: {
        "content": """<p><strong>1. Multiple Choice:</strong> What is the computational complexity of reverse-mode autograd for N inputs and M outputs?</p>
<ul>
  <li>A) O(N) backward passes</li>
  <li>B) O(M) backward passes</li>
  <li>C) O(N * M) passes</li>
  <li>D) O(1) passes</li>
</ul>
<p><strong>2. Coding Challenge:</strong> Write a topological sort function in Python that takes a node and yields its dependencies in topological order.</p>
<p><strong>3. Conceptual Essay:</strong> Why is backtracking line search used in Newton's optimization method?</p>""",
        "answers": """<p><strong>1. Multiple Choice Answer:</strong> B</p>
<p><em>Explanation:</em> Reverse-mode autograd propagates gradients backward from outputs to inputs, requiring only O(M) backward sweeps where M is the number of outputs. This makes it highly efficient for neural networks where N (parameters) >> M (scalar loss).</p>
<p><strong>2. Coding Challenge Answer:</strong></p>
<pre><code class="language-python">def topological_sort(node):
    visited = set()
    order = []
    def dfs(n):
        if n not in visited:
            visited.add(n)
            for child in getattr(n, 'children', []):
                dfs(child)
            order.append(n)
    dfs(node)
    return order[::-1]</code></pre>
<p><strong>3. Conceptual Essay Answer:</strong></p>
<p>While Newton's method converges quadratically near a local minimum, a full step size (α = 1.0) can overshoot and diverge if the starting point is far from the minimum or if the loss surface has high curvature. Backtracking line search systematically shrinks the step size until the Armijo condition (sufficient decrease) is satisfied, guaranteeing global convergence.</p>"""
    },
    4: {
        "content": """<p><strong>1. Multiple Choice:</strong> What condition must a kernel function satisfy to be valid for SVMs?</p>
<ul>
  <li>A) Shannon condition</li>
  <li>B) Mercer's condition</li>
  <li>C) Armijo condition</li>
  <li>D) Fiedler condition</li>
</ul>
<p><strong>2. Coding Challenge:</strong> Write a Python function to compute the Shannon entropy of a binary label list.</p>
<p><strong>3. Conceptual Essay:</strong> Explain why PCA projections correspond to the eigenvectors of the covariance matrix.</p>""",
        "answers": """<p><strong>1. Multiple Choice Answer:</strong> B</p>
<p><em>Explanation:</em> Mercer's condition guarantees that a kernel function corresponds to an inner product in some Hilbert space, ensuring the optimization problem remains convex.</p>
<p><strong>2. Coding Challenge Answer:</strong></p>
<pre><code class="language-python">import math
def compute_entropy(labels):
    n = len(labels)
    if n == 0: return 0.0
    p1 = sum(labels) / n
    p0 = 1.0 - p1
    if p1 == 0.0 or p0 == 0.0: return 0.0
    return -p0 * math.log2(p0) - p1 * math.log2(p1)</code></pre>
<p><strong>3. Conceptual Essay Answer:</strong></p>
<p>PCA seeks a projection direction w that maximizes the variance of the projected data. Geometrically, the variance of projected centered data X is w^T C w, where C is the covariance matrix. Finding the direction w that maximizes this quadratic form subject to the constraint ||w|| = 1 can be formulated via Lagrange multipliers as C w = λ w, which is the eigenvalue equation for C. The direction of maximum variance is therefore the leading eigenvector of C.</p>"""
    },
    5: {
        "content": """<p><strong>1. Multiple Choice:</strong> Which vector is used to partition a graph in spectral clustering?</p>
<ul>
  <li>A) PageRank vector</li>
  <li>B) Fiedler vector</li>
  <li>C) Stationary distribution</li>
  <li>D) Singular vector</li>
</ul>
<p><strong>2. Coding Challenge:</strong> Implement a simple forward-pass class for a 2-layer MLP in Python using NumPy.</p>
<p><strong>3. Conceptual Essay:</strong> Prove why a single-layer Perceptron cannot solve the XOR problem.</p>
<p><strong>4. Multiple Choice:</strong> What weight initialization scale is mathematically derived to keep activation variance stable in networks with ReLU activation functions?</p>
<ul>
  <li>A) scale = 1 / n_in</li>
  <li>B) scale = sqrt(1 / n_in)</li>
  <li>C) scale = sqrt(2 / n_in)</li>
  <li>D) scale = 2 / n_in</li>
</ul>
<p><strong>5. Coding Challenge:</strong> Implement a custom ReLU backward pass module class method <code>backward(self, grad_output)</code> in Python/NumPy, assuming the forward-pass input was cached in <code>self.input</code>.</p>
<p><strong>6. Conceptual Essay:</strong> Explain the mathematical origin of vanishing gradients and how the product of Jacobians across depth causes this pathology.</p>""",
        "answers": """<p><strong>1. Multiple Choice Answer:</strong> B</p>
<p><em>Explanation:</em> The Fiedler vector (eigenvector corresponding to the second smallest eigenvalue of the graph Laplacian) contains signs that reveal the optimal partition cut of the graph.</p>
<p><strong>2. Coding Challenge Answer:</strong></p>
<pre><code class="language-python">import numpy as np
def relu(x): return np.maximum(0, x)

class MLP2Layer:
    def __init__(self, W1, b1, W2, b2):
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return z2</code></pre>
<p><strong>3. Conceptual Essay Answer:</strong></p>
<p>A single-layer Perceptron computes a linear decision boundary of the form w^T x + b = 0. The XOR logic gate inputs (0,0) and (1,1) produce output 0, while (1,0) and (0,1) produce output 1. These points cannot be separated by a single straight line in 2D space (they are not linearly separable). A multi-layer architecture with non-linear activations is required to warp the space and allow linear separation in a higher-dimensional representation.</p>
<p><strong>4. Multiple Choice Answer:</strong> C</p>
<p><em>Explanation:</em> He (Kaiming) initialization uses scale = sqrt(2 / n_in). The factor of 2 compensates for the fact that ReLU sets approximately half of the values to zero, halving the variance at each layer.</p>
<p><strong>5. Coding Challenge Answer:</strong></p>
<pre><code class="language-python">def backward(self, grad_output):
    # Gradients w.r.t input: only pass gradient back where forward input > 0
    return grad_output * (self.input > 0.0)</code></pre>
<p><strong>6. Conceptual Essay Answer:</strong></p>
<p>Under the multi-dimensional chain rule, the gradient w.r.t the first layer's activations is computed as the product of the Jacobians of all subsequent layers: dL/dh_1 = dL/dh_L * J_L * J_{L-1} * ... * J_2. For activation functions like Sigmoid, the derivative is bounded by 0.25. When many such bounded terms are multiplied across a deep network, the gradient magnitude decays exponentially with depth (e.g., 0.25^20 ≈ 1e-13), leaving the earliest feature representation layers with no gradient signal to update their weights.</p>"""
    }
}

LEARNING_GOALS = {
    0: "Identify and interpret basic mathematical notations including functions (domain and codomain), set membership, summation, products, and ceiling/floor functions. Trace basic Python control flow, list comprehensions, and nested loops, and translate them to mathematical notation.",
    1: "Derive OLS slope and intercept for simple linear regression. Build a multiple OLS regression model from scratch, run diagnostics on residuals (homoskedasticity, normality), and implement standard logistic regression using gradient descent.",
    2: "Specify canonical exponential family representations for Gaussian, Binomial, and Poisson distributions. Apply the IRLS algorithm to fit count models (Poisson) and proportion models (Binomial) with exposure offsets.",
    3: "Compute strides for any dimensional array under C-contiguous and F-contiguous ordering. Perform affine index conversions to map slice indices to flat memory offsets. Identify GPU memory bottlenecks using arithmetic intensity.",
    4: "Trace forward and backward variables in a DAG. Differentiate between forward and reverse mode automatic differentiation. Write a modular tape-based autograd engine in Python.",
    5: "Confirm function convexity using Hessian eigenvalues. Implement Newton's optimization method with backtracking line search. Write LU and QR factorization solvers from scratch.",
    6: "Derive primal and dual formulations of SVMs. Explain Mercer's condition and apply RBF and polynomial kernels. Build a decision tree using Shannon entropy splits and design a cache-friendly tree storage layout.",
    7: "Prove the maximum-variance projection direction. Factorize matrices using full and economy SVD. Implement power iteration to extract the dominant eigenvector.",
    8: "Construct unnormalized and normalized graph Laplacians. Compute Fiedler vectors for spectral graph partitioning. Find stationary PageRank distributions via power iteration.",
    9: "State the Universal Approximation Theorem. Prove the XOR impossibility for linear boundaries. Compare properties of Sigmoid, Tanh, ReLU, and GELU activation functions.",
    10: "Derive the backpropagation gradient equations for linear projection layers, activation functions, and cross-entropy loss. Diagnose training pathologies including vanishing/exploding gradients and dead ReLUs. Implement numerical gradient checking to verify analytical correctness."
}

PRACTICAL_TESTS = {
    0: {
        "test": "Write a python function <code>sum_even_squares(lst)</code> that takes a list of integers, filters for even numbers, and returns the sum of their squares. Translate this operation to mathematical summation notation.",
        "answer": """<p><strong>Code Implementation:</strong></p>
<pre><code class="language-python">def sum_even_squares(lst):
    return sum(x**2 for x in lst if x % 2 == 0)</code></pre>
<p><strong>Mathematical Summation Notation:</strong></p>
<div class="formula">\\sum_{x \\in \\text{lst}, x \\text{ is even}} x^2</div>"""
    },
    1: {
        "test": "Given data points X = [1, 2, 3] and Y = [2, 4, 5], calculate the OLS slope <i>m</i> and intercept <i>c</i> by hand. Write a python snippet to verify your answers.",
        "answer": """<p><strong>Mathematical Derivation:</strong></p>
<ul>
  <li>Mean of X (x̄) = 2, Mean of Y (ȳ) = 11/3 ≈ 3.667.</li>
  <li>Covariance (X, Y) = [ (1-2)(2-3.667) + (2-2)(4-3.667) + (3-2)(5-3.667) ] / 2 = [ 1.667 + 0 + 1.333 ] / 2 = 1.5.</li>
  <li>Variance (X) = [ (1-2)² + (2-2)² + (3-2)² ] / 2 = 1.0.</li>
  <li>Slope m = Cov(X,Y) / Var(X) = 1.5 / 1.0 = 1.5.</li>
  <li>Intercept c = ȳ - m(x̄) = 3.667 - 1.5(2) = 0.667.</li>
</ul>
<p><strong>Python Verification Code:</strong></p>
<pre><code class="language-python">import numpy as np
X, Y = np.array([1, 2, 3]), np.array([2, 4, 5])
m, c = np.polyfit(X, Y, 1)
print(f"Slope: {m:.3f}, Intercept: {c:.3f}")  # Slope: 1.500, Intercept: 0.667</code></pre>"""
    },
    2: {
        "test": "Write down the Poisson deviance formula. Write a python script to compute the deviance for actual counts Y = [3, 5] and predicted means μ = [2.5, 5.2].",
        "answer": """<p><strong>Poisson Deviance Formula:</strong></p>
<div class="formula">D = 2 \\sum_{i} \\left[ y_i \\log\\left(\\frac{y_i}{\\mu_i}\\right) - (y_i - \\mu_i) \\right]</div>
<p><strong>Python Implementation:</strong></p>
<pre><code class="language-python">import math
y = [3, 5]
mu = [2.5, 5.2]
dev = 2 * sum((yi * math.log(yi/mui) - (yi - mui)) if yi else mui for yi, mui in zip(y, mu))
print(f"Poisson Deviance: {dev:.4f}")  # Poisson Deviance: 0.1415</code></pre>"""
    },
    3: {
        "test": "For a C-contiguous tensor of shape (3, 4, 5), calculate the strides vector. If we take a slice <code>t[1, 2, 3]</code>, what is its flat memory offset (assuming 4-byte float values)?",
        "answer": """<p><strong>Strides Vector Calculation:</strong></p>
<ul>
  <li>Shape = (d0, d1, d2) = (3, 4, 5).</li>
  <li>s2 = 1</li>
  <li>s1 = d2 * s2 = 5</li>
  <li>s0 = d1 * s1 = 4 * 5 = 20</li>
  <li>Strides Vector = (20, 5, 1).</li>
</ul>
<p><strong>Memory Offset Calculation:</strong></p>
<ul>
  <li>Flat index = (1 * 20) + (2 * 5) + (3 * 1) = 20 + 10 + 3 = 33.</li>
  <li>Byte offset = 33 * 4 bytes = 132 bytes.</li>
</ul>"""
    },
    4: {
        "test": "For the function <i>f(x, y) = x²y + sin(x)</i>, calculate by hand the partial derivatives at <i>x = 2.0, y = 3.0</i>.",
        "answer": """<p><strong>Mathematical Derivation:</strong></p>
<ul>
  <li>∂f/∂x = 2xy + cos(x). At x = 2.0, y = 3.0: 2(2.0)(3.0) + cos(2.0) ≈ 12 + (-0.416) = 11.584.</li>
  <li>∂f/∂y = x². At x = 2.0, y = 3.0: 2.0² = 4.0.</li>
</ul>"""
    },
    5: {
        "test": "Compute one iteration of Newton's optimization method by hand for minimizing the function <i>f(x) = x⁴ - 2x²</i> starting from <i>x₀ = 2.0</i>.",
        "answer": """<p><strong>Mathematical Derivation:</strong></p>
<ul>
  <li>First derivative: f'(x) = 4x³ - 4x. At x = 2.0: f'(2) = 4(8) - 8 = 24.</li>
  <li>Second derivative (Hessian): f''(x) = 12x² - 4. At x = 2.0: f''(2) = 12(4) - 4 = 44.</li>
  <li>Newton update step: x₁ = x₀ - f'(x₀) / f''(x₀) = 2.0 - 24 / 44 = 2.0 - 0.545 = 1.455.</li>
</ul>"""
    },
    6: {
        "test": "For a dataset with binary labels [1, 1, 0, 0, 0], calculate the Shannon entropy in bits by hand.",
        "answer": """<p><strong>Shannon Entropy Calculation:</strong></p>
<ul>
  <li>p(1) = 2/5 = 0.4.</li>
  <li>p(0) = 3/5 = 0.6.</li>
  <li>Entropy H = -p(0) log₂ p(0) - p(1) log₂ p(1)</li>
  <li>H = -0.6 log₂ 0.6 - 0.4 log₂ 0.4 ≈ -0.6(-0.737) - 0.4(-1.322) = 0.442 + 0.529 = 0.971 bits.</li>
</ul>"""
    },
    7: {
        "test": "Given a centered data matrix <i>X = [[-1, -1], [0, 0], [1, 1]]</i>, compute the empirical covariance matrix and find its leading eigenvector.",
        "answer": """<p><strong>Covariance Matrix Calculation:</strong></p>
<div class="formula">C = \\frac{1}{N-1} X^\\top X = \\frac{1}{2} \\begin{bmatrix} 2 & 2 \\\\ 2 & 2 \\end{bmatrix} = \\begin{bmatrix} 1 & 1 \\\\ 1 & 1 \\end{bmatrix}</div>
<p><strong>Leading Eigenvector Calculation:</strong></p>
<ul>
  <li>Characteristic Equation: det(C - λI) = (1 - λ)² - 1 = 0 ⇒ λ² - 2λ = 0 ⇒ λ₁ = 2, λ₂ = 0.</li>
  <li>Leading Eigenvector (for λ = 2): Solve (C - 2I)v = 0 ⇒ [[-1, 1], [1, -1]] v = 0 ⇒ v = [1, 1]ᵀ.</li>
  <li>Normalized Eigenvector: [1/√2, 1/√2]ᵀ.</li>
</ul>"""
    },
    8: {
        "test": "For a path graph with 3 nodes connected sequentially (1-2 and 2-3), construct the adjacency matrix A, degree matrix D, and unnormalized Laplacian matrix L.",
        "answer": """<p><strong>Graph Matrices:</strong></p>
<p>Adjacency Matrix A:</p>
<div class="formula">A = \\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & 0 & 1 \\\\ 0 & 1 & 0 \\end{bmatrix}</div>
<p>Degree Matrix D:</p>
<div class="formula">D = \\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 2 & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}</div>
<p>Unnormalized Laplacian Matrix L = D - A:</p>
<div class="formula">L = \\begin{bmatrix} 1 & -1 & 0 \\\\ -1 & 2 & -1 \\\\ 0 & -1 & 1 \\end{bmatrix}</div>"""
    },
    9: {
        "test": "Write a python function <code>relu(x)</code> and its derivative function <code>relu_derivative(x)</code> for a float input x.",
        "answer": """<p><strong>Python Implementation:</strong></p>
<pre><code class="language-python">def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0</code></pre>"""
    },
    10: {
        "test": "Write a python function <code>sigmoid_backward(out, grad_out)</code> that implements the backward pass of a Sigmoid activation layer, where <code>out</code> is the forward pass output of the Sigmoid layer.",
        "answer": """<p><strong>Python Implementation:</strong></p>
<pre><code class="language-python">def sigmoid_backward(out, grad_out):
    return grad_out * out * (1.0 - out)</code></pre>"""
    }
}

def slugify(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')

def clean_heading_text(heading_str: str) -> str:
    return re.sub(r'\s+', ' ', heading_str).strip()

def shift_markdown_headings(text: str, chapter_title: str = None) -> str:
    lines = text.splitlines()
    new_lines = []
    first_h1_seen = False
    in_code_block = False
    
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
            
        if not in_code_block and line.startswith("#"):
            num_hashes = len(line) - len(line.lstrip('#'))
            if num_hashes < len(line) and line[num_hashes] == ' ':
                if num_hashes == 1:
                    if not first_h1_seen:
                        if chapter_title:
                            line = f"# {chapter_title}"
                        first_h1_seen = True
                    else:
                        line = "## " + line[2:]
        new_lines.append(line)
        
    shifted_lines = []
    in_code_block = False
    for line in new_lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            shifted_lines.append(line)
            continue
            
        if not in_code_block and line.startswith("#"):
            num_hashes = len(line) - len(line.lstrip('#'))
            if num_hashes < len(line) and line[num_hashes] == ' ':
                # Shift down by 2 levels (## becomes ####, # becomes ###)
                line = '##' + line
        shifted_lines.append(line)
        
    return '\n'.join(shifted_lines)

def shift_html_headings_down_twice(soup):
    for tag_name in ["h4", "h3", "h2"]:
        for tag in soup.find_all(tag_name):
            new_level = int(tag_name[1]) + 2
            tag.name = f"h{new_level}"

def renumber_content(content: str) -> str:
    def repl(match):
        prefix = match.group(1)
        num = int(match.group(2))
        return f"{prefix}{num + 2}"
        
    content = re.sub(r'(\b[Cc]hapter\s+)(\d+)\b', repl, content)
    
    def repl_underscore(match):
        prefix = match.group(1)
        num = int(match.group(2))
        return f"{prefix}{num + 2:02d}"
    content = re.sub(r'(\b[Cc]hapter_?)(\d+)\b', repl_underscore, content)
    
    return content

def inject_chapter_components(soup, chapter_num, section_element):
    # Find heading (h3 title now)
    h3_title = section_element.find("h3")
    if not h3_title:
        h3_title = section_element.find("h2")
        if not h3_title:
            return
        
    # Create Learning Goal Box
    goal_box = soup.new_tag("div", attrs={"class": "box review-note"})
    goal_box.append(BeautifulSoup(f"<strong>Learning Goal:</strong> {LEARNING_GOALS[chapter_num]}", "html.parser"))
    h3_title.insert_after(goal_box)

    # Create Practical Test Box at the end of the chapter
    test_box = soup.new_tag("div", attrs={"class": "box warn"})
    test_box.append(BeautifulSoup(f"<strong>Chapter {chapter_num} Practical Test:</strong><p>{PRACTICAL_TESTS[chapter_num]['test']}</p>", "html.parser"))
    
    details = soup.new_tag("details", attrs={"class": "deep-dive"})
    summary = soup.new_tag("summary")
    summary.string = "View Answer Key & Explanation"
    details.append(summary)
    
    body_div = soup.new_tag("div", attrs={"class": "deep-dive-body"})
    body_div.append(BeautifulSoup(PRACTICAL_TESTS[chapter_num]["answer"], "html.parser"))
    details.append(body_div)
    
    test_box.append(details)
    section_element.append(test_box)

def inject_section_components(soup, sec_num, section_element):
    h2_title = section_element.find("h2")
    if not h2_title:
        return
        
    # Create Section Introduction Box
    intro_box = soup.new_tag("div", attrs={"class": "box review-note"})
    intro_box.append(BeautifulSoup(f"""<strong>Section {sec_num} Mastery Lesson:</strong>
<p><strong>Starting Assumptions:</strong> {SECTION_INTRODUCTIONS[sec_num]['assumptions']}</p>
<p><strong>Advanced Research Deliverable:</strong> {SECTION_INTRODUCTIONS[sec_num]['deliverable']}</p>
<p><strong>Curriculum Path:</strong> {SECTION_INTRODUCTIONS[sec_num]['continuation']}</p>""", "html.parser"))
    h2_title.insert_after(intro_box)

    # Create Section Comprehensive Test Box & Resources at the end
    test_box = soup.new_tag("div", attrs={"class": "box warn"})
    test_box.append(BeautifulSoup(f"""<strong>Section {sec_num} Comprehensive Testing:</strong>
<p>Gauge your section understanding with these multiple-choice, coding, and essay questions. Solve them before viewing the answers.</p>
{SECTION_TESTS[sec_num]['content']}""", "html.parser"))
    
    # Collapsible answers
    details = soup.new_tag("details", attrs={"class": "deep-dive"})
    summary = soup.new_tag("summary")
    summary.string = "View Section Answer Keys & Explanations"
    details.append(summary)
    
    body_div = soup.new_tag("div", attrs={"class": "deep-dive-body"})
    body_div.append(BeautifulSoup(SECTION_TESTS[sec_num]['answers'], "html.parser"))
    details.append(body_div)
    test_box.append(details)
    
    # Read More Resources Box
    resources_box = soup.new_tag("div", attrs={"class": "box key"})
    resources_box.append(BeautifulSoup(f"""<strong>Read More Resources:</strong>
<ul>
  {SECTION_RESOURCES[sec_num]}
</ul>""", "html.parser"))
    
    section_element.append(test_box)
    section_element.append(resources_box)

def add_cross_links(soup):
    links = [
        (0, r'\b(Beginner Prerequisites|arithmetic basics|set membership)\b', 'chapter-0-beginner-prerequisites-for-machine-learning', 'Chapter 0 Prerequisites'),
        (1, r'\b(linear regression|Ordinary Least Squares|OLS|logistic regression)\b', 'chapter-1-linear-regression-logistic-models', 'Chapter 1 Regression Model'),
        (2, r'\b(Generalized Linear Models|GLMs|exponential family|Poisson regression|IRLS)\b', 'chapter-2-generalized-linear-models', 'Chapter 2 GLMs'),
        (3, r'\b(strided tensors?|strides? vector|C-contiguous|F-contiguous|GPU memory hierarchy|memory bandwidth)\b', 'chapter-3-the-anatomy-of-a-tensor-compute-hardware', 'Chapter 3 Strided Tensors'),
        (4, r'\b(automatic differentiation|autograd|computational graphs?|reverse-mode autograd|backpropagation)\b', 'chapter-4-the-core-optimization-engine-automatic-differentiation', 'Chapter 4 Autograd Engine'),
        (5, r'\b(convex optimization|Newton\'s method|Hessian eigenvalues|LU factorization)\b', 'chapter-5-convex-optimization', 'Chapter 5 Convex Optimization'),
        (6, r'\b(Support Vector Machines|SVMs|Mercer\'s condition|decision trees?|Shannon entropy)\b', 'chapter-6-non-linear-spaces-regularization', 'Chapter 6 SVMs & Trees'),
        (7, r'\b(Singular Value Decomposition|SVD|Principal Component Analysis|PCA|variance maximization)\b', 'chapter-7-dimensionality-reduction-latent-spaces', 'Chapter 7 SVD & PCA'),
        (8, r'\b(graph Laplacians?|Fiedler vector|spectral clustering|PageRank)\b', 'chapter-8-graph-spaces-spectral-clustering', 'Chapter 8 Graph Spaces'),
        (9, r'\b(Universal Approximation Theorem|activation functions?|perceptrons?|ReLU|Sigmoid|GELU)\b', 'chapter-9-foundations-of-neural-spaces-perceptrons-activation-functions', 'Chapter 9 Neural Spaces'),
    ]

    def owning_chapter(node):
        section = node if getattr(node, "name", None) == "section" else node.find_parent("section")
        while section is not None:
            sec_id = section.get("id") or ""
            m = re.match(r'^chapter-(\d+)$', sec_id)
            if m:
                return int(m.group(1))
            section = section.find_parent("section")
        return None

    for text_node in list(soup.find_all(text=True)):
        parent = text_node.parent
        if not parent:
            continue
        if parent.name in ['code', 'pre', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'title', 'script', 'style']:
            continue

        text_str = str(text_node)
        current_chapter = owning_chapter(parent)

        for chapter_num, pattern, anchor, display in links:
            if chapter_num == current_chapter:
                # Don't link a term back to the very chapter the text already lives in.
                continue
            if re.search(pattern, text_str, re.IGNORECASE):
                def repl(match):
                    matched_text = match.group(0)
                    return f'<a href="#{anchor}" target="_blank" class="cross-link">{matched_text} ({display})</a>'

                new_html_str = re.sub(pattern, repl, text_str, flags=re.IGNORECASE)
                if new_html_str != text_str:
                    new_soup = BeautifulSoup(new_html_str, "html.parser")
                    text_node.replace_with(new_soup)
                    break

def compile_book():
    print("Loading original regression_guide.html...")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    if soup.title:
        soup.title.string = "Zero-to-Research Machine Learning Textbook (Complete)"
    if soup.header:
        h1 = soup.header.find("h1")
        if h1:
            h1.string = "Zero-to-Research Machine Learning Textbook (Complete)"
        p_lede = soup.header.find("p", class_="lede")
        if p_lede:
            p_lede.string = "A complete 10-chapter, step-by-step, zero-to-research-level machine learning curriculum. Designed for NotebookLM and guided self-study, incorporating basics, classical linear models, hardware architectures, autograd, convex optimization, non-linear spaces, dimensionality reduction, graph embeddings, and neural network foundations."

    # Inject styled CSS for cross-links
    if soup.head and soup.head.style:
        soup.head.style.append("""
    .cross-link {
      color: var(--accent);
      text-decoration: underline dotted;
      font-weight: 500;
      transition: color 0.15s ease;
    }
    .cross-link:hover {
      color: #0d9488;
      text-decoration: underline solid;
    }
""")

    head_content = str(soup.head)
    header_content = str(soup.header)
    notes_section = soup.find("section", id="notes")
    floating_panel = soup.find("aside", id="floating-note-panel")
    scripts = soup.find_all("script")
    script_content = "\n".join(str(s) for s in scripts)

    # --- Chapter 0: Prerequisites ---
    print("Compiling Chapter 0...")
    ch0_file = SOURCE_DIR / "chapter0_Basics" / "beginner_prerequisites_for_tensors_with_models.md"
    with open(ch0_file, "r", encoding="utf-8") as f:
        ch0_md = f.read()
    ch0_md = shift_markdown_headings(ch0_md, "Chapter 0: Beginner Prerequisites for Machine Learning")
    ch0_html = markdown.markdown(ch0_md, extensions=["fenced_code", "tables", "toc", "md_in_html"])
    
    ch0_soup = BeautifulSoup(ch0_html, "html.parser")
    ch0_section = soup.new_tag("section", id="chapter-0")
    ch0_section.extend(ch0_soup.contents)
    inject_chapter_components(soup, 0, ch0_section)

    # --- Chapter 1: Regression (OLS) ---
    print("Compiling Chapter 1...")
    ch1_file = ROOT / "Regression_Mathematical_Statistical_and_Python_Guide_Updated.md"
    with open(ch1_file, "r", encoding="utf-8") as f:
        ch1_md = f.read()
    ch1_md = shift_markdown_headings(ch1_md, "Chapter 1: Linear Regression & Logistic Models")
    ch1_html = markdown.markdown(ch1_md, extensions=["fenced_code", "tables", "toc", "md_in_html"])

    ch1_soup = BeautifulSoup(ch1_html, "html.parser")
    ch1_section = soup.new_tag("section", id="chapter-1")
    ch1_section.extend(ch1_soup.contents)
    inject_chapter_components(soup, 1, ch1_section)

    # --- Chapter 2: GLM ---
    print("Compiling Chapter 2...")
    glm_file = ROOT / "Generalized_Linear_Models_Mathematical_Statistical_and_Python_Guide.md"
    with open(glm_file, "r", encoding="utf-8") as f:
        glm_md = f.read()
    glm_md = shift_markdown_headings(glm_md, "Chapter 2: Generalized Linear Models")
    glm_html = markdown.markdown(glm_md, extensions=["fenced_code", "tables", "toc", "md_in_html"])
    
    glm_soup = BeautifulSoup(glm_html, "html.parser")
    ch2_section = soup.new_tag("section", id="chapter-2")
    ch2_section.extend(glm_soup.contents)
    inject_chapter_components(soup, 2, ch2_section)

    # --- Chapters 3 to 9 ---
    source_files = [
        (3, "ch01_introduction/chapter_01.md"),
        (4, "chapter02_autograde/chapter_02.md"),
        (5, "chapter03_covex_optimization/chapter03_Convex_Optimization.md"),
        (6, "chapter 4/chapter4.md"),
        (7, "chapter5/chapter5.md"),
        (8, "chapter6/chapter6.md"),
        (9, "chapter7/chapter7.md"),
        (10, "chapter10/chapter10.md")
    ]

    ch_sections = {}
    for ch_num, rel_path in source_files:
        print(f"Compiling Chapter {ch_num} ({rel_path})...")
        file_path = SOURCE_DIR / rel_path
        with open(file_path, "r", encoding="utf-8") as f:
            ch_md = f.read()
            
        if ch_num < 10:
            ch_md = renumber_content(ch_md)
        ch_md = shift_markdown_headings(ch_md)
        
        ch_html = markdown.markdown(ch_md, extensions=["fenced_code", "tables", "toc", "md_in_html"])
        ch_soup = BeautifulSoup(ch_html, "html.parser")
        
        # Extract the first heading if it represents the chapter title
        first_h = ch_soup.find(["h3", "h4"])
        chapter_title = ""
        if first_h and re.match(r'^[Cc]hapter\s+\d+:', first_h.get_text().strip(), re.IGNORECASE):
            chapter_title = first_h.get_text().strip()

        # Decompose any headings that match the chapter title pattern
        for h_tag in list(ch_soup.find_all(["h3", "h4", "h5"])):
            text = h_tag.get_text().strip()
            if re.match(r'^[Cc]hapter\s+\d+:', text, re.IGNORECASE):
                h_tag.decompose()

        ch_section = soup.new_tag("section", id=f"chapter-{ch_num}")
        
        # Inject the clean H3 heading at the top
        if chapter_title:
            ch_title_tag = soup.new_tag("h3", id=slugify(chapter_title))
            ch_title_tag.string = chapter_title
            ch_section.append(ch_title_tag)
            
        ch_section.extend(ch_soup.contents)
        inject_chapter_components(soup, ch_num, ch_section)
        ch_sections[ch_num] = ch_section

    # --- Build Section Hierarchy ---
    print("Building Section Hierarchy...")
    compiled_sections = []

    # Section I
    sec1 = soup.new_tag("section", id="section-1")
    sec1_title = soup.new_tag("h2")
    sec1_title.string = "Section I: Foundations & Prerequisites"
    sec1.append(sec1_title)
    sec1.append(ch0_section)
    sec1.append(ch1_section)
    inject_section_components(soup, 1, sec1)
    compiled_sections.append(sec1)

    # Section II
    sec2 = soup.new_tag("section", id="section-2")
    sec2_title = soup.new_tag("h2")
    sec2_title.string = "Section II: Generalized Linear Models & Hardware Anatomy"
    sec2.append(sec2_title)
    sec2.append(ch2_section)
    sec2.append(ch_sections[3])
    inject_section_components(soup, 2, sec2)
    compiled_sections.append(sec2)

    # Section III
    sec3 = soup.new_tag("section", id="section-3")
    sec3_title = soup.new_tag("h2")
    sec3_title.string = "Section III: Deep Learning Engine (Autograd & Convex Optimization)"
    sec3.append(sec3_title)
    sec3.append(ch_sections[4])
    sec3.append(ch_sections[5])
    inject_section_components(soup, 3, sec3)
    compiled_sections.append(sec3)

    # Section IV
    sec4 = soup.new_tag("section", id="section-4")
    sec4_title = soup.new_tag("h2")
    sec4_title.string = "Section IV: Non-Linear Spaces & Unsupervised Latent Embeddings"
    sec4.append(sec4_title)
    sec4.append(ch_sections[6])
    sec4.append(ch_sections[7])
    inject_section_components(soup, 4, sec4)
    compiled_sections.append(sec4)

    # Section V
    sec5 = soup.new_tag("section", id="section-5")
    sec5_title = soup.new_tag("h2")
    sec5_title.string = "Section V: Topological Embeddings & Neural Architectures"
    sec5.append(sec5_title)
    sec5.append(ch_sections[8])
    sec5.append(ch_sections[9])
    sec5.append(ch_sections[10])
    inject_section_components(soup, 5, sec5)
    compiled_sections.append(sec5)

    print("Assembling final HTML structure...")
    new_article = soup.new_tag("article")
    new_article.append(notes_section)
    
    for sec in compiled_sections:
        new_article.append(sec)

    # Add dynamic cross references across the text
    print("Running cross-linking engine...")
    add_cross_links(new_article)

    print("Generating continuous Table of Contents...")
    headings = new_article.find_all(["h2", "h3"])
    toc_html = ['<div class="toc"><ul>']
    current_h2_item = None
    used_ids = set()

    for heading in headings:
        if heading.get("id") == "notes" or heading.find_parent(id="notes"):
            continue
            
        title = heading.get_text().strip()
        if "Add note" in title:
            title = title.replace("Add note", "").strip()
            
        anchor = heading.get("id")
        if not anchor or anchor in used_ids:
            base_anchor = anchor or slugify(title)
            anchor = base_anchor
            counter = 2
            while anchor in used_ids:
                anchor = f"{base_anchor}-{counter}"
                counter += 1
            heading["id"] = anchor
        used_ids.add(anchor)
        
        if heading.name == "h2":
            if current_h2_item:
                toc_html.append('</ul></li>')
            toc_html.append(f'<li><a href="#{anchor}">{title}</a><ul>')
            current_h2_item = True
        elif heading.name == "h3":
            if current_h2_item:
                toc_html.append(f'<li><a href="#{anchor}">{title}</a></li>')
                
    if current_h2_item:
        toc_html.append('</ul></li>')
    toc_html.append('</ul></div>')
    
    new_nav = soup.new_tag("nav", attrs={"aria-label": "Table of contents"})
    new_nav.append(BeautifulSoup('<a href="#notes">Notes Hub</a>\n' + "\n".join(toc_html), "html.parser"))

    soup.find("article").replace_with(new_article)
    soup.find("nav").replace_with(new_nav)

    for s in soup.find_all("script"):
        s.decompose()

    final_html = str(soup) + "\n" + script_content
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_html)
    print("Handbook successfully compiled to regression_guide.html!")

if __name__ == "__main__":
    compile_book()
