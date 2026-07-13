# Planning: Chapter 10 — Deep Feedforward Networks & Backpropagation

This document outlines the topics, mathematical derivations, software engineering abstractions, and deliverables for the next chapter in the Zero-to-Research ML curriculum.

---

## 1. Chapter Overview
* **Title:** Chapter 10: Deep Feedforward Networks & Backpropagation (Gradient Flow Dynamics)
* **Preceding Chapter:** Chapter 9: Foundations of Neural Spaces (Perceptrons & Activation Functions)
* **Goal:** Demystify backpropagation by deriving the matrix calculus equations end-to-end, constructing a modular deep learning library in pure Python/NumPy, and mathematically proving gradient correctness using numerical finite differences.

---

## 2. Mathematical Foundations (Part I)
* **Multi-Dimensional Chain Rule (Matrix Calculus):**
  * Jacobian matrices and the formal tensor layout of derivatives.
  * Derivation of the backward pass for a linear layer:
    $$\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Z}} \mathbf{W}$$
    $$\frac{\partial L}{\partial \mathbf{W}} = \left(\frac{\partial L}{\partial \mathbf{Z}}\right)^\top \mathbf{X}$$
    $$\frac{\partial L}{\partial \mathbf{b}} = \sum_{\text{rows}} \frac{\partial L}{\partial \mathbf{Z}}$$
* **Gradient Flow Pathology:**
  * The mathematical origin of vanishing and exploding gradients in deep networks.
  * The role of activation derivatives (e.g., Sigmoid saturation vs. ReLU dead states).
* **Weight Initialization Derivations:**
  * Deriving **Xavier (Glorot) Initialization** variance scaling under linear activations:
    $$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$
  * Deriving **He (Kaiming) Initialization** variance scaling under ReLU activations:
    $$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$
* **Loss Functions and Gradients:**
  * Derivation of Softmax coupled with Categorical Cross-Entropy Loss gradients.

---

## 3. From Scratch Implementation (Part II)
* **Modular Interface Design:**
  * Establishing the `Module` contract: `forward(*args)` and `backward(grad_output)`.
  * Implementing parameter registration and gradient accumulation.
* **Component Abstractions:**
  * `Linear(in_features, out_features)`: Strided parameter matrices and bias vectors.
  * `ReLU()` and `Sigmoid()`: Element-wise activation gates tracking activation states for backward passes.
  * `Sequential(*layers)`: A pipeline container driving sequential forward execution and reversed backward propagation.
  * `CrossEntropyLoss()`: Numerically stable Log-Sum-Exp computation and backward gradient calculation.
* **Mathematical Verification (Numerical Gradient Check):**
  * Implementing **Finite Difference Approximation**:
    $$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$
  * Writing an automated check comparing analytical backpropagation gradients against numerical approximations, requiring a relative error threshold of $< 10^{-7}$.

---

## 4. Industry Libraries Integration (Part III)
* **PyTorch Equivalents:**
  * Constructing the identical network architecture using `torch.nn.Module`.
  * Tracking PyTorch autograd graph creation and dynamic parameter updates.
* **Optimization and Diagnostics:**
  * Standard Stochastic Gradient Descent (SGD) training loops.
  * Diagnostic plotting recipes:
    * Activation distribution tracking per layer (detecting dead ReLUs).
    * Gradient norm tracking across training epochs (detecting vanishing gradients).
    * Training/Validation loss curves.

---

## 5. Exercises & Deliverables (Part IV)
* **Core Research Deliverables:**
  1. **Modular MLP Library:** A clean, dependency-free Python file `mlp_engine.py` containing the custom modules.
  2. **Numerical Gradient Tester:** A validation script proving backpropagation mathematical correctness to machine precision.
  3. **Non-Linear Decision Boundary Solver:** An execution script training the scratch model on a synthetic 2D non-linear dataset (e.g., double spiral) and plotting the decision boundary.
* **Read More Bibliography:**
  * *Deep Learning* (Goodfellow, Bengio, Courville) - Chapter 6 (Deep Feedforward Networks).
  * *Understanding the difficulty of training deep feedforward neural networks* (Glorot & Bengio, 2010).
  * *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* (He et al., 2015).
