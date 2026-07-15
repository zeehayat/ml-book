# Chapter 10: Deep Feedforward Networks & Backpropagation
## Gradient Flow Dynamics

---

## Technical Brief

Chapter 10 turns the forward-pass neural network from the previous chapter into a trainable deep model. Chapter 9 showed why non-linear activations are necessary: stacked linear maps collapse into one linear map, while stacked affine maps with non-linear activations can learn task-specific representations. This chapter answers the next question: how are all of those weights trained together?

The answer is backpropagation. Backpropagation is not a separate learning rule. It is reverse-mode differentiation applied to a network written as a composition of matrix operations and element-wise non-linearities. The optimizer still performs gradient descent or one of its variants. Backpropagation supplies the gradients that make the optimizer possible.

This chapter develops backpropagation in three layers:

1. **Mathematics.** We derive the backward pass for dense linear layers, activation functions, and softmax cross-entropy using explicit matrix shapes. We then analyze why gradients vanish or explode when many Jacobians are multiplied through depth.
2. **Implementation.** We build a small deep learning library in pure Python/NumPy: `Module`, `Parameter`, `Linear`, `ReLU`, `Sigmoid`, `Sequential`, `CrossEntropyLoss`, and an SGD training loop.
3. **Verification and diagnostics.** We prove that the implementation is correct by numerical finite differences, then reproduce the same model in PyTorch and inspect activation distributions, gradient norms, and loss curves.

The goal is not to replace mature deep learning libraries. The goal is to remove the mystery from them. By the end of the chapter, `loss.backward()` should no longer feel like magic: it should feel like a compact name for a sequence of matrix multiplications you can derive and test yourself.

---

## Section 1: Learning Objectives

By the end of this chapter, you will be able to:

**1.1 Derive Matrix Backpropagation.**
Given a dense layer

$$
\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^\top
$$

derive the gradients

$$
\frac{\partial L}{\partial \mathbf{X}} =
\frac{\partial L}{\partial \mathbf{Z}}\mathbf{W}^\top
$$

$$
\frac{\partial L}{\partial \mathbf{W}} =
\mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Z}}
$$

$$
\frac{\partial L}{\partial \mathbf{b}} =
\sum_{i=1}^{B}
\frac{\partial L}{\partial \mathbf{Z}_{i,:}}
$$

with all dimensions checked.

**1.2 Explain Gradient Flow.**
Trace how gradient signals move backward through a stack of Jacobians and explain why repeated multiplication by derivatives less than one causes vanishing gradients, while repeated multiplication by large singular values causes exploding gradients.

**1.3 Derive Initialization Rules.**
Derive the variance scaling behind Xavier/Glorot initialization and He/Kaiming initialization by preserving activation variance and gradient variance across depth.

**1.4 Implement a Modular MLP.**
Build a minimal deep learning library with reusable layer objects, cached forward-pass values, backward methods, parameter registration, gradient accumulation, and SGD updates.

**1.5 Verify Gradients Numerically.**
Implement central finite difference checking

$$
f'(\theta_j) \approx
\frac{f(\theta_j + h) - f(\theta_j - h)}{2h}
$$

and compare numerical gradients to analytical backpropagation gradients using relative error.

**1.6 Use PyTorch With Understanding.**
Construct the same architecture in PyTorch, inspect its parameters, call `loss.backward()`, and connect the resulting `.grad` tensors to the equations derived in this chapter.

---

## Section 2: Prerequisites

This chapter assumes the following material from earlier chapters.

### 2.1 From Automatic Differentiation

You must understand the scalar chain rule:

$$
\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)
$$

Backpropagation is this rule applied to many variables at once. Instead of multiplying scalar derivatives, we multiply Jacobians and vector-Jacobian products.

### 2.2 From Linear Algebra

You must be comfortable with:

- Matrix multiplication dimensions.
- Transposes.
- Row-wise batches of examples.
- Dot products as weighted sums.
- Jacobians as matrices of partial derivatives.

Throughout this chapter we use the following convention:

- $\mathbf{X} \in \mathbb{R}^{B \times D_{\text{in}}}$ is a batch of $B$ examples.
- $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ is a dense weight matrix.
- $\mathbf{b} \in \mathbb{R}^{D_{\text{out}}}$ is a bias vector.
- $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$ is the pre-activation matrix.
- $\mathbf{A} = \phi(\mathbf{Z})$ is the post-activation matrix.
- $L$ is a scalar loss.

### 2.3 From Numerical Computing

You must understand why floating-point arithmetic is approximate. This matters because gradient checking compares two approximate quantities: the analytical gradient and the finite-difference gradient. The correct threshold is small but not zero.

### 2.4 From Neural Spaces

You must understand why a neural network needs non-linear activations. If every layer is linear, then

$$
\mathbf{X}\mathbf{W}_1\mathbf{W}_2\cdots\mathbf{W}_L
= \mathbf{X}\mathbf{W}_{\text{eff}}
$$

which is just one linear layer. Backpropagation trains the non-linear composition, but the non-linearity is what makes the architecture expressive.

---

## Section 3: Motivation

### 3.1 Why the Forward Pass Is Not Enough

A feedforward network computes predictions by composing layers:

$$
\hat{\mathbf{Y}} =
f_\theta(\mathbf{X})
= f_L(f_{L-1}(\cdots f_1(\mathbf{X})))
$$

The parameters $\theta$ include all weights and biases. To train the network we choose a loss function $L(\theta)$ and update parameters in the direction of steepest decrease:

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

The entire problem is now compressed into one object: $\nabla_\theta L$. For a small linear model, this gradient can be derived in one line. For a deep network with thousands, millions, or billions of parameters, computing every derivative independently would be prohibitively expensive.

Backpropagation solves this by reusing intermediate derivatives. The forward pass stores local values needed for differentiation. The backward pass moves from the loss back to the input, applying the chain rule one layer at a time.

### 3.2 The Reuse Principle

Suppose a network has three layers:

$$
\mathbf{H}_1 = f_1(\mathbf{X}), \quad
\mathbf{H}_2 = f_2(\mathbf{H}_1), \quad
L = f_3(\mathbf{H}_2)
$$

The gradient with respect to the first layer's parameters contains the derivatives of all later layers:

$$
\frac{\partial L}{\partial \theta_1}
=
\frac{\partial L}{\partial \mathbf{H}_2}
\frac{\partial \mathbf{H}_2}{\partial \mathbf{H}_1}
\frac{\partial \mathbf{H}_1}{\partial \theta_1}
$$

Once $\partial L / \partial \mathbf{H}_2$ has been computed, it can be reused for every parameter inside layer 2 and for every earlier layer. This dynamic programming structure is the reason backpropagation is efficient.

### 3.3 Why Backpropagation Is Often Misunderstood

Backpropagation is sometimes described as "sending error backward." That phrase is useful but incomplete. The backward signal is not just an error. It is a gradient: the sensitivity of the scalar loss to each intermediate variable. At layer $\ell$, the backward input is

$$
\frac{\partial L}{\partial \mathbf{A}^{(\ell)}}
$$

which means: if the activation at layer $\ell$ changed by a tiny amount, how would the loss change?

That interpretation is central. A gradient is a local sensitivity measure, not a blame assignment. Large gradients identify directions where small changes strongly affect the loss. Small gradients identify directions where the loss is locally insensitive.

---

## Section 4: The Multidimensional Chain Rule

### 4.1 Scalar Chain Rule

For scalar variables:

$$
y = f(u), \quad u = g(x)
$$

the chain rule is

$$
\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}
$$

The derivative $dy/du$ tells how $y$ changes when $u$ changes. The derivative $du/dx$ tells how $u$ changes when $x$ changes. Multiplying them gives how $y$ changes when $x$ changes through $u$.

### 4.2 Vector Chain Rule

Let

$$
\mathbf{y} = f(\mathbf{u}), \quad \mathbf{u} = g(\mathbf{x})
$$

where $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{u} \in \mathbb{R}^m$, and $\mathbf{y} \in \mathbb{R}^k$.

The Jacobian of $\mathbf{u}$ with respect to $\mathbf{x}$ is

$$
\mathbf{J}_{g}
=
\frac{\partial \mathbf{u}}{\partial \mathbf{x}}
\in \mathbb{R}^{m \times n}
$$

The Jacobian of $\mathbf{y}$ with respect to $\mathbf{u}$ is

$$
\mathbf{J}_{f}
=
\frac{\partial \mathbf{y}}{\partial \mathbf{u}}
\in \mathbb{R}^{k \times m}
$$

The composite Jacobian is

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
=
\mathbf{J}_f \mathbf{J}_g
\in \mathbb{R}^{k \times n}
$$

For neural network training, the output of interest is usually a scalar loss $L$. Then the gradient with respect to an intermediate vector $\mathbf{u}$ is a row or column vector depending on convention. Rather than materializing full Jacobians, backpropagation repeatedly computes vector-Jacobian products.

### 4.3 Why We Avoid Full Jacobians

Consider a layer mapping a batch matrix $\mathbf{X} \in \mathbb{R}^{B \times D}$ to $\mathbf{Z} \in \mathbb{R}^{B \times H}$. If flattened, the Jacobian

$$
\frac{\partial \text{vec}(\mathbf{Z})}{\partial \text{vec}(\mathbf{X})}
$$

has shape $(BH) \times (BD)$. For realistic batches and hidden sizes this matrix is enormous. Fortunately, the backward pass does not need the full Jacobian. It needs the product of the upstream gradient with the local Jacobian. This product has the same shape as the layer input or parameter being differentiated.

The practical rule is:

> Do not build the Jacobian when the layer structure lets you compute the vector-Jacobian product directly.

Dense layers, convolutions, activations, normalizations, and attention all exploit this rule.

---

## Section 5: Backward Pass for a Linear Layer

### 5.1 Forward Definition

For a dense layer:

$$
\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^\top
$$

with dimensions:

- $\mathbf{X} \in \mathbb{R}^{B \times D_{\text{in}}}$
- $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$
- $\mathbf{b} \in \mathbb{R}^{D_{\text{out}}}$
- $\mathbf{Z} \in \mathbb{R}^{B \times D_{\text{out}}}$

Elementwise:

$$
Z_{ij} = \sum_{k=1}^{D_{\text{in}}} X_{ik}W_{kj} + b_j
$$

Let the upstream gradient be

$$
\mathbf{G} =
\frac{\partial L}{\partial \mathbf{Z}}
\in \mathbb{R}^{B \times D_{\text{out}}}
$$

where $G_{ij} = \partial L / \partial Z_{ij}$.

### 5.2 Gradient With Respect to Input

We need $\partial L / \partial X_{ik}$. Since

$$
Z_{ij} = \sum_{k=1}^{D_{\text{in}}} X_{ik}W_{kj} + b_j
$$

the derivative of $Z_{ij}$ with respect to $X_{ik}$ is

$$
\frac{\partial Z_{ij}}{\partial X_{ik}} = W_{kj}
$$

Therefore:

$$
\frac{\partial L}{\partial X_{ik}}
=
\sum_{j=1}^{D_{\text{out}}}
\frac{\partial L}{\partial Z_{ij}}
\frac{\partial Z_{ij}}{\partial X_{ik}}
=
\sum_{j=1}^{D_{\text{out}}} G_{ij}W_{kj}
$$

This is exactly the matrix product:

$$
\frac{\partial L}{\partial \mathbf{X}}
=
\mathbf{G}\mathbf{W}^\top
$$

Dimension check:

$$
(B \times D_{\text{out}})
(D_{\text{out}} \times D_{\text{in}})
=
B \times D_{\text{in}}
$$

### 5.3 Gradient With Respect to Weights

We need $\partial L / \partial W_{kj}$. Since

$$
\frac{\partial Z_{ij}}{\partial W_{kj}} = X_{ik}
$$

we get

$$
\frac{\partial L}{\partial W_{kj}}
=
\sum_{i=1}^{B}
\frac{\partial L}{\partial Z_{ij}}
\frac{\partial Z_{ij}}{\partial W_{kj}}
=
\sum_{i=1}^{B} X_{ik}G_{ij}
$$

This is the matrix product:

$$
\frac{\partial L}{\partial \mathbf{W}}
=
\mathbf{X}^\top \mathbf{G}
$$

Dimension check:

$$
(D_{\text{in}} \times B)
(B \times D_{\text{out}})
=
D_{\text{in}} \times D_{\text{out}}
$$

### 5.4 Gradient With Respect to Bias

Since $Z_{ij}$ depends on $b_j$ with derivative 1:

$$
\frac{\partial Z_{ij}}{\partial b_j} = 1
$$

Therefore:

$$
\frac{\partial L}{\partial b_j}
=
\sum_{i=1}^{B}
\frac{\partial L}{\partial Z_{ij}}
=
\sum_{i=1}^{B} G_{ij}
$$

In vector form:

$$
\frac{\partial L}{\partial \mathbf{b}}
=
\sum_{\text{rows}} \mathbf{G}
$$

The result has shape $D_{\text{out}}$.

### 5.5 The Linear Layer Backward Contract

Given upstream gradient $\mathbf{G}$, a linear layer returns:

$$
d\mathbf{X} = \mathbf{G}\mathbf{W}^\top
$$

and accumulates:

$$
d\mathbf{W} = \mathbf{X}^\top \mathbf{G}
$$

$$
d\mathbf{b} = \sum_{\text{rows}}\mathbf{G}
$$

This is the central derivation of this chapter. Most of deep learning is this pattern repeated under different layer structures.

---

## Section 6: Backward Pass for Activation Functions

Activation functions are applied elementwise. If

$$
\mathbf{A} = \phi(\mathbf{Z})
$$

then

$$
A_{ij} = \phi(Z_{ij})
$$

Given upstream gradient

$$
\mathbf{G} = \frac{\partial L}{\partial \mathbf{A}}
$$

the chain rule gives:

$$
\frac{\partial L}{\partial Z_{ij}}
=
\frac{\partial L}{\partial A_{ij}}
\phi'(Z_{ij})
=
G_{ij}\phi'(Z_{ij})
$$

In matrix notation:

$$
\frac{\partial L}{\partial \mathbf{Z}}
=
\mathbf{G} \odot \phi'(\mathbf{Z})
$$

where $\odot$ denotes elementwise multiplication.

### 6.1 ReLU

The rectified linear unit is

$$
\text{ReLU}(z) = \max(0,z)
$$

Its derivative is

$$
\text{ReLU}'(z) =
\begin{cases}
1, & z > 0 \\
0, & z < 0
\end{cases}
$$

At $z=0$ the derivative is undefined. Implementations usually choose 0 or 1. This does not matter in ordinary floating-point training because hitting exactly zero has probability near zero under continuous weights and inputs.

Backward pass:

$$
d\mathbf{Z} = d\mathbf{A} \odot \mathbb{1}[\mathbf{Z} > 0]
$$

ReLU preserves gradient magnitude for active units and blocks it for inactive units. This is why ReLU helped deep networks train: active neurons do not multiply the gradient by a small sigmoid derivative.

### 6.2 Sigmoid

The sigmoid function is

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Its derivative is

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

If the forward pass cached $\mathbf{A}=\sigma(\mathbf{Z})$, the backward pass can compute:

$$
d\mathbf{Z} = d\mathbf{A} \odot \mathbf{A} \odot (1 - \mathbf{A})
$$

The maximum value of $\sigma'(z)$ is $1/4$, attained at $z=0$. For large positive or negative $z$, the derivative approaches zero. This saturation is a direct cause of vanishing gradients.

### 6.3 Tanh

The hyperbolic tangent is

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

Its derivative is

$$
\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)
$$

Tanh is zero-centered, unlike sigmoid, but it still saturates for large $|z|$.

### 6.4 GELU

The Gaussian Error Linear Unit is often written as:

$$
\text{GELU}(z) = z\Phi(z)
$$

where $\Phi$ is the standard normal CDF. A common approximation is:

$$
\text{GELU}(z)
\approx
0.5z\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right)\right)
$$

GELU is smoother than ReLU and is common in Transformer architectures. For this chapter's implementation, ReLU and sigmoid are enough to expose the mechanics of backpropagation.

---

## Section 7: Softmax Cross-Entropy

### 7.1 Why Softmax Is Used for Multi-Class Classification

For $K$ classes, a network typically outputs logits

$$
\mathbf{S} \in \mathbb{R}^{B \times K}
$$

The logits are unnormalized scores. Softmax converts them into probabilities:

$$
P_{ik} =
\frac{e^{S_{ik}}}{\sum_{j=1}^{K}e^{S_{ij}}}
$$

Each row sums to 1.

### 7.2 Numerical Stability

Direct exponentiation can overflow. Since softmax is invariant to adding or subtracting the same constant from every logit in a row:

$$
\text{softmax}(\mathbf{s})
=
\text{softmax}(\mathbf{s} - c)
$$

we choose

$$
c = \max_j s_j
$$

and compute:

$$
p_k =
\frac{e^{s_k - \max_j s_j}}{\sum_m e^{s_m - \max_j s_j}}
$$

This prevents exponentials from becoming too large.

### 7.3 Cross-Entropy Loss

For one example with one-hot target vector $\mathbf{y}$:

$$
L = -\sum_{k=1}^{K} y_k \log p_k
$$

If the true class is $c$, then $y_c=1$ and all other entries are zero, so:

$$
L = -\log p_c
$$

For a batch of $B$ examples:

$$
L =
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_{i,y_i}
$$

### 7.4 Gradient of Softmax Cross-Entropy

The most important result:

$$
\frac{\partial L}{\partial S_{ik}}
=
\frac{1}{B}(P_{ik} - Y_{ik})
$$

where $\mathbf{Y}$ is the one-hot target matrix.

This elegant expression appears because the derivative of softmax and the derivative of cross-entropy simplify when combined. In practice, this is why deep learning libraries combine `softmax` and `cross_entropy` into one numerically stable operation.

### 7.5 Derivation for One Example

For one example:

$$
p_k = \frac{e^{s_k}}{\sum_j e^{s_j}}
$$

Taking logs:

$$
\log p_c = s_c - \log\sum_j e^{s_j}
$$

The negative log-likelihood is:

$$
L = -s_c + \log\sum_j e^{s_j}
$$

Differentiate with respect to logit $s_k$:

$$
\frac{\partial L}{\partial s_k}
=
-\mathbb{1}[k=c]
+
\frac{e^{s_k}}{\sum_j e^{s_j}}
$$

Therefore:

$$
\frac{\partial L}{\partial s_k}
=
p_k - y_k
$$

Averaging over $B$ examples divides the gradient by $B$.

---

## Section 8: Gradient Flow Pathologies

### 8.1 Backpropagation Through Many Layers

Consider a deep network:

$$
\mathbf{h}^{(L)}
=
f_L(f_{L-1}(\cdots f_1(\mathbf{x})))
$$

The gradient with respect to an early hidden state contains a product of Jacobians:

$$
\frac{\partial L}{\partial \mathbf{h}^{(1)}}
=
\frac{\partial L}{\partial \mathbf{h}^{(L)}}
\mathbf{J}_{L}
\mathbf{J}_{L-1}
\cdots
\mathbf{J}_{2}
$$

If the norms of these Jacobians are consistently less than 1, the gradient norm shrinks exponentially with depth. If they are consistently greater than 1, the gradient norm grows exponentially.

### 8.2 Vanishing Gradients

Suppose every layer contributes a derivative factor around $0.25$, as sigmoid can near its maximum derivative. After 20 layers:

$$
0.25^{20} \approx 9.09 \times 10^{-13}
$$

The early layers receive almost no learning signal. Mathematically, this occurs because the gradient flow depends on the product of Jacobians. Under activation functions like the sigmoid, the derivative is bounded by $\sigma'(z) \le 0.25$. When these bounded derivatives are multiplied across depth, the signal decays exponentially, rendering the early feature-extraction layers unable to update their parameters.

![Vanishing Gradients: Sigmoid vs. ReLU](source_material/chapter10/images/vanishing_gradients.png)

This visual graph demonstrates how the average gradient norm collapses exponentially when using Sigmoid activations, compared to the stable gradient norms preserved under ReLU activations. The network may still reduce training loss by adjusting later layers, but the earliest representation layers remain close to random.

### 8.3 Exploding Gradients

If each Jacobian has a norm around 1.5, then after 20 layers:

$$
1.5^{20} \approx 3325
$$

Gradients can become extremely large, causing unstable parameter updates and numerical overflow. Gradient clipping is a common mitigation, especially in recurrent networks.

### 8.4 Dead ReLUs

ReLU avoids saturation for positive pre-activations, but it has another failure mode. If a ReLU neuron's pre-activation is negative for every training example, its derivative is zero for every example:

$$
\text{ReLU}'(z)=0 \quad \text{for } z<0
$$

The neuron receives no gradient and may never recover. This is called a dead ReLU. Causes include overly large learning rates, poor initialization, or input distributions that push many pre-activations negative.

### 8.5 Diagnostics

A practical training loop should inspect:

- **Activation distributions.** Are many activations saturated or zero?
- **Gradient norms.** Are early-layer gradients near zero or exploding?
- **Parameter norms.** Are weights growing without bound?
- **Loss curves.** Is training loss decreasing smoothly, diverging, or flat?
- **Train-validation gap.** Is the model underfitting or overfitting?

These diagnostics are not optional decoration. They are how engineers determine whether a deep network is learning for the right reason.

---

## Section 9: Weight Initialization

### 9.1 The Initialization Problem

If weights are initialized too small, activations and gradients shrink as they move through the network. If weights are initialized too large, activations and gradients explode. The goal is to choose an initialization scale that keeps signal variance approximately stable across layers.

### 9.2 Forward Variance Calculation

Consider one pre-activation:

$$
z_j = \sum_{i=1}^{n_{\text{in}}} x_i w_{ij}
$$

Assume:

- $x_i$ are independent with mean 0 and variance $\text{Var}(x)$.
- $w_{ij}$ are independent with mean 0 and variance $\text{Var}(w)$.
- Inputs and weights are independent.

Then:

$$
\text{Var}(z_j)
=
\sum_{i=1}^{n_{\text{in}}}
\text{Var}(x_i w_{ij})
$$

Since the terms are independent and zero mean:

$$
\text{Var}(x_i w_{ij})
=
\text{Var}(x_i)\text{Var}(w_{ij})
$$

Therefore:

$$
\text{Var}(z_j)
=
n_{\text{in}}\text{Var}(x)\text{Var}(w)
$$

To preserve variance forward, we want:

$$
n_{\text{in}}\text{Var}(w) \approx 1
$$

so:

$$
\text{Var}(w) \approx \frac{1}{n_{\text{in}}}
$$

### 9.3 Xavier/Glorot Initialization

Forward preservation suggests:

$$
\text{Var}(w) = \frac{1}{n_{\text{in}}}
$$

Backward preservation suggests:

$$
\text{Var}(w) = \frac{1}{n_{\text{out}}}
$$

Xavier initialization balances both:

$$
\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

For a uniform distribution $U(-a,a)$:

$$
\text{Var}(U(-a,a)) = \frac{a^2}{3}
$$

Set:

$$
\frac{a^2}{3}
=
\frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

so:

$$
a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}
$$

This is commonly used with tanh or approximately linear activations.

### 9.4 He/Kaiming Initialization

ReLU sets roughly half of its inputs to zero when pre-activations are symmetric around zero. This halves the variance of the activation signal. To compensate, He initialization doubles the variance:

$$
\text{Var}(w) = \frac{2}{n_{\text{in}}}
$$

For a normal distribution:

$$
w_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
$$

For ReLU networks, this usually gives better gradient flow than Xavier initialization.

![Weight Initialization Variance](source_material/chapter10/images/weight_initialization.png)

This visual plot contrasts how activation variance propagates through successive layers under standard small random initialization versus correctly scaled Xavier/He initialization. Under small random initialization, the signal variance collapses rapidly toward zero, preventing subsequent layers from extracting useful features.

---

## Section 10: A Minimal NumPy Deep Learning Library

This section builds a compact implementation suitable for study. It is not optimized for production. It is designed so every line maps back to a mathematical equation.

### 10.1 Parameter Object

```python
import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad.fill(0.0)
```

Each trainable tensor stores two arrays:

- `data`: the current parameter value.
- `grad`: the accumulated derivative of the loss with respect to `data`.

### 10.2 Module Base Class

```python
class Module:
    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
```

Every layer follows the same contract:

- `forward` computes outputs and caches whatever the backward pass needs.
- `backward` receives upstream gradients and returns downstream gradients.
- `parameters` exposes trainable parameters to the optimizer.

### 10.3 Linear Layer

```python
class Linear(Module):
    def __init__(self, in_features, out_features, initialization="he", rng=None):
        self.in_features = in_features
        self.out_features = out_features
        self.rng = np.random.default_rng() if rng is None else rng

        if initialization == "he":
            scale = np.sqrt(2.0 / in_features)
            weight = self.rng.normal(0.0, scale, size=(in_features, out_features))
        elif initialization == "xavier":
            limit = np.sqrt(6.0 / (in_features + out_features))
            weight = self.rng.uniform(-limit, limit, size=(in_features, out_features))
        else:
            raise ValueError("initialization must be 'he' or 'xavier'")

        bias = np.zeros(out_features, dtype=np.float64)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.input = None

    def forward(self, x):
        self.input = x
        return x @ self.weight.data + self.bias.data

    def backward(self, grad_output):
        self.weight.grad += self.input.T @ grad_output
        self.bias.grad += grad_output.sum(axis=0)
        return grad_output @ self.weight.data.T

    def parameters(self):
        return [self.weight, self.bias]
```

The three backward lines correspond exactly to:

$$
d\mathbf{W} = \mathbf{X}^\top d\mathbf{Z}
$$

$$
d\mathbf{b} = \sum_{\text{rows}} d\mathbf{Z}
$$

$$
d\mathbf{X} = d\mathbf{Z}\mathbf{W}^\top
$$

### 10.4 ReLU

```python
class ReLU(Module):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0.0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0.0)
```

### 10.5 Sigmoid

```python
class Sigmoid(Module):
    def __init__(self):
        self.output = None

    def forward(self, x):
        out = np.empty_like(x, dtype=np.float64)
        positive = x >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        out[~positive] = exp_x / (1.0 + exp_x)
        self.output = out
        return out

    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)
```

The implementation uses a stable sigmoid formula to avoid overflow for large negative inputs.

### 10.6 Sequential Container

```python
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
```

The backward pass reverses the forward order because the chain rule works from the output back to the input.

### 10.7 Cross-Entropy Loss

```python
class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        targets = np.asarray(targets, dtype=np.int64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        batch_size = logits.shape[0]
        losses = -np.log(probs[np.arange(batch_size), targets] + 1e-15)

        self.probs = probs
        self.targets = targets
        return losses.mean()

    def backward(self):
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.targets] -= 1.0
        grad /= batch_size
        return grad
```

The backward method returns:

$$
\frac{1}{B}(\mathbf{P} - \mathbf{Y})
$$

### 10.8 SGD Optimizer

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
```

### 10.9 Training Loop

```python
def accuracy(logits, targets):
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == targets)


def train_epoch(model, loss_fn, optimizer, x, y):
    optimizer.zero_grad()
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)
    grad_logits = loss_fn.backward()
    model.backward(grad_logits)
    optimizer.step()
    return loss, accuracy(logits, y)
```

This full loop has four stages:

1. Clear old gradients.
2. Run the forward pass.
3. Run the backward pass.
4. Update parameters.

Forgetting stage 1 is a common bug. Gradients accumulate by design, so stale gradients must be cleared before each independent update unless accumulation is intentional.

---

## Section 11: Numerical Gradient Checking

### 11.1 Central Difference Approximation

For a scalar function $f(\theta)$:

$$
f'(\theta)
\approx
\frac{f(\theta+h) - f(\theta-h)}{2h}
$$

This is more accurate than the forward difference:

$$
\frac{f(\theta+h)-f(\theta)}{h}
$$

because first-order truncation errors cancel.

### 11.2 Relative Error

Absolute error can be misleading when gradients are tiny. Use relative error:

$$
\text{relative error}
=
\frac{|g_{\text{analytical}} - g_{\text{numerical}}|}
{\max(1, |g_{\text{analytical}}|, |g_{\text{numerical}}|)}
$$

For a correct float64 implementation, many gradients should pass below $10^{-7}$.

### 11.3 Gradient Check Implementation

```python
def gradient_check(model, loss_fn, x, y, h=1e-5, tolerance=1e-7):
    model.zero_grad()
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)
    grad_logits = loss_fn.backward()
    model.backward(grad_logits)

    max_relative_error = 0.0
    worst = None

    for param_index, param in enumerate(model.parameters()):
        analytical = param.grad.copy()
        numerical = np.zeros_like(param.data)

        it = np.nditer(param.data, flags=["multi_index"], op_flags=["readwrite"])
        for value in it:
            idx = it.multi_index
            original = float(value)

            param.data[idx] = original + h
            loss_plus = loss_fn.forward(model.forward(x), y)

            param.data[idx] = original - h
            loss_minus = loss_fn.forward(model.forward(x), y)

            param.data[idx] = original

            numerical[idx] = (loss_plus - loss_minus) / (2.0 * h)

        denom = np.maximum(1.0, np.maximum(np.abs(analytical), np.abs(numerical)))
        rel_error = np.abs(analytical - numerical) / denom
        param_max = float(rel_error.max())

        if param_max > max_relative_error:
            max_relative_error = param_max
            worst = (param_index, np.unravel_index(np.argmax(rel_error), rel_error.shape))

    if max_relative_error > tolerance:
        raise AssertionError(
            f"gradient check failed: max_relative_error={max_relative_error:.3e}, "
            f"worst={worst}"
        )

    return max_relative_error
```

### 11.4 What Gradient Checking Can and Cannot Prove

Gradient checking can prove that the implemented backward pass matches the implemented forward loss locally. It cannot prove that:

- The model architecture is appropriate.
- The loss function is the right one for the task.
- The training loop will converge.
- The data preprocessing is correct.
- The model generalizes.

It is a correctness test for derivatives, not a validation test for the whole machine learning project.

---

## Section 12: Example - A Non-Linear Decision Boundary

### 12.1 Synthetic Two-Spiral Data

```python
def make_spirals(n_per_class=100, noise=0.2, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    n = n_per_class
    theta = np.sqrt(rng.random(n)) * 2.0 * np.pi

    r_a = 2.0 * theta + np.pi
    x_a = np.column_stack([np.cos(theta) * r_a, np.sin(theta) * r_a])

    r_b = -2.0 * theta - np.pi
    x_b = np.column_stack([np.cos(theta) * r_b, np.sin(theta) * r_b])

    x = np.vstack([x_a, x_b])
    x += rng.normal(0.0, noise, size=x.shape)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x, y
```

The two-spiral problem is deliberately non-linear. A linear classifier cannot separate the classes. A small MLP can learn a curved decision boundary by transforming the input into hidden features.

![Synthetic Two-Spiral Dataset](source_material/chapter10/images/spiral_dataset.png)

### 12.2 Model

```python
rng = np.random.default_rng(42)
x, y = make_spirals(n_per_class=200, noise=0.25, rng=rng)

model = Sequential(
    Linear(2, 32, initialization="he", rng=rng),
    ReLU(),
    Linear(32, 32, initialization="he", rng=rng),
    ReLU(),
    Linear(32, 2, initialization="xavier", rng=rng),
)

loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.05)

for epoch in range(1000):
    loss, acc = train_epoch(model, loss_fn, optimizer, x, y)
    if epoch % 100 == 0:
        print(f"epoch={epoch:04d} loss={loss:.4f} acc={acc:.3f}")
```

This example is intentionally small enough for inspection. If the model fails to learn, inspect:

- Whether gradients pass the finite-difference check.
- Whether the learning rate is too high or too low.
- Whether activations are mostly dead.
- Whether the data was standardized.
- Whether the final layer has the correct number of class logits.

### 12.3 Decision Boundary Grid

```python
def decision_grid(model, x, grid_size=200, padding=0.5):
    x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
    y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    logits = model.forward(grid)
    pred = np.argmax(logits, axis=1).reshape(xx.shape)
    return xx, yy, pred
```

Plotting `pred` as a colored background and the training points on top reveals the learned non-linear boundary.

![Learned Non-Linear Decision Boundary](source_material/chapter10/images/decision_boundary.png)

---

## Section 13: PyTorch Equivalent

### 13.1 Model Definition

```python
import torch
from torch import nn


class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)
```

This is the PyTorch version of the scratch `Sequential` model. The main difference is that PyTorch builds a dynamic computation graph during the forward pass and automatically applies reverse-mode differentiation when `loss.backward()` is called.

### 13.2 Training Loop

```python
x_t = torch.tensor(x, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

model_t = TorchMLP()
criterion = nn.CrossEntropyLoss()
optimizer_t = torch.optim.SGD(model_t.parameters(), lr=0.05)

for epoch in range(1000):
    optimizer_t.zero_grad()
    logits = model_t(x_t)
    loss = criterion(logits, y_t)
    loss.backward()
    optimizer_t.step()

    if epoch % 100 == 0:
        pred = logits.argmax(dim=1)
        acc = (pred == y_t).float().mean().item()
        print(f"epoch={epoch:04d} loss={loss.item():.4f} acc={acc:.3f}")
```

The PyTorch loop has the same four stages:

1. Clear gradients.
2. Forward pass.
3. Backward pass.
4. Parameter update.

### 13.3 Inspecting Gradients

```python
for name, param in model_t.named_parameters():
    if param.grad is not None:
        print(name, param.grad.norm().item())
```

This prints a gradient norm for every parameter tensor. Compare this to the scratch implementation's `param.grad` arrays. Conceptually, they are the same object: derivatives of the scalar loss with respect to trainable parameters.

### 13.4 Autograd Graph

PyTorch tensors with `requires_grad=True` record operations in a dynamic graph. When a scalar loss calls `.backward()`, PyTorch traverses that graph in reverse topological order. Each operation has a registered backward rule, equivalent to the `backward` methods implemented earlier in this chapter.

The scratch implementation stores caches manually. PyTorch stores them as part of the autograd graph.

---

## Section 14: Optimization and Diagnostics

### 14.1 SGD

Stochastic gradient descent updates parameters by:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L_t
$$

where $L_t$ is the loss on the current batch. Full-batch gradient descent uses all training examples for every update. Mini-batch SGD uses a subset. Mini-batching introduces noise but improves computational efficiency and often helps optimization escape shallow poor regions.

### 14.2 Mini-Batch Training

```python
def iterate_minibatches(x, y, batch_size, rng):
    indices = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]
```

```python
for epoch in range(100):
    for xb, yb in iterate_minibatches(x, y, batch_size=32, rng=rng):
        train_epoch(model, loss_fn, optimizer, xb, yb)
```

### 14.3 Activation Distribution Tracking

For each activation layer, inspect the distribution of outputs:

```python
def collect_activations(model, x):
    activations = []
    current = x
    for layer in model.layers:
        current = layer.forward(current)
        if isinstance(layer, (ReLU, Sigmoid)):
            activations.append(current.copy())
    return activations
```

Warning signs:

- ReLU layers with nearly all zeros.
- Sigmoid layers with values nearly all 0 or 1.
- Activations whose magnitude grows layer by layer.
- Activations that collapse to constant values.

### 14.4 Gradient Norm Tracking

```python
def gradient_norms(model):
    norms = []
    for p in model.parameters():
        norms.append(float(np.linalg.norm(p.grad)))
    return norms
```

Warning signs:

- Early-layer norms are many orders of magnitude smaller than final-layer norms.
- Norms grow rapidly until overflow.
- Norms are exactly zero for many parameters.

### 14.5 Loss Curves

Training and validation loss should be interpreted together:

- Both high and flat: underfitting or optimization failure.
- Training decreases but validation increases: overfitting.
- Training loss becomes `nan`: numerical instability, too high a learning rate, or invalid log/exp operation.
- Training loss oscillates violently: learning rate likely too high.

Loss curves do not prove correctness, but they reveal failure modes quickly.

---

## Section 15: Common Implementation Bugs

### 15.1 Missing Batch Averaging

If the loss averages over the batch, the gradient must also average over the batch. In softmax cross-entropy this means:

```python
grad /= batch_size
```

If omitted, the effective learning rate scales with batch size.

### 15.2 Wrong Weight Orientation

This chapter uses:

$$
\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$

with $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$. Some libraries store weights as $D_{\text{out}} \times D_{\text{in}}$. Both conventions are valid, but the backward formulas must match the chosen convention.

### 15.3 Forgetting to Clear Gradients

Gradients accumulate:

```python
self.weight.grad += self.input.T @ grad_output
```

This is intentional because a parameter may contribute to the loss through multiple paths. But before a new training step, call `zero_grad()`.

### 15.4 Mutating Cached Inputs

Backward methods depend on cached forward-pass values. If those arrays are mutated before backward runs, gradients become wrong. Treat cached tensors as read-only.

### 15.5 Applying Softmax Twice

Cross-entropy implementations often expect logits, not probabilities. Passing softmax probabilities into a loss that already applies softmax produces incorrect gradients and worse numerical stability.

---

## Section 16: Exercises

### 16.1 Conceptual Exercises

1. Explain why backpropagation is an application of the chain rule rather than a new optimization algorithm.
2. Why does a sigmoid network suffer from vanishing gradients more easily than a ReLU network?
3. What does it mean for an activation function to saturate?
4. Why does the cross-entropy loss combine naturally with softmax?
5. Why is gradient checking too expensive to run during ordinary training?

### 16.2 Derivation Exercises

1. Derive $d\mathbf{X}$, $d\mathbf{W}$, and $d\mathbf{b}$ for $\mathbf{Z}=\mathbf{X}\mathbf{W}+\mathbf{b}$ using index notation.
2. Derive the sigmoid derivative $\sigma'(z)=\sigma(z)(1-\sigma(z))$.
3. Derive the softmax cross-entropy gradient $p_k-y_k$ for one example.
4. Starting from $\text{Var}(z)=n_{\text{in}}\text{Var}(x)\text{Var}(w)$, derive Xavier initialization.
5. Explain why He initialization uses $2/n_{\text{in}}$ for ReLU networks.

### 16.3 Coding Exercises

1. Add a `Tanh` module with forward and backward methods.
2. Add momentum to the SGD optimizer.
3. Implement L2 weight decay.
4. Write a gradient check that samples only 20 random parameter entries instead of checking every entry.
5. Train the scratch MLP on XOR and verify that a linear model fails.
6. Plot activation histograms for each hidden layer every 100 epochs.
7. Plot gradient norms for each parameter tensor during training.

### 16.4 Debugging Exercises

For each bug, predict the symptom before running the code:

1. Remove `grad /= batch_size` from `CrossEntropyLoss.backward`.
2. Replace `self.input.T @ grad_output` with `self.input @ grad_output`.
3. Forget to call `optimizer.zero_grad()`.
4. Initialize all weights to zero.
5. Use sigmoid activations in a 20-layer network with large random weights.
6. Use a learning rate of 100.0.

### 16.5 Research Reading Exercises

Read the original or standard references listed below and answer:

1. What training problem did the paper identify?
2. What mathematical explanation did it provide?
3. What empirical evidence supported the claim?
4. How did subsequent research validate, extend, or challenge the claim?
5. Which parts have been superseded?

### 16.6 Heavy-Hitting Research-Grade Exercises

1. **Backpropagation through a Convolutional Layer (Weight Sharing):**
   Consider a 1D convolution layer without padding, where the output is:
   $$z_k = \sum_{i=0}^{K-1} x_{k+i} w_i$$
   where $\mathbf{x} \in \mathbb{R}^N$ is the input vector, $\mathbf{w} \in \mathbb{R}^K$ is the filter weights, and $\mathbf{z} \in \mathbb{R}^{N-K+1}$ is the output.
   * Derive the analytical gradients for $\frac{\partial L}{\partial w_i}$ and $\frac{\partial L}{\partial x_j}$ using index notation.
   * Prove mathematically how the weight sharing constraint results in an accumulation (summation) of gradient signals over the input locations.

2. **Backpropagation through Layer Normalization:**
   Consider a Layer Normalization block that maps $\mathbf{x} \in \mathbb{R}^D$ to $\mathbf{y} \in \mathbb{R}^D$:
   $$\mu = \frac{1}{D}\sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D}\sum_{i=1}^D (x_i - \mu)^2, \quad \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i$$
   * Derive the complete analytical gradient $\frac{\partial L}{\partial x_i}$ in terms of the incoming gradient $\frac{\partial L}{\partial y_i}$ and parameters $\gamma_i, \beta_i$. Show every intermediate derivative step, including the terms for $\frac{\partial L}{\partial \sigma^2}$ and $\frac{\partial L}{\partial \mu}$.
   * Verify your analytical formula by writing a python function and testing it against numerical gradients.

3. **Residual Connections and Gradient Flow Limits:**
   Consider a deep residual network where the forward pass is defined as:
   $$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l)$$
   where $\mathbf{x}_l$ is the activation at layer $l$, and $\mathcal{F}$ is the residual block.
   * Prove by induction that for any deep layer $L > l$, the activation can be written as:
     $$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathbf{W}_i)$$
   * Derive the gradient of the loss $L$ with respect to $\mathbf{x}_l$:
     $$\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathbf{W}_i) \right)$$
   * Prove mathematically why this additive term prevents vanishing gradients even as the depth $L \to \infty$.

---

## Section 17: Project Deliverables

### 17.1 Modular MLP Library

Create `mlp_engine.py` containing:

- `Parameter`
- `Module`
- `Linear`
- `ReLU`
- `Sigmoid`
- `Sequential`
- `CrossEntropyLoss`
- `SGD`
- Initialization utilities
- Accuracy and mini-batch helpers

The file must have no dependency beyond NumPy.

### 17.2 Numerical Gradient Tester

Create `test_gradients.py` that:

- Builds a tiny network.
- Uses a tiny batch.
- Runs analytical backpropagation.
- Computes central finite-difference gradients.
- Fails if relative error exceeds $10^{-7}$.

Use float64 for this test.

### 17.3 Non-Linear Decision Boundary Solver

Create `train_spiral.py` that:

- Generates a two-spiral dataset.
- Trains the scratch MLP.
- Logs training loss and accuracy.
- Saves a decision boundary plot.
- Reports final train accuracy.

### 17.4 PyTorch Reproduction

Create `torch_spiral.py` that:

- Builds the same architecture in PyTorch.
- Trains with `torch.optim.SGD`.
- Logs loss and accuracy.
- Prints parameter gradient norms.
- Compares training behavior with the scratch implementation.

---

## Section 18: Summary

Backpropagation is reverse-mode differentiation applied to layered functions. For dense neural networks, the central result is the linear layer backward pass:

$$
d\mathbf{X}=d\mathbf{Z}\mathbf{W}^\top
$$

$$
d\mathbf{W}=\mathbf{X}^\top d\mathbf{Z}
$$

$$
d\mathbf{b}=\sum_{\text{rows}}d\mathbf{Z}
$$

Activation layers multiply by elementwise derivatives. Softmax cross-entropy simplifies to:

$$
d\mathbf{S} = \frac{1}{B}(\mathbf{P}-\mathbf{Y})
$$

Deep networks become difficult to train when products of Jacobians shrink or grow exponentially. Initialization rules, activation choices, normalization methods, residual connections, and optimizers are all responses to this gradient flow problem.

The practical engineering lesson is direct: never trust a custom backward pass until it passes a numerical gradient check. Once the gradients are correct, diagnose training with activation distributions, gradient norms, and loss curves.

---

## Section 19: Read More

- Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning*, Chapter 6: Deep Feedforward Networks.
- David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams, "Learning representations by back-propagating errors," *Nature*, 1986.
- Xavier Glorot and Yoshua Bengio, "Understanding the difficulty of training deep feedforward neural networks," AISTATS, 2010.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification," ICCV, 2015.
- Sepp Hochreiter, "Untersuchungen zu dynamischen neuronalen Netzen," Diploma thesis, 1991.
- Michael Nielsen, *Neural Networks and Deep Learning*, Chapters 2 and 3.
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 5.

