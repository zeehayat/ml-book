# Chapter 2 Video Tutorial — Narration Script

## 01. The Core Optimization Engine

Welcome to Chapter Two: The Core Optimization Engine, Automatic Differentiation. Every neural network learns by answering one question: if a parameter changed slightly, how would the loss change? Automatic differentiation answers that question for every parameter by recording ordinary arithmetic and replaying the chain rule in reverse. We will build that mechanism from first principles, trace a complete example, and connect our tiny scalar engine to the architecture used by PyTorch at production scale.

## 02. What you will learn

By the end, you should be able to turn an arithmetic expression into a directed acyclic graph, annotate its forward values, and propagate exact gradients backward. You will see why gradients must accumulate when paths merge, why reverse topological order is essential, and why reverse mode fits neural networks with millions or billions of inputs and one scalar loss. We will then translate the mathematics into a small Value class and examine what changes when values become multidimensional tensors.

## 03. Chapter 1 plus gradient state

Chapter One gave us a tensor as storage plus shape, strides, offset, and dtype. Autograd augments that object with gradient state. A scalar Value stores its forward data, the gradient of the final loss with respect to that data, pointers to the Values that produced it, and a backward function containing the local derivative rule. A production tensor keeps the Chapter One storage model and adds equivalent graph metadata. Autograd is therefore not separate magic layered above tensors; it is graph state woven into tensor operations.

## 04. Why derivatives must scale

Gradient descent updates each parameter by subtracting the learning rate times the derivative of loss with respect to that parameter. The formula is simple; obtaining the derivatives is the challenge. Hand derivation does not survive deep, changing architectures. Finite differences require a new forward evaluation for each parameter and introduce approximation error. Symbolic differentiation can create enormous expressions. Reverse-mode automatic differentiation records elementary operations, then computes all parameter gradients in one backward traversal whose cost is a small multiple of the forward computation.

## 05. Automatic differentiation is its own method

It is important to separate three ideas. Numerical differentiation perturbs an input and observes a finite difference, so the answer is approximate. Symbolic differentiation rewrites an expression into another expression, which is exact but may grow explosively. Automatic differentiation evaluates the actual program on concrete values. Every elementary operation records a small local derivative recipe. Combining those recipes with the chain rule produces an exact derivative of the implemented computation, up to ordinary floating-point rounding. If the program is wrong, autograd faithfully differentiates the wrong program.

## 06. A computational graph

We will carry one example through the forward and backward passes. Let f equal a plus b, multiplied by b plus one. Introduce intermediate nodes u equals a plus b, and v equals b plus one. Then f equals u times v. The graph is directed because values flow from inputs to output. It is acyclic because an operation cannot depend on a result that has not yet been computed. Notice that b fans out into two branches. That shared input will force us to add gradient contributions during the backward pass.

## 07. Forward pass: evaluate and record

Set a to two and b to three. The forward pass computes u equals five, v equals four, and f equals twenty. Alongside each result, the engine records its parents and a backward closure. The multiplication node remembers that its local derivative with respect to u is v, and with respect to v is u. The addition nodes remember derivative one for each input. This collection of nodes and local recipes is often called a tape or Wengert list: the breadcrumb trail needed to retrace the execution.

## 08. Backward pass: begin with one

The backward pass begins by setting the output gradient to one, because the derivative of f with respect to itself is one. At the multiplication f equals u times v, the arriving gradient is multiplied by each local partial derivative. U receives one times v, which is four. V receives one times u, which is five. These numbers are adjoints: u dot means partial f over partial u, and v dot means partial f over partial v. We then continue backward through the two addition nodes.

## 09. Shared nodes require accumulation

Variable b influences f along two distinct paths. Through u equals a plus b, it receives contribution four times one. Through v equals b plus one, it receives contribution five times one. The total derivative is the sum, nine. Variable a appears on only one path, so its derivative is four. This is the multivariable chain rule in graph form: whenever paths merge at a node, add their contributions. A backward closure must use plus equals, not assignment. Overwriting would silently discard one path and return the wrong gradient.

## 10. The multivariable chain rule

Formally, if x influences the loss through several downstream variables y sub j, the derivative of loss with respect to x is the sum over j of the upstream derivative with respect to y sub j times the local derivative of y sub j with respect to x. Every backward rule implements this same pattern: multiply the gradient arriving from downstream by a local derivative, then accumulate the result into the parent. The global derivative emerges without any node needing to understand the entire function.

## 11. Local backward rules

An autograd engine needs a derivative library for primitive operations. Addition sends the incoming gradient unchanged to both inputs. Multiplication scales it by the other operand. A power applies n times u to the n minus one. Tanh can reuse its forward output w, giving one minus w squared. ReLU passes the gradient only when its input was positive. Complicated neural networks are compositions of these simple rules. Supporting a new primitive means defining its forward computation and one correct vector-Jacobian product for backward.

## 12. Why reverse topological order matters

A node must not run its backward rule until every downstream consumer has contributed to its gradient. A topological sort places every parent before every child in forward order. Reversing that list places children before parents for backward. We first perform a depth-first traversal from the output, append each node after visiting its parents, seed the output gradient, and execute backward closures in reversed order. This guarantees that a shared node such as b has received all contributions before it propagates further upstream.

## 13. Forward mode versus reverse mode

Forward mode propagates input sensitivities alongside values and efficiently computes a Jacobian-vector product. It is attractive when there are few inputs and many outputs. Reverse mode propagates output sensitivities backward and computes a vector-Jacobian product. Neural network training has millions or billions of parameters but usually one scalar loss. That is many inputs and one output, so one reverse pass produces the complete parameter gradient. This dimensional asymmetry, not a special property of neural networks, is why backpropagation uses reverse mode.

## 14. The scalar Value object

The scalar engine begins with a Value object. Data stores the forward scalar. Grad starts at zero and accumulates the derivative of the final output with respect to this node. Prev stores parent nodes, while backward is a closure installed by the operation that created the Value. Calling backward topologically sorts the reachable graph, seeds the output gradient with one, and invokes every closure in reverse order. The graph is constructed dynamically as ordinary Python operators execute, which is the define-by-run model.

## 15. A multiplication closure

Multiplication creates an output Value and a closure that captures the two inputs and the output. During backward, the left input receives the right input's forward value times the output gradient. The right input receives the left value times the output gradient. Both updates use plus equals because either input may receive contributions from other branches. The closure architecture is powerful: each output node carries exactly the recipe needed to send its accumulated adjoint one step backward.

## 16. From scalars to tensors

Moving from scalars to tensors adds shape-aware backward rules. Suppose a three-by-four tensor x is added to a length-four bias b. Broadcasting repeats b across three rows during the forward pass. Backward must undo that repetition: x receives the output gradient unchanged, while b receives a sum across the broadcasted row axis. A helper often called sum to shape reduces extra leading axes and any axis whose original size was one. Matrix multiplication, transpose, reshape, and reductions each need similarly correct shape transformations.

## 17. The tape has a memory cost

Reverse mode is computationally efficient, but it must retain forward information until backward consumes it. Multiplication needs its input values. Activations such as tanh often save an output or mask. In deep networks, these saved tensors can exceed parameter memory. Gradient checkpointing reduces peak memory by saving only selected boundary activations. During backward, it recomputes missing forward regions, rebuilds their temporary tape, and then differentiates them. The trade is explicit: more computation in exchange for less memory.

## 18. Production autograd uses the same architecture

PyTorch's grad function corresponds to our backward closure, and saved tensors correspond to captured forward values. Define-by-run makes Python control flow natural because the graph reflects the path actually executed. Torch compile can capture regions of that dynamic program, optimize and fuse operations, then execute generated kernels. Mixed-precision training adds loss scaling to protect small gradients. Distributed training computes local backward passes and then reduces parameter gradients across devices. The scale changes dramatically, but the graph and chain-rule architecture does not.

## 19. Three failures that silently break learning

Autograd bugs are dangerous because code may run while learning fails. Overwriting rather than accumulating loses shared-path contributions. Forgetting zero grad unintentionally mixes gradients from different training steps. In-place mutation can make saved values inconsistent with the forward output they produced. Incorrect traversal order propagates a partial gradient too early. Retaining references to losses or graph nodes can keep entire tapes alive and exhaust memory. Unit-test tiny expressions against hand derivatives and finite-difference checks before trusting a larger engine.

## 20. Mini project: train an XOR MLP

The chapter's mini project composes scalar Values into neurons, layers, and a multilayer perceptron, then trains it on XOR. The forward pass produces predictions and a loss. Backward fills every parameter's grad field. Gradient descent updates each parameter and zeroes gradients before the next iteration. XOR is deliberately small but nonlinear: a single linear boundary cannot solve it. When the loss falls and predictions separate, you have evidence that graph construction, local derivatives, accumulation, ordering, and optimization all work together.

## 21. The four-primitive mental model

Compress automatic differentiation into four primitives. Record operations and dependencies during the forward pass. Store a local derivative recipe with every result. Traverse reachable nodes in reverse topological order. Multiply the incoming gradient by each local derivative and accumulate into parents. Reverse mode turns one scalar loss into gradients for every parameter with one backward traversal. Chapter One explained where tensor values live; Chapter Two explains how sensitivity flows between them. With this engine in place, Chapter Three can focus on optimization rather than derivative bookkeeping.
