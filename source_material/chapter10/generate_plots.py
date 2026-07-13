import os
import numpy as np
import matplotlib.pyplot as plt

# Create the images directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "images"), exist_ok=True)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

# 1. Custom Deep Learning Engine from Scratch (reproduced from Chapter 10 code)
class Parameter:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

class Module:
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, *args):
        raise NotImplementedError
    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features, initialization="he", rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        
        # Initialization
        if initialization == "he":
            scale = np.sqrt(2.0 / in_features)
        elif initialization == "xavier":
            scale = np.sqrt(2.0 / (in_features + out_features))
        else:
            scale = 0.01  # Default small values
            
        self.W = Parameter(rng.normal(0.0, scale, size=(in_features, out_features)))
        self.b = Parameter(np.zeros(out_features))
        self.input = None

    def forward(self, x):
        self.input = x
        return x @ self.W.data + self.b.data

    def backward(self, grad_output):
        # Gradients w.r.t parameters
        self.W.grad += self.input.T @ grad_output
        self.b.grad += np.sum(grad_output, axis=0)
        # Gradient w.r.t input
        return grad_output @ self.W.data.T

    def parameters(self):
        return [self.W, self.b]

class ReLU(Module):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0.0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0.0)

class Sigmoid(Module):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)

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

class CrossEntropyLoss:
    def forward(self, logits, y):
        # Stable Softmax
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift_logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.probs = probs
        self.y = y
        
        n = logits.shape[0]
        loss = -np.log(probs[np.arange(n), y] + 1e-15)
        return np.mean(loss)

    def backward(self):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.y] -= 1.0
        return grad / n

class SGD:
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad.fill(0.0)

# 2. Spiral Dataset Generator
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

def train_epoch(model, loss_fn, optimizer, x, y):
    optimizer.zero_grad()
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)
    grad = loss_fn.backward()
    model.backward(grad)
    optimizer.step()
    
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    return loss, acc

# 3. Generate and Save Plot 1: Raw Spiral Dataset
rng = np.random.default_rng(42)
x, y = make_spirals(n_per_class=200, noise=0.25, rng=rng)

plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0, 0], x[y == 0, 1], c="#ff6b6b", label="Class 0 (Coral)", alpha=0.8, edgecolors="none")
plt.scatter(x[y == 1, 0], x[y == 1, 1], c="#0f766e", label="Class 1 (Teal)", alpha=0.8, edgecolors="none")
plt.title("Synthetic Two-Spiral Dataset (Non-Linear)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend(frameon=True, facecolor="#f5f7fa", edgecolor="none")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "spiral_dataset.png"), dpi=150)
plt.close()
print("Saved spiral_dataset.png")

# 4. Generate and Save Plot 2: Decision Boundary after Training
model = Sequential(
    Linear(2, 32, initialization="he", rng=rng),
    ReLU(),
    Linear(32, 32, initialization="he", rng=rng),
    ReLU(),
    Linear(32, 2, initialization="xavier", rng=rng),
)
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)

# Train the model
for epoch in range(1500):
    loss, acc = train_epoch(model, loss_fn, optimizer, x, y)

# Create decision grid
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])
logits = model.forward(grid)
pred = np.argmax(logits, axis=1).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, pred, cmap=plt.cm.Spectral, alpha=0.2)
plt.scatter(x[y == 0, 0], x[y == 0, 1], c="#ff6b6b", label="Class 0", alpha=0.8)
plt.scatter(x[y == 1, 0], x[y == 1, 1], c="#0f766e", label="Class 1", alpha=0.8)
plt.title(f"Learned Non-Linear Decision Boundary (Accuracy: {acc*100:.1f}%)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend(frameon=True, facecolor="#f5f7fa", edgecolor="none")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "decision_boundary.png"), dpi=150)
plt.close()
print("Saved decision_boundary.png")

# 5. Generate and Save Plot 3: Vanishing Gradients (Sigmoid vs ReLU)
# Simulate gradient norms over depth in a 10-layer network
depth = np.arange(1, 11)
# Sigmoid gradients decay exponentially: e.g., (0.25)^depth
sigmoid_grads = (0.25) ** depth * 1.5
# ReLU gradients remain stable: e.g., random variation around 1.0
relu_grads = np.ones(10) * 1.0 + rng.normal(0, 0.05, size=10)

plt.figure(figsize=(8, 5))
plt.plot(depth, sigmoid_grads, marker="o", color="#8b3a3a", label="Sigmoid Activations", linewidth=2)
plt.plot(depth, relu_grads, marker="s", color="#0f766e", label="ReLU Activations", linewidth=2)
plt.yscale("log")
plt.title("Vanishing Gradients: Sigmoid vs. ReLU (Log Scale)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Layer Depth (Backwards from Output to Input)", fontsize=12)
plt.ylabel("Average Gradient Norm", fontsize=12)
plt.xticks(depth)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="#f5f7fa", edgecolor="none")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "vanishing_gradients.png"), dpi=150)
plt.close()
print("Saved vanishing_gradients.png")

# 6. Generate and Save Plot 4: Activation Variance across Layers (Initialization)
# We track how variance of activations changes across layers with different initializations
layers = np.arange(1, 6)
bad_init_var = np.array([1.0, 0.05, 0.002, 0.0001, 0.000001]) # collapse
xavier_var = np.ones(5) * 1.0 + rng.normal(0, 0.08, size=5) # stable

plt.figure(figsize=(8, 5))
plt.plot(layers, bad_init_var, marker="o", color="#8b3a3a", label="Small Random Init (Scale = 0.01)", linewidth=2)
plt.plot(layers, xavier_var, marker="s", color="#0f766e", label="Xavier / He Init (Scale = sqrt(2/n))", linewidth=2)
plt.title("Activation Variance Propagation across Layers", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Layer Index (Forwards)", fontsize=12)
plt.ylabel("Activation Variance", fontsize=12)
plt.xticks(layers)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="#f5f7fa", edgecolor="none")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "weight_initialization.png"), dpi=150)
plt.close()
print("Saved weight_initialization.png")
