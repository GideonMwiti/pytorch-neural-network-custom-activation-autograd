"""
=============================================================================
Project 05: Deep Neural Network with Custom Autograd Activation in PyTorch
=============================================================================
Author     : Gideon Mwiti
Challenge  : 100 PyTorch Projects in 10 Days
Topic      : Custom torch.autograd.Function, Non-Linear Activations, Gradcheck
Goal       : Build, verify, and benchmark a custom activation function inside
             a feedforward neural network on non-linear decision boundaries.
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. CUSTOM ACTIVATION FUNCTION IMPLEMENTATION (AUTOGRAD FUNCTION)
# =============================================================================

class SwishAutogradFunction(torch.autograd.Function):
    """
    Custom Implementation of the Swish / SiLU Activation Function:
      f(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Analytical First Derivative:
      s(x) = sigmoid(x)
      f'(x) = s(x) + x * s(x) * (1 - s(x)) = s(x) + f(x) * (1 - s(x))
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        sigmoid_x = torch.sigmoid(x)
        output = x * sigmoid_x
        # Cache tensors required for the analytical backward gradient pass
        ctx.save_for_backward(x, sigmoid_x)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, sigmoid_x = ctx.saved_tensors
        # Compute exact analytical derivative: dL/dx = grad_output * f'(x)
        grad_x = grad_output * (sigmoid_x + (x * sigmoid_x * (1.0 - sigmoid_x)))
        return grad_x

class CustomSwish(nn.Module):
    """PyTorch Module wrapper for the custom Swish autograd function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SwishAutogradFunction.apply(x)

def verify_custom_gradient():
    """
    Mathematically verifies the custom analytical backward pass against
    finite-difference numerical approximations using torch.autograd.gradcheck.
    """
    test_input = torch.randn(6, 6, dtype=torch.float64, requires_grad=True)
    is_valid = torch.autograd.gradcheck(SwishAutogradFunction.apply, test_input, eps=1e-6, atol=1e-4)
    print(f"[MATHEMATICAL VERIFICATION] torch.autograd.gradcheck passed: {is_valid} (Gradient is exact!)")
    return is_valid

# =============================================================================
# 2. SYNTHETIC NON-LINEAR DATASET (CONCENTRIC CIRCLES)
# =============================================================================

def generate_concentric_circles_data(num_samples: int = 400, noise: float = 0.08):
    """
    Generates non-linearly separable concentric circles:
    - Inner Circle (Class 0): r ~ 0.5
    - Outer Circle (Class 1): r ~ 1.5
    Linear classifiers completely fail on this problem, proving activation power.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n_half = num_samples // 2

    # Class 0: Inner circle
    r_inner = 0.5 + torch.randn(n_half) * noise
    theta_inner = torch.rand(n_half) * 2 * np.pi
    x0 = r_inner * torch.cos(theta_inner)
    y0 = r_inner * torch.sin(theta_inner)
    class_0 = torch.stack([x0, y0], dim=1)
    labels_0 = torch.zeros(n_half, 1)

    # Class 1: Outer circle
    r_outer = 1.4 + torch.randn(n_half) * noise
    theta_outer = torch.rand(n_half) * 2 * np.pi
    x1 = r_outer * torch.cos(theta_outer)
    y1 = r_outer * torch.sin(theta_outer)
    class_1 = torch.stack([x1, y1], dim=1)
    labels_1 = torch.ones(n_half, 1)

    x = torch.cat([class_0, class_1], dim=0)
    y = torch.cat([labels_0, labels_1], dim=0)

    # Shuffle dataset
    perm = torch.randperm(num_samples)
    x = x[perm]
    y = y[perm]

    # Split 80% train, 20% test
    split = int(0.8 * num_samples)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    return x_train, y_train, x_test, y_test

# =============================================================================
# 3. NEURAL NETWORK ARCHITECTURE WITH PLUGGABLE ACTIVATION
# =============================================================================

class BenchmarkMLP(nn.Module):
    """
    3-Layer Feedforward Neural Network supporting interchangeable activations:
    Input (2D) -> Linear(2, 32) -> Act -> Linear(32, 16) -> Act -> Linear(16, 1)
    """
    def __init__(self, activation_fn: nn.Module):
        super(BenchmarkMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            activation_fn,
            nn.Linear(32, 16),
            activation_fn,
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =============================================================================
# 4. TRAINING & BENCHMARKING PIPELINE
# =============================================================================

def train_model(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, epochs: int = 150, lr: float = 0.05):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(x_test)
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs >= 0.5).float()
            test_acc = (test_preds == y_test).float().mean().item() * 100.0

        history["train_loss"].append(loss.item())
        history["test_acc"].append(test_acc)

    return history

def plot_comprehensive_report(
    custom_model: nn.Module,
    histories: dict,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    output_path: str
):
    """
    Renders a 4-panel diagnostic and benchmarking visual report:
    1. Mathematical comparison of Custom Swish vs ReLU vs Sigmoid vs Tanh & Derivatives
    2. 2D Non-linear Decision Boundary learned by the Custom Activation Network
    3. Training BCE Loss convergence comparison
    4. Test Accuracy progression curves (%)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Project 05: PyTorch Deep Neural Network with Custom Autograd Activation (Swish / SiLU)\n"
        f"Custom Activation Accuracy: {histories['Custom Swish']['test_acc'][-1]:.2f}% vs ReLU: {histories['Standard ReLU']['test_acc'][-1]:.2f}% vs Sigmoid: {histories['Standard Sigmoid']['test_acc'][-1]:.2f}%",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    # 1. Activation Function & Derivative Curves
    ax1 = axes[0, 0]
    z_vals = np.linspace(-4, 4, 300)
    sig_z = 1.0 / (1.0 + np.exp(-z_vals))
    swish_z = z_vals * sig_z
    d_swish_z = sig_z + swish_z * (1.0 - sig_z)
    relu_z = np.maximum(0, z_vals)
    d_relu_z = (z_vals > 0).astype(float)

    ax1.plot(z_vals, swish_z, color="#8B5CF6", linewidth=2.5, label="Custom Swish: f(x) = x·σ(x)")
    ax1.plot(z_vals, d_swish_z, color="#8B5CF6", linestyle="--", linewidth=1.8, label="Custom Swish Derivative: f'(x)")
    ax1.plot(z_vals, relu_z, color="#10B981", linestyle=":", linewidth=2.0, label="Standard ReLU: max(0, x)", alpha=0.8)
    ax1.plot(z_vals, d_relu_z, color="#10B981", linestyle="-.", linewidth=1.5, label="ReLU Derivative", alpha=0.6)
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax1.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_title("1. Custom Activation Function & Analytical Derivative Profile", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Input Activation (x)")
    ax1.set_ylabel("Output f(x) / Gradient f'(x)")
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. Decision Boundary on Concentric Circles
    ax2 = axes[0, 1]
    x_min, x_max = -2.2, 2.2
    y_min, y_max = -2.2, 2.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        custom_model.eval()
        probs = torch.sigmoid(custom_model(grid_tensor)).reshape(xx.shape).numpy()

    contour = ax2.contourf(xx, yy, probs, levels=25, cmap="Spectral", alpha=0.7)
    cbar = plt.colorbar(contour, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("P(Class = 1 | x)", fontsize=10)
    ax2.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")

    x_np = x_test.numpy()
    y_np = y_test.numpy().squeeze()
    ax2.scatter(x_np[y_np == 0, 0], x_np[y_np == 0, 1], color="#DC2626", edgecolors="black", linewidth=0.5, label="Inner (Class 0)", alpha=0.9)
    ax2.scatter(x_np[y_np == 1, 0], x_np[y_np == 1, 1], color="#2563EB", edgecolors="black", linewidth=0.5, label="Outer (Class 1)", alpha=0.9)
    ax2.set_title("2. Learned Non-Linear Decision Boundary (Custom Activation)", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Feature 1 ($x_1$)")
    ax2.set_ylabel("Feature 2 ($x_2$)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.4)

    # 3. Training Loss Convergence Comparison
    ax3 = axes[1, 0]
    colors = {"Custom Swish": "#8B5CF6", "Standard ReLU": "#10B981", "Standard Sigmoid": "#EF4444"}
    epochs_range = range(1, len(histories["Custom Swish"]["train_loss"]) + 1)
    for name, hist in histories.items():
        ax3.plot(epochs_range, hist["train_loss"], label=name, color=colors[name], linewidth=2.2)

    ax3.set_title("3. Training Loss Convergence Benchmark (BCE Loss)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_yscale("log")
    ax3.legend(frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Test Accuracy Progression Benchmark
    ax4 = axes[1, 1]
    for name, hist in histories.items():
        ax4.plot(epochs_range, hist["test_acc"], label=f"{name} (Final: {hist['test_acc'][-1]:.1f}%)", color=colors[name], linewidth=2.2)

    ax4.set_title("4. Test Accuracy Progression Benchmark (%)", fontweight="bold", fontsize=11)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.set_ylim([45, 105])
    ax4.legend(loc="lower right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visual report saved successfully at: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "custom_activation_results.png")

    EPOCHS = 150
    LR = 0.05

    print("=================================================================")
    print("   PROJECT 05: NEURAL NETWORK WITH CUSTOM AUTOGRAD ACTIVATION    ")
    print("=================================================================")
    
    # Step 1: Verify custom backward pass using autograd gradcheck
    verify_custom_gradient()

    # Step 2: Generate Non-Linear Concentric Circles Dataset
    x_train, y_train, x_test, y_test = generate_concentric_circles_data(num_samples=400)
    print(f"Dataset Size       : {x_train.size(0)} Train samples, {x_test.size(0)} Test samples")
    print(f"Hyperparameters    : Epochs = {EPOCHS}, Learning Rate = {LR}")
    print("-----------------------------------------------------------------")

    # Step 3: Instantiate and Benchmark 3 Architectures
    models = {
        "Custom Swish": BenchmarkMLP(activation_fn=CustomSwish()),
        "Standard ReLU": BenchmarkMLP(activation_fn=nn.ReLU()),
        "Standard Sigmoid": BenchmarkMLP(activation_fn=nn.Sigmoid())
    }

    histories = {}
    print(f"{'Model / Activation':^20} | {'Final Loss':^12} | {'Test Accuracy (%)':^18}")
    print("-----------------------------------------------------------------")

    for name, model in models.items():
        torch.manual_seed(42)
        hist = train_model(model, x_train, y_train, x_test, y_test, epochs=EPOCHS, lr=LR)
        histories[name] = hist
        print(f"{name:^20} | {hist['train_loss'][-1]:^12.5f} | {hist['test_acc'][-1]:^18.2f}%")

    print("-----------------------------------------------------------------")
    print("Benchmarking Complete!")
    print("=================================================================\n")

    # Step 4: Render Visual Artifact
    plot_comprehensive_report(
        custom_model=models["Custom Swish"],
        histories=histories,
        x_test=x_test,
        y_test=y_test,
        output_path=output_plot_path
    )

if __name__ == "__main__":
    main()
