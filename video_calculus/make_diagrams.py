"""Generate the four supporting diagrams used in the calculus video."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "diagrams"

NAVY = "#1B264F"
TEAL = "#0D9488"
AMBER = "#B86A00"
GRAY = "#666666"

plt.rcParams.update({
    "font.size": 15,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": NAVY,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Slope of a straight line ------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(0, 5, 10)
y = 1 + 1.5 * x
ax.plot(x, y, color=NAVY, linewidth=3)
x0, x1 = 1.0, 3.0
y0, y1 = 1 + 1.5 * x0, 1 + 1.5 * x1
ax.plot([x0, x1], [y0, y0], color=AMBER, linewidth=2, linestyle="--")
ax.plot([x1, x1], [y0, y1], color=AMBER, linewidth=2, linestyle="--")
ax.scatter([x0, x1], [y0, y1], color=TEAL, s=80, zorder=5)
ax.annotate("run = 2", ((x0 + x1) / 2, y0 - 0.5), color=AMBER, ha="center", fontsize=14)
ax.annotate("rise = 3", (x1 + 0.15, (y0 + y1) / 2), color=AMBER, fontsize=14)
ax.set_title("Slope = rise / run = 3 / 2 = 1.5", color=NAVY, fontsize=16)
ax.set_xlim(0, 5)
ax.set_ylim(0, 9)
save(fig, "slope_line")

# 2. Secant line approaching the tangent line --------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(-0.2, 3.2, 200)
f = lambda t: t ** 2
ax.plot(x, f(x), color=NAVY, linewidth=3, label="f(x) = x^2")
x0 = 1.0
colors = [AMBER, "#D98C00", TEAL]
for h, c in zip([1.4, 0.7, 0.05], colors):
    x1 = x0 + h
    slope = (f(x1) - f(x0)) / h
    xs = np.array([x0 - 0.4, x1 + 0.4])
    ys = f(x0) + slope * (xs - x0)
    ax.plot(xs, ys, color=c, linewidth=2, linestyle="--",
            label=f"secant, h={h}  (slope={slope:.2f})")
    ax.scatter([x0, x1], [f(x0), f(x1)], color=c, s=40, zorder=5)
ax.scatter([x0], [f(x0)], color=NAVY, s=90, zorder=6)
ax.set_title("As h shrinks, the secant line becomes the tangent line", color=NAVY, fontsize=15)
ax.legend(fontsize=10, loc="upper left")
ax.set_xlim(-0.2, 3.2)
ax.set_ylim(-0.5, 9)
save(fig, "secant_to_tangent")

# 3. A curve with its slope changing at different points ---------------------
fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(-2.2, 2.2, 200)
f = lambda t: t ** 2
ax.plot(x, f(x), color=NAVY, linewidth=3, label="f(x) = x^2")
for x0, c in zip([-1.5, 0.0, 1.2], [AMBER, TEAL, "#7C3AED"]):
    slope = 2 * x0
    xs = np.array([x0 - 0.8, x0 + 0.8])
    ys = f(x0) + slope * (xs - x0)
    ax.plot(xs, ys, color=c, linewidth=2.5)
    ax.scatter([x0], [f(x0)], color=c, s=70, zorder=5)
    ax.annotate(f"slope = {slope:.0f}", (x0, f(x0) + 0.9), color=c, ha="center", fontsize=12)
ax.set_title("The derivative gives a DIFFERENT slope at every point", color=NAVY, fontsize=15)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1, 6.5)
save(fig, "varying_slope")

# 4. Gradient descent rolling down a bowl ------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(-3, 3, 200)
f = lambda t: t ** 2 + 1
ax.plot(x, f(x), color=NAVY, linewidth=3)
start = -2.6
lr = 0.35
pts = [start]
for _ in range(6):
    grad = 2 * pts[-1]
    pts.append(pts[-1] - lr * grad)
pts = np.array(pts)
ax.plot(pts, f(pts), "o--", color=AMBER, markersize=9, linewidth=2, zorder=5, label="gradient descent steps")
for i, px in enumerate(pts):
    ax.annotate(str(i), (px, f(px) + 0.4), color=AMBER, ha="center", fontsize=11)
ax.scatter([0], [f(0)], color=TEAL, s=120, zorder=6, marker="*", label="minimum")
ax.set_title("Gradient descent: step opposite the gradient, again and again", color=NAVY, fontsize=14)
ax.legend(fontsize=11, loc="upper center")
ax.set_xlim(-3, 3)
ax.set_ylim(0, 10)
save(fig, "gradient_descent")

print("Diagrams saved.")
