import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import os

# ============================================================
# === GLOBAL STYLE (MATCH NEAT SCRIPT) ========================
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
})

# ============================================================
# === AIRFOIL FUNCTIONS ======================================
# ============================================================

def thickness_distribution(x, t):
    return 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

def compute_airfoil(x, yc, yt):
    dyc_dx = np.gradient(yc, x)
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl

# ============================================================
# === BASE AIRFOIL ===========================================
# ============================================================

m, p, t = 0.02, 0.4, 0.12
num_points = 1000
num_ctrl = 10

beta = np.linspace(0, np.pi, num_points)
x = (1 - np.cos(beta)) / 2

yc_base = np.where(
    x < p,
    m / p**2 * (2 * p * x - x**2),
    m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2),
)

yt_base = thickness_distribution(x, t)

# ============================================================
# === CONTROL POINTS =========================================
# ============================================================

n_each_side = num_ctrl // 2
x_ctrl = np.concatenate([
    np.linspace(0, 1/3, n_each_side, endpoint=False),
    np.linspace(2/3, 1, n_each_side)
])

y_ctrl = np.interp(x_ctrl, x, yc_base)
y_ctrl_original = y_ctrl.copy()
offsets = np.zeros_like(y_ctrl)

# ============================================================
# === SMOOTHING ==============================================
# ============================================================

def smooth_camber(x_ctrl, y_ctrl, x_dense):
    spline = make_interp_spline(x_ctrl, y_ctrl, k=1)
    return gaussian_filter1d(spline(x_dense), sigma=25)

def smooth_neighbors(i, strength=0.22, radius=2):
    for j in range(max(0, i - radius), min(len(y_ctrl), i + radius + 1)):
        if j != i:
            dist = abs(j - i)
            w = strength * np.exp(-dist)
            y_ctrl[j] += w * (y_ctrl[i] - y_ctrl[j])

# ============================================================
# === INITIAL COMPUTE ========================================
# ============================================================

yc = smooth_camber(x_ctrl, y_ctrl, x)
xu, yu, xl, yl = compute_airfoil(x, yc, yt_base)

# ============================================================
# === PLOT SETUP =============================================
# ============================================================

plt.ion()
fig, ax = plt.subplots(figsize=(13, 6))

ax.set_title("Baseline NACA 2412 Airfoil with Control Points", fontweight='bold')
ax.set_xlabel("x/c", fontweight='bold')
ax.set_ylabel("y/c", fontweight='bold')

ax.axis("equal")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.15, 0.15)
ax.grid(True)

plt.subplots_adjust(bottom=0.35)

# --- Airfoil ---
(upper_line,) = ax.plot(xu, yu, color="black", lw=1.8)
(lower_line,) = ax.plot(xl, yl, color="black", lw=1.8)

# --- Camber ---
(control_line,) = ax.plot(
    x, yc,
    linestyle="--",
    color="black",
    lw=1.2,
    alpha=0.7
)

# --- Control points ---
(ctrl_pts,) = ax.plot(
    x_ctrl, y_ctrl,
    'o',
    color="red",
    markersize=6
)

# ============================================================
# === UPDATE =================================================
# ============================================================

def update_plot():
    global offsets

    yc = smooth_camber(x_ctrl, y_ctrl, x)
    xu, yu, xl, yl = compute_airfoil(x, yc, yt_base)

    upper_line.set_data(xu, yu)
    lower_line.set_data(xl, yl)
    control_line.set_data(x, yc)
    ctrl_pts.set_data(x_ctrl, y_ctrl)

    offsets = y_ctrl - np.interp(x_ctrl, x, yc_base)

    for i, tb in enumerate(textboxes):
        tb.set_val(f"{offsets[i]:.4f}")

    fig.canvas.draw_idle()

# ============================================================
# === RESET ==================================================
# ============================================================

def reset_points(event):
    global y_ctrl, offsets
    y_ctrl = y_ctrl_original.copy()
    offsets = np.zeros_like(y_ctrl)

    for tb in textboxes:
        tb.set_val("0.0000")

    update_plot()

# ============================================================
# === MOUSE ==================================================
# ============================================================

dragging = None

def on_press(event):
    global dragging
    if event.inaxes != ax:
        return

    d = np.hypot(x_ctrl - event.xdata, y_ctrl - event.ydata)
    i = np.argmin(d)

    if d[i] < 0.02:
        dragging = i

def on_release(event):
    global dragging
    dragging = None

def on_motion(event):
    global dragging
    if dragging is None or event.inaxes != ax:
        return

    y_ctrl[dragging] = event.ydata
    smooth_neighbors(dragging)
    update_plot()

# ============================================================
# === TEXTBOXES ==============================================
# ============================================================

textboxes = []

rows = 2
cols = int(np.ceil(num_ctrl / rows))

box_width = 0.07
h_spacing = 0.025
v_spacing = 0.065
base_y = 0.10

for i in range(num_ctrl):
    row = i // cols
    col = i % cols

    left = 0.1 + col * (box_width + h_spacing)
    bottom = base_y + (rows - 1 - row) * v_spacing

    axbox = plt.axes([left, bottom, box_width, 0.05])
    tb = TextBox(axbox, f"P{i+1}", initial="0.0000")

    textboxes.append(tb)

def submit_factory(i):
    def submit(text):
        try:
            val = float(text)
            y_ctrl[i] = np.interp(x_ctrl[i], x, yc_base) + val
            update_plot()
        except:
            pass
    return submit

for i, tb in enumerate(textboxes):
    tb.on_submit(submit_factory(i))

# ============================================================
# === BUTTONS ================================================
# ============================================================

save_ax = plt.axes([0.80, 0.10, 0.12, 0.05])
reset_ax = plt.axes([0.65, 0.10, 0.12, 0.05])

save_button = Button(save_ax, "Save")
reset_button = Button(reset_ax, "Reset")

# ============================================================
# === SAVE ===================================================
# ============================================================

filename_box_ax = plt.axes([0.35, 0.02, 0.30, 0.05])
filename_box = TextBox(filename_box_ax, "Filename:")
filename_box_ax.set_visible(False)

def save_points(event):
    filename_box_ax.set_visible(True)
    filename_box.set_val("")
    plt.draw()

def save_with_filename(name):
    filename_box_ax.set_visible(False)

    if not name.strip():
        return

    base_name = name.strip()
    os.makedirs("Geometry(Not)", exist_ok=True)

    yc = smooth_camber(x_ctrl, y_ctrl, x)
    xu, yu, xl, yl = compute_airfoil(x, yc, yt_base)

    with open(f"Geometry(Not)/{base_name}.dat", "w") as f:
        for xi, yi in zip(np.concatenate([xu[::-1], xl[1:]]),
                          np.concatenate([yu[::-1], yl[1:]])):
            f.write(f"{xi:.6f} {yi:.6f}\n")

    print(f"Saved: {base_name}.dat")

filename_box.on_submit(save_with_filename)
save_button.on_clicked(save_points)
reset_button.on_clicked(reset_points)

# ============================================================
# === CONNECT ================================================
# ============================================================

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)

update_plot()
plt.show(block=True)