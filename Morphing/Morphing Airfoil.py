import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import os

# ============================================================
# === AIRFOIL GEOMETRY AND COMPUTATION FUNCTIONS ============
# ============================================================

def thickness_distribution(x, t):
    """Computes NACA 4-digit half-thickness distribution (yt)."""
    return 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

def compute_airfoil(x, yc, yt):
    """Computes upper and lower surface coordinates of airfoil using provided half-thickness array yt."""
    dyc_dx = np.gradient(yc, x)
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl

# ============================================================
# === BASE AIRFOIL PARAMETERS ================================
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
# === CONTROL POINT SETUP ====================================
# ============================================================

n_each_side = num_ctrl // 2
x_ctrl_left = np.linspace(0, 1/3, n_each_side, endpoint=False)
x_ctrl_right = np.linspace(2/3, 1, n_each_side)
x_ctrl = np.concatenate([x_ctrl_left, x_ctrl_right])

y_ctrl = np.interp(x_ctrl, x, yc_base)
y_ctrl_original = y_ctrl.copy()
offsets = np.zeros_like(y_ctrl)

# ============================================================
# === CAMBER LINE SMOOTHING =================================
# ============================================================

def smooth_camber(x_ctrl, y_ctrl, x_dense):
    spline = make_interp_spline(x_ctrl, y_ctrl, k=3)
    yc_dense = spline(x_dense)
    yc_dense = gaussian_filter1d(yc_dense, sigma=1.2)
    return yc_dense

def smooth_neighbors(i, strength=0.18, radius=2):
    for j in range(max(0, i - radius), min(len(y_ctrl), i + radius + 1)):
        if j != i:
            dist = abs(j - i)
            w = strength * np.exp(-dist)
            y_ctrl[j] += w * (y_ctrl[i] - y_ctrl[j])

# ============================================================
# === INITIAL COMPUTATION ====================================
# ============================================================

yc = smooth_camber(x_ctrl, y_ctrl, x)
xu, yu, xl, yl = compute_airfoil(x, yc, yt_base)

# ============================================================
# === PLOT SETUP =============================================
# ============================================================

plt.ion()
fig, ax = plt.subplots(figsize=(13, 6))  # ⬅️ Larger window
ax.set_title("Interactive Morphing Airfoil (Thickness Preserved)", fontsize=14, weight="bold")
ax.set_xlabel("x (chord)", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.axis("equal")
ax.grid(True)
plt.subplots_adjust(bottom=0.40)  # ⬅️ Extra bottom space for filename box

(upper_line,) = ax.plot(xu, yu, "b-", lw=1.8)
(lower_line,) = ax.plot(xl, yl, "b-", lw=1.8)
(control_line,) = ax.plot(x, yc, "r--", lw=1.2)
(ctrl_pts,) = ax.plot(x_ctrl, y_ctrl, "ro", markersize=6)

dragging = None
textboxes = []

# ============================================================
# === UPDATE FUNCTION ========================================
# ============================================================

def update_plot():
    global offsets
    yc = smooth_camber(x_ctrl, y_ctrl, x)
    xu, yu, xl, yl = compute_airfoil(x, yc, yt_base)
    ctrl_pts.set_data(x_ctrl, y_ctrl)
    control_line.set_data(x, yc)
    upper_line.set_data(xu, yu)
    lower_line.set_data(xl, yl)
    offsets = y_ctrl - np.interp(x_ctrl, x, yc_base)
    for i, tb in enumerate(textboxes):
        tb.set_val(f"{offsets[i]:.4f}")
    fig.canvas.draw_idle()

# ============================================================
# === RESET FUNCTION =========================================
# ============================================================

def reset_points(event):
    global y_ctrl, offsets
    y_ctrl = y_ctrl_original.copy()
    offsets = np.zeros_like(y_ctrl)
    for tb in textboxes:
        tb.set_val("0.0000")
    update_plot()
    print("🔄 Points reset to original positions.")

# ============================================================
# === MOUSE EVENTS ===========================================
# ============================================================

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
    smooth_neighbors(dragging, strength=0.25, radius=2)
    update_plot()

# ============================================================
# === TEXTBOX SETUP ==========================================
# ============================================================

textbox_axes = []
rows = 2
cols = int(np.ceil(num_ctrl / rows))
box_width = 0.08
h_spacing = 0.03
v_spacing = 0.07
base_y = 0.10  # Moved upward slightly to make room for filename box

for i in range(num_ctrl):
    row = i // cols
    col = i % cols
    left = 0.1 + col * (box_width + h_spacing)
    bottom = base_y + (rows - 1 - row) * v_spacing
    axbox = plt.axes([left, bottom, box_width, 0.05])
    tb = TextBox(axbox, f"P{i+1}", initial=f"{offsets[i]:.4f}")
    textbox_axes.append(axbox)
    textboxes.append(tb)

def submit_factory(i):
    def submit(text):
        try:
            val = float(text)
            y_ctrl[i] = np.interp(x_ctrl[i], x, yc_base) + val
            update_plot()
        except ValueError:
            pass
    return submit

for i, tb in enumerate(textboxes):
    tb.on_submit(submit_factory(i))

# ============================================================
# === SAVE & RESET BUTTONS ===================================
# ============================================================

save_ax = plt.axes([0.82, 0.10, 0.12, 0.06])
reset_ax = plt.axes([0.67, 0.10, 0.12, 0.06])
save_button = Button(save_ax, "Save Airfoil", color="#aaffaa", hovercolor="#77ff77")
reset_button = Button(reset_ax, "Reset Points", color="#ffcccc", hovercolor="#ff9999")

# Hidden textbox for filename input BELOW everything
filename_box_ax = plt.axes([0.35, 0.02, 0.30, 0.05])  # ⬅️ Very bottom center
filename_box = TextBox(filename_box_ax, "File name: ", initial="", color="lightgray")
filename_box_ax.set_visible(False)

def save_points(event):
    filename_box_ax.set_visible(True)
    filename_box.set_val("")
    plt.draw()
    print("💾 Enter a filename (without extension) and press Enter to save:")

def save_with_filename(name):
    filename_box_ax.set_visible(False)
    if not name.strip():
        print("⚠️ No filename entered. Save cancelled.")
        plt.draw()
        return

    base_name = name.strip()
    os.makedirs("Geometry", exist_ok=True)
    txt_filename = f"Geometry/{base_name}.txt"
    dat_filename = f"Geometry/{base_name}.dat"

    yc = smooth_camber(x_ctrl, y_ctrl, x)
    xu, yu, xl, yl = compute_airfoil(x, yc, yt_base)

    # Save .txt
    with open(txt_filename, "w") as f:
        f.write("=== Airfoil Parameters ===\n")
        f.write(f"m = {m}\n")
        f.write(f"p = {p}\n")
        f.write(f"t = {t}\n\n")
        f.write("=== Control Points ===\n")
        for xi, yi, off in zip(x_ctrl, y_ctrl, offsets):
            f.write(f"{xi:.5f}, {yi:.5f}, {off:.5f}\n")
        f.write("\n=== Upper Surface ===\n")
        for xi, yi in zip(xu, yu):
            f.write(f"{xi:.5f}, {yi:.5f}\n")
        f.write("\n=== Lower Surface ===\n")
        for xi, yi in zip(xl, yl):
            f.write(f"{xi:.5f}, {yi:.5f}\n")

    print(f"✅ Airfoil saved as {txt_filename}")

    # Save .dat for XFOIL
    N = 100
    beta = np.linspace(0, np.pi, N)
    x_cos = 0.5 * (1 - np.cos(beta))
    y_upper_interp = np.interp(x_cos, xu, yu)
    y_lower_interp = np.interp(x_cos, xl, yl)
    x_all = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_all = np.concatenate([y_upper_interp[::-1], y_lower_interp[1:]])

    with open(dat_filename, "w") as f:
        f.write(f"{base_name}\n")
        for xi, yi in zip(x_all, y_all):
            f.write(f"{xi:.6f} {yi:.6f}\n")

    print(f"✅ XFOIL-compatible file saved as {dat_filename}")
    print(f"📂 Load in XFOIL using:  LOAD {base_name}.dat")
    plt.draw()

filename_box.on_submit(save_with_filename)
save_button.on_clicked(save_points)
reset_button.on_clicked(reset_points)

# ============================================================
# === CONNECT INTERACTIONS ===================================
# ============================================================

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)

update_plot()
plt.show(block=True)