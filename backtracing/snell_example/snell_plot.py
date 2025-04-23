import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','grid'])
# Load the saved values
dt_values = np.load("dt_values.npy")
h_distances = np.load("haussdorf_distances.npy")

# Log-log regression (base e)
log_dt = np.log(dt_values)
log_error = np.log(h_distances)
slope, intercept = np.polyfit(log_dt[0:18], log_error[0:18], 1)

print(f"Estimated convergence rate (slope): {slope:.3f}")

# Optional: plot log-log line
plt.figure(figsize=(8,6))
plt.plot(log_dt, log_error, 'x', label='log-log points')
plt.plot(log_dt, slope * log_dt + intercept, '-', label=f'Fit line (slope={slope:.3f})')
plt.xlabel("$\log(\Delta_t)$", fontsize=16)
plt.ylabel("$\log(d_H(S_{\Delta_t},S))$",fontsize=16)
plt.title("Log-Log Convergence Plot of Haussdorff Distance ($d_H(S_{\Delta_t},S)$)", fontsize=16, pad = 15)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig('../../plots/loglog_plot_convergence_path.png', dpi = 600)
plt.show()


# SECOND PLOT SHOWING LARGER WINDOW FOR VALUES OD DELTA_T

# Load the saved values
dt_values_larger_window = np.load("dt_values_larger_window.npy")
h_distances_larger_window = np.load("haussdorf_distances_larger_window.npy")

# Log-log regression (base e)
log_dt_larger_window = np.log(dt_values_larger_window)
log_error_larger_window = np.log(h_distances_larger_window)
slope, intercept = np.polyfit(log_dt_larger_window[0:15], log_error_larger_window[0:15], 1)

print(f"Estimated convergence rate (slope): {slope:.3f}")

# Optional: plot log-log line
plt.figure(figsize=(8,6))
plt.plot(log_dt_larger_window, log_error_larger_window, 'x', label='log-log points')
plt.plot(log_dt_larger_window, slope * log_dt_larger_window + intercept, '-', label=f'Fit line (slope={slope:.3f})')
plt.xlabel("$\log(\Delta_t)$", fontsize=16)
plt.ylabel("$\log(d_H(S_{\Delta_t},S))$",fontsize=16)
plt.title("Log-Log Convergence Plot of Haussdorff Distance ($d_H(S_{\Delta_t},S)$)", fontsize=16, pad = 15)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig('../../plots/loglog_plot_convergence_path_larger_window.png', dpi = 600)
plt.show()