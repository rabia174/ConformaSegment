import ast
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ruptures as rpt

# import visualization libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# import tensorflow libraries
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# import keras libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# VISUALISATION METHOD
def plot_intervals_with_changes(x_test, intervals, significance_values, base_prediction, lower_bound, upper_bound, change_points=None):
    # Create a color map
    norm = mcolors.Normalize(vmin=min(significance_values), vmax=max(significance_values))
    cmap = plt.cm.viridis

    # Plot the signal (x_test[0])
    plt.figure(figsize=(20, 6))  # Adjusted size for better visibility
    plt.plot(x_test.flatten(), label="Signal", color='black', linewidth=2)

    # Add vertical lines and colorize the intervals based on significance values
    for i, (start, end) in enumerate(intervals):
        color = cmap(norm(significance_values[i]))
        plt.axvline(x=start, color=color, linestyle='--', linewidth=2)
        plt.axvline(x=end, color=color, linestyle='--', linewidth=2)
        plt.fill_between(np.arange(start, end), np.min(x_test), np.max(x_test), color=color, alpha=0.2)

    # Plot change points if provided
    if change_points:
        for cp in change_points:
            plt.axvline(x=cp, color='red', linestyle='-', linewidth=2, label='Change Point' if 'Change Point' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Add a colorbar to show the significance scale (horizontal position)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', fraction=0.05, pad=0.2, shrink=1)
    cbar.set_label("Significance Value", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Add a rectangle for the base prediction interval
    rect_x = len(x_test)  # Rectangle positioned at the end of the signal
    rect_width = 5  # Arbitrary width for visualization
    rect_y = lower_bound  # Lower bound of the prediction interval
    rect_height = upper_bound - lower_bound  # Height of the interval

    rect_y = float(rect_y[0][0])  # Extract scalar value from the array
    rect_height = float(rect_height[0][0])  # Similarly, extract the scalar value

    rect = patches.Rectangle(
        (rect_x, rect_y), rect_width, rect_height,
        color='orange', alpha=0.3, label="Prediction Interval"
    )
    plt.gca().add_patch(rect)

    # Add text annotations for base prediction, upper and lower bounds
    plt.text(rect_x - 5 + rect_width, base_prediction, f"Base Prediction: {base_prediction:.2f}",
             color='black', fontsize=14, va='center', bbox=dict(facecolor='white', edgecolor='purple', alpha=0.7))

    # Add text annotation at the top of the orange rectangle (upper bound)
    upper_bound_scalar = float(upper_bound[0][0])  # Extract scalar value
    plt.text(rect_x - 5 + rect_width, upper_bound_scalar,
             f"Upper Bound: {upper_bound_scalar:.2f}",
             color='black', fontsize=14, va='center',
             bbox=dict(facecolor='white', edgecolor='darkorange', alpha=0.7))

    # Add text annotation at the bottom of the orange rectangle (lower bound)
    lower_bound_scalar = float(lower_bound[0][0])  # Extract scalar value
    plt.text(rect_x - 5 + rect_width, lower_bound_scalar,
             f"Lower Bound: {lower_bound_scalar:.2f}",
             color='black', fontsize=14, va='center',
             bbox=dict(facecolor='white', edgecolor='darkorange', alpha=0.7))

    # Labeling and showing the plot
    plt.title('95% Guarantee Prediction with Intervals and Change Points', fontsize=18)
    plt.xlabel('Time (or Index)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# VISUALISATION FUNCTION 2
def plot_intervals(x_test, intervals, significance_values, base_prediction, lower_bound, upper_bound):
    # Create a color map
    norm = mcolors.Normalize(vmin=min(significance_values), vmax=max(significance_values))
    cmap = plt.cm.viridis
    
    # Plot the signal
    plt.figure(figsize=(20, 5))
    plt.plot(x_test.flatten(), label="Signal", color='black')
    
    # Add the regressor prediction and the confidence intervals
    plt.plot(x_test, regressor_prediction, label="Regressor Prediction", color='blue', linewidth=2)
    # Add vertical lines and colorize the intervals based on significance values
    change_points = []  # To store change points for x-axis ticks
    for i, (start, end) in enumerate(intervals):
        color = cmap(norm(significance_values[i]))
        plt.axvline(x=start, color=color, linestyle='--', linewidth=2)
        plt.axvline(x=end, color=color, linestyle='--', linewidth=2)
        plt.fill_between(np.arange(start, end), np.min(x_test), np.max(x_test), color=color, alpha=0.1)
        
        # Add significance value as text near each interval
        mid_point = (start + end) / 2
        plt.text(mid_point, np.min(x_test) + 0.01 * (np.max(x_test) - np.min(x_test)),
                 f"{significance_values[i]:.2f}",
                 color='black', fontsize=12, ha='center',
                 bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
        
        # Collect start points for x-axis ticks
        if start not in change_points:
            change_points.append(start)
        if end not in change_points:
            change_points.append(end)
    
    # Set x-axis ticks only for the change points
    plt.xticks(change_points, labels=[f"{cp}" for cp in change_points], fontsize=14)
    
    # Add a colorbar to show the significance scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', fraction=0.05, pad=0.2, shrink=1)
    cbar.set_label("Significance Value", fontsize=14)
    
    # Add a rectangle for the base prediction interval
    rect_x = len(x_test)  # Rectangle positioned at the end of the signal
    rect_width = 5  # Arbitrary width for visualization
    rect_y = float(lower_bound[0][0])  # Extract scalar value from the array
    rect_height = float(upper_bound[0][0]) - rect_y  # Height of the interval
    
    rect = patches.Rectangle(
        (rect_x, rect_y), rect_width, rect_height,
        color='orange', alpha=0.3, label="Prediction Interval"
    )
    plt.gca().add_patch(rect)
    
    # Add text annotations for the base prediction, upper bound, and lower bound
    plt.text(rect_x-3 + rect_width, base_prediction, f"Base Prediction: {base_prediction:.2f}",
             color='black', fontsize=15, va='center',
             bbox=dict(facecolor='white', edgecolor='purple', alpha=0.7))
    
    upper_bound_scalar = float(upper_bound[0][0])  # Extract scalar value
    plt.text(rect_x-3 + rect_width, upper_bound_scalar,
             f"Upper Bound: {upper_bound_scalar:.2f}",
             color='black', fontsize=15, va='center',
             bbox=dict(facecolor='white', edgecolor='darkorange', alpha=0.7))
    
    lower_bound_scalar = float(lower_bound[0][0])  # Extract scalar value
    plt.text(rect_x-3 + rect_width, lower_bound_scalar,
             f"Lower Bound: {lower_bound_scalar:.2f}",
             color='black', fontsize=15, va='center',
             bbox=dict(facecolor='white', edgecolor='darkorange', alpha=0.7))
    
    # Labeling and showing the plot
    plt.title('95% Guarantee Prediction with Regressor and Confidence Interval', fontsize=18)
    plt.xlabel('Change Points (Index)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.xticks(fontsize=16)                     # Larger font for x-axis tick labels
    plt.yticks(fontsize=16)                     # Larger font for y-axis tick labels
    plt.legend(loc='upper left', fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()

# VISUAL FUNCTION 3
def plot_intervals(x_test, intervals, significance_values, base_prediction, lower_bound, upper_bound):
    # Create a color map
    norm = mcolors.Normalize(vmin=min(significance_values), vmax=max(significance_values))
    cmap = plt.cm.viridis
    
    # Plot the signal
    plt.figure(figsize=(20, 5))
    plt.plot(x_test.flatten(), label="Signal", color='black')
    
    # Add vertical lines and colorize the intervals based on significance values
    change_points = []  # To store change points for x-axis ticks
    for i, (start, end) in enumerate(intervals):
        color = cmap(norm(significance_values[i]))
        plt.axvline(x=start, color=color, linestyle='--', linewidth=2)
        plt.axvline(x=end, color=color, linestyle='--', linewidth=2)
        plt.fill_between(np.arange(start, end), np.min(x_test), np.max(x_test), color=color, alpha=0.1)
        
        # Add significance value as text near each interval
        mid_point = (start + end) / 2
        plt.text(mid_point, np.min(x_test) + 0.01 * (np.max(x_test) - np.min(x_test)),
                 f"{significance_values[i]:.2f}",
                 color='black', fontsize=12, ha='center',
                 bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
        
        # Collect start points for x-axis ticks
        if start not in change_points:
            change_points.append(start)
        if end not in change_points:
            change_points.append(end)
    
    # Set x-axis ticks only for the change points
    plt.xticks(change_points, labels=[f"{cp}" for cp in change_points], fontsize=14)
    
    # Add a colorbar to show the significance scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', fraction=0.05, pad=0.2, shrink=1)
    cbar.set_label("Significance Value", fontsize=14)
    
    # Add a rectangle for the base prediction interval
    rect_x = len(x_test)  # Rectangle positioned at the end of the signal
    rect_width = 5  # Arbitrary width for visualization
    rect_y = float(lower_bound[0][0])  # Extract scalar value from the array
    rect_height = float(upper_bound[0][0]) - rect_y  # Height of the interval
    
    rect = patches.Rectangle(
        (rect_x, rect_y), rect_width, rect_height,
        color='orange', alpha=0.3, label="Prediction Interval"
    )
    plt.gca().add_patch(rect)
    
    # Add text annotations for the base prediction, upper bound, and lower bound
    plt.text(rect_x-3 + rect_width, base_prediction, f"Base Prediction: {base_prediction:.2f}",
             color='black', fontsize=15, va='center',
             bbox=dict(facecolor='white', edgecolor='purple', alpha=0.7))
    
    upper_bound_scalar = float(upper_bound[0][0])  # Extract scalar value
    plt.text(rect_x-3 + rect_width, upper_bound_scalar,
             f"Upper Bound: {upper_bound_scalar:.2f}",
             color='black', fontsize=15, va='center',
             bbox=dict(facecolor='white', edgecolor='darkorange', alpha=0.7))
    
    lower_bound_scalar = float(lower_bound[0][0])  # Extract scalar value
    plt.text(rect_x-3 + rect_width, lower_bound_scalar,
             f"Lower Bound: {lower_bound_scalar:.2f}",
             color='black', fontsize=15, va='center',
             bbox=dict(facecolor='white', edgecolor='darkorange', alpha=0.7))
    
    # Labeling and showing the plot
    plt.title('95% Guarantee Prediction', fontsize=18)
    plt.xlabel('Change Points (Index)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.xticks(fontsize=16)                     # Larger font for x-axis tick labels
    plt.yticks(fontsize=16)                     # Larger font for y-axis tick labels
    plt.legend(loc='upper left', fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()
    
# NOISE GENERATION FUNCTION
def omit_single_interval_with_uniform_noise(signal, omit_interval, noise_range=(-0.5, 0.5)):
    """
    Omit a single interval from a time series signal by replacing it with uniform noise.
    
    Args:
    - signal (array-like): The input time series data (e.g., a row of x_train).
    - omit_interval (tuple): A tuple (start, end) specifying the interval to omit.
    - noise_range (tuple): The range for the uniform noise (min, max).
    
    Returns:
    - modified_signal (array-like): The time series with the specified interval replaced by noise.
    """
    # Make sure signal is 1D for ease of manipulation
    signal = signal.flatten()  # Flatten if it's 2D (e.g., shape (n_samples, n_features))
    
    start, end = ast.literal_eval(str(omit_interval))
    modified_signal = signal.copy()  # Make a copy to avoid modifying the original signal
    
    # Generate random noise sampled from a uniform distribution within the specified range
    noise = np.random.uniform(noise_range[0], noise_range[1], end - start + 1)
    try:
        # Add noise to the specified interval
        modified_signal[start:end + 1] = noise
    except:
        noise = noise[:-1]
        modified_signal[start:end + 1] = noise
    return modified_signal

# GET BOUNDS FOR PERTURBED INTERVALS METHOD
def collect_interval_base_changes(regressor, intervals, signal, y_test, X_cal, y_cal, alpha):
    coverage_ls = []
    average_interval_size_ls = []
    lower_bound_ls = []
    upper_bound_ls = []
    
    for interval in intervals:
        # Example: A single interval to omit (this can come from your change point detection results)
        omit_interval = interval  # Example interval
        print(interval)
        # Omit the interval by replacing it with uniform noise
        modified_signal = omit_single_interval_with_uniform_noise(signal, omit_interval, noise_range=(-0.5, 0.5))
        modified_signal = modified_signal.reshape(1, modified_signal.shape[0], 1)
        print(modified_signal.shape)
        # perform conformal prediction on modified signal
        coverage, average_interval_size, pred, lower_bound, upper_bound = produce_interval_for_single_signal(regressor, alpha, X_cal, y_cal, modified_signal, y_test)
        coverage_ls.append(coverage)
        average_interval_size_ls.append(average_interval_size)
        print(lower_bound, upper_bound)
        lower_bound_ls.append(lower_bound)
        upper_bound_ls.append(upper_bound)
    return coverage_ls, average_interval_size_ls, pred, upper_bound_ls, lower_bound_ls

# DETECT CHANGE POINTS METHOD
def detect_and_visualize_change_points(signal, penalty=5):
    """
    Detects change points in a time series using PELT, converts them into intervals, and visualizes the result.

    Args:
    - signal (array-like): Input time series data.
    - penalty (int, optional): Penalty to control the number of change points. Default is 5.

    Returns:
    - intervals (list of tuples): List of intervals represented as (start, end).
    """
    # Detect change points using PELT
    model = rpt.Pelt(model="l2").fit(signal)
    change_points = model.predict(pen=penalty)
    
     # Convert change points into intervals
    intervals = [(0, change_points[0])]  # Add the interval from 0 to the first change point
    intervals += [(change_points[i], change_points[i + 1]) for i in range(len(change_points) - 1)]

    # Convert change points into intervals
    # intervals = [(change_points[i], change_points[i + 1]) for i in range(len(change_points) - 1)]

    """     # Visualization
        plt.figure(figsize=(20, 4))
        plt.plot(signal, label="Time Series", color="blue")
        for cp in change_points:
            plt.axvline(cp, color="red", linestyle="--", label="Change Point" if cp == change_points[0] else None)
        plt.title("Change Point Detection using Ruptures")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
        
        print(f"Detected change points: {change_points}")
        print(f"Intervals: {intervals}")"""

    return intervals
    
# COMPUTE FEATURE IMPORTANCE METHOD
def get_feature_importance(regressor, sample, y_test, X_cal, y_cal, alpha, pen):
    """
    Computes feature importance intervals using change point detection and conformal prediction.

    Returns:
    - significance_values (list): Significance values for each interval.
    """
    # Step 1: Compute base intervals
    base_coverage, base_avg_interval_size, pred, base_lower_bound, base_upper_bound = produce_interval_for_single_signal(
        regressor, alpha, X_cal, y_cal, sample.reshape(1, sample.shape[0], 1), y_test
    )
    print(pred)
    # Step 2: Detect change points and intervals
    signal = sample.reshape(-1, 1)  # Reshape signal for processing
    intervals = detect_and_visualize_change_points(signal, penalty=pen)

    # Step 3: Collect interval-based changes
    coverage_ls, avg_interval_size_ls, pred_updated, upper_bound_ls, lower_bound_ls = collect_interval_base_changes(regressor, intervals, signal,y_test, X_cal, y_cal, alpha)

    # Step 4: Compute significance values
    significance_values =get_significance_values(base_upper_bound, base_lower_bound, coverage_ls, upper_bound_ls, lower_bound_ls)
    print(significance_values)
    # Step 5: Visualize intervals with significance values
    plot_intervals(signal, intervals, significance_values, pred.item(), base_lower_bound, base_upper_bound)

# FEATURE IMPORTANCE W/O PLOT
def get_feature_importance_wo_plot(sample, alpha):
    """
    Computes feature importance intervals using change point detection and conformal prediction.

    Returns:
    - significance_values (list): Significance values for each interval.
    """
    # Step 1: Compute base intervals
    base_coverage, base_avg_interval_size, pred, base_lower_bound, base_upper_bound = produce_interval_for_single_signal(
        regressor, alpha, X_cal, y_cal, sample.reshape(1, sample.shape[0], 1), y_test
    )
    print(pred)
    # Step 2: Detect change points and intervals
    signal = sample.reshape(-1, 1)  # Reshape signal for processing
    intervals = detect_and_visualize_change_points(signal, penalty=2)

    # Step 3: Collect interval-based changes
    coverage_ls, avg_interval_size_ls, pred_updated, upper_bound_ls, lower_bound_ls = collect_interval_base_changes(intervals, signal, y_test, alpha)

    # Step 4: Compute significance values
    significance_values =get_significance_values(base_upper_bound, base_lower_bound, coverage_ls, upper_bound_ls, lower_bound_ls)
    return significance_values, intervals
    
# GET SINGLE CONFORMAL FORECAST METHOD
def produce_interval_for_single_signal(regressor, alpha, X_cal, y_cal, X_test, y_test):
    y_pred_single = regressor.predict(X_test)
    # Get predictions and residuals for calibration set
    y_cal_pred = regressor.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred.flatten())
    
    quantile = np.quantile(residuals, 1 - alpha)  # Residual quantile for 95% coverage
    
    # Prediction interval for the single point
    lower_bound = y_pred_single - quantile
    upper_bound = y_pred_single + quantile
    
    # Compute interval size
    interval_sizes =  np.abs(lower_bound - upper_bound)
    average_interval_size = interval_sizes
    
    # Check if true values are within prediction intervals
    in_interval = (y_test >= lower_bound) & (y_test <= upper_bound)
    
    # Calculate coverage
    coverage = np.mean(in_interval)  # Fraction of points within intervals
    #print(f"Coverage: {coverage * 100:.2f}%")
    #print(f"Average Interval Size: {average_interval_size }")
    #print(lower_bound, upper_bound)
    
    return coverage, average_interval_size, y_pred_single, lower_bound, upper_bound

# COMPUTE IMPORTANCE WEIGHTS METHOD
def get_significance_values(base_upper_bound, base_lower_bound, coverage_ls, upper_bound_ls, lower_bound_ls):
    
    changes_in_interval_shift_up = abs(np.array([base_upper_bound]*len(upper_bound_ls))) - abs(np.array(upper_bound_ls))
    changes_in_interval_shift_low = abs(np.array([base_lower_bound]*len(lower_bound_ls))) - abs(np.array(lower_bound_ls))

    significance_values = np.array(abs(changes_in_interval_shift_up.flatten())) + np.array(abs(changes_in_interval_shift_low.flatten())) #+ np.array(abs(changes_in_cov.flatten())) 
    
    return significance_values