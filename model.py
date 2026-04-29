import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_pipeline
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import pandas as pd
 
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess_pipeline
from utils import drop_minmax
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score



 

# CARE Score Calculation Module with Event F-beta (Reliability)
class CAREScore:

    # acknowdledgement : The care score was co-authored with kashif Riyaz and Nitesh morem


    """
    CAREScore evaluates anomaly detection models using a composite metric
    that emphasizes not just classification performance, but also event-level
    reliability and early warning capabilities.

    Acknowledgement:
    The CARE score was co-authored with Kashif Riyaz and Nitesh Morem.

    Parameters:
    ----------
    beta : float
        Weighting factor for precision vs. recall in F-beta score (default is 0.5, favoring precision).
    early_weight : float
        Weight assigned to the earliness score in the final CARE formula.
    """
     
    def __init__(self, beta=0.5, early_weight=2.0):
        self.beta = 0.5
        self.early_weight = 2.0

    def f_beta(self, tp, fp, fn):
        """
        Compute point-level F-beta score.

        Parameters:
        ----------
        tp : int
            True positives.
        fp : int
            False positives.
        fn : int
            False negatives.

        Returns:
        -------
        float
            F-beta score for point-wise anomaly detection.
        """
        '''this one is for the individual points'''
        beta_sq = self.beta ** 2
        numerator = (1 + beta_sq) * tp
        denominator = (1 + beta_sq) * tp + beta_sq * fn + fp
        return numerator / denominator if denominator != 0 else 0.0

    def accuracy(self, tn, fp):
        """
        Compute accuracy over normal (non-anomalous) data points.

        Parameters:
        ----------
        tn : int
            True negatives.
        fp : int
            False positives.

        Returns:
        -------
        float
            Accuracy over normal class.
        """
        return tn / (tn + fp) if (tn + fp) != 0 else 0.0

    def weighted_earliness(self, predictions, anomaly_start, anomaly_end):

        """
        Compute weighted earliness of detection within a known anomaly window.

        Parameters:
        ----------
        predictions : list[int]
            List of binary predictions (0 or 1).
        anomaly_start : int
            Start index of the anomaly event.
        anomaly_end : int
            End index of the anomaly event.

        Returns:
        -------
        float
            Weighted score prioritizing earlier detections within the anomaly window.
        """
        weights = []
        scores = []
        length = anomaly_end - anomaly_start + 1
        for i in range(length):
            weight = 1 - (i / length)
            weights.append(weight)
            scores.append(predictions[anomaly_start + i])
        weighted_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights) if sum(weights) != 0 else 0.0
        return weighted_score

    def event_f_beta(self, tp_events, fp_events, fn_events):

        """
        Compute F-beta score over entire anomaly events (not points).

        Parameters:
        ----------
        tp_events : int
            Number of correctly predicted anomaly events.
        fp_events : int
            Number of false alarm events.
        fn_events : int
            Number of missed anomaly events.

        Returns:
        -------
        float
            Event-level F-beta score.
        """
        '''This one is for the dataset'''
        beta_sq = self.beta ** 2
        numerator = (1 + beta_sq) * tp_events
        denominator = numerator + beta_sq * fn_events + fp_events
        return numerator / denominator if denominator != 0 else 0.0

    def compute_care(self, coverage_fbeta, accuracy_score, reliability_fbeta, earliness_score):

        """
        Compute the final CARE score.

        Parameters:
        ----------
        coverage_fbeta : float
            Point-level F-beta score.
        accuracy_score : float
            Accuracy over the normal class.
        reliability_fbeta : float
            Event-level F-beta score.
        earliness_score : float
            Weighted early detection score.

        Returns:
        -------
        float
            Composite CARE score.
        """
        if coverage_fbeta == 0:
            return 0.0
        if accuracy_score < 0.5:
            return accuracy_score
        return (coverage_fbeta + accuracy_score + reliability_fbeta + self.early_weight * earliness_score) / (3 + self.early_weight)
    

#####################################################################################################
    




def theoretical_power(wind, v_in=3.0, v_r=12.0, v_out=25.0, P_rated=1.0):


    """
    Compute the theoretical power output of a wind turbine based on wind speed.

    This follows a typical three-region wind turbine power curve:
    - Region I: Below cut-in speed (no power generated).
    - Region II: Between cut-in and rated speed (power increases cubically).
    - Region III: Rated power output.
    - Region IV: Above cut-out speed (no power for safety).

    Parameters:
    ----------
    wind : array-like
        Wind speeds (in m/s).
    v_in : float
        Cut-in wind speed (m/s), below which no power is generated. Default is 3.0.
    v_r : float
        Rated wind speed (m/s), where turbine reaches maximum output. Default is 12.0.
    v_out : float
        Cut-out wind speed (m/s), above which turbine shuts down for safety. Default is 25.0.
    P_rated : float
        Rated power output (normalized or actual in kW/MW). Default is 1.0.

    Returns:
    -------
    np.ndarray
        Power output values corresponding to input wind speeds.
    """
    wind = np.asarray(wind)
    power = np.piecewise(
        wind,
        [wind < v_in,
         (wind >= v_in) & (wind < v_r),
         (wind >= v_r) & (wind < v_out),
         wind >= v_out],
        [0,
         lambda v: P_rated * (v ** 3 - v_in ** 3) / (v_r ** 3 - v_in ** 3),
         P_rated,
         0]
    )
    return power

def dbscan_outlier_detection_new(
    df,
    wind_col='wind_speed_235_avg',
    power_col='power_2_avg',
    eps=0.08,
    min_samples=25,
    v_in=3.0,
    v_r=10.0,
    v_out=25.0,
    plot=True
):

    """
    Perform DBSCAN-based outlier detection using wind turbine SCADA data.

    The method clusters (wind speed, power) pairs after scaling and labels
    outliers as anomalies. A theoretical power curve is used for context
    in visualization.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing wind and power columns.
    wind_col : str
        Name of the wind speed column. Default is 'wind_speed_235_avg'.
    power_col : str
        Name of the power output column. Default is 'power_2_avg'.
    eps : float
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered a core point.
    v_in : float
        Cut-in wind speed (m/s). Default is 3.0.
    v_r : float
        Rated wind speed (m/s). Default is 10.0.
    v_out : float
        Cut-out wind speed (m/s). Default is 25.0.
    plot : bool
        Whether to plot the clustering results with the theoretical power curve.

    Returns
    -------
    df : pd.DataFrame
        The original DataFrame with additional columns:
        - 'cluster': cluster label assigned by DBSCAN
        - 'label': 'normal' or 'anomaly' based on DBSCAN output
    """  
    
    
    df = df.copy()
    X = df[[wind_col, power_col]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(X_scaled)
   
    # Assign numeric cluster
    df['cluster'] = labels
    
    # ✅ Improved labeling (binary labels)
    df['label'] = np.where(labels == -1, 'anomaly', 'normal')

    # Silhouette Score
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        score = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {score:.4f} (clusters: {n_clusters})")
    else:
        print("Silhouette Score not valid.")

    # Plot
    if plot:
        v_plot = np.linspace(0, max(df[wind_col].max(), v_out) * 1.1, 300)
        P_rated = df[power_col].max()
        P_theo_plot = theoretical_power(v_plot, v_in, v_r, v_out, P_rated)

        plt.figure(figsize=(16, 8))
        plt.scatter(df[df['label'] == 'normal'][wind_col], df[df['label'] == 'normal'][power_col], s=10, alpha=0.5, label="Normal")
        plt.scatter(df[df['label'] == 'anomaly'][wind_col], df[df['label'] == 'anomaly'][power_col], s=10, color='red', alpha=0.7, label="Anomaly")
        plt.plot(v_plot, P_theo_plot, color='orange', lw=2, label="Theoretical Curve")
        plt.xlabel("Wind Speed (m/s)")
        plt.ylabel("Power")
        plt.title(f"DBSCAN Outlier Detection (eps={eps}, min_samples={min_samples})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return df





 

#######################################################################################################################
def downsample_to_hourly_with_index(df, time_col="time_stamp", interval_minutes=10):
    """
    Downsample time series data to hourly resolution while tracking original row indices.

    This function ensures that each hour is constructed from at least 6 samples (e.g., if original frequency is 10 minutes),
    and retains a mapping of source indices for potential back-referencing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time series data.
    time_col : str
        Name of the timestamp column. Default is "time_stamp".
    interval_minutes : int
        Expected frequency of input data in minutes (used for reindexing). Default is 10.

    Returns
    -------
    df_hourly : pd.DataFrame
        Hourly-aggregated DataFrame containing:
        - Mean of numeric columns.
        - A "source_indices" column with lists of row indices from the original data used in the hourly bin.

    Notes
    -----
    - Rows with fewer than 6 values (per hour) are filtered out.
    - Missing intervals are forward-filled during reindexing.
    """


    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    df["original_index"] = df.index  # Save original index
    df = df.set_index(time_col)

    df = df.asfreq(f"{interval_minutes}min", method='pad')
    grouped = df.groupby(pd.Grouper(freq="1H"))

    df_hourly = grouped.mean(numeric_only=True)
    df_hourly["source_indices"] = grouped["original_index"].apply(lambda x: list(x))

    valid_counts = grouped.count()
    valid_mask = (valid_counts >= 6).all(axis=1)
    df_hourly = df_hourly[valid_mask]

    df_hourly = df_hourly.reset_index()
    return df_hourly
#######################################################################################################################
 
def plot_anomalies_and_kde(df, wind_col='wind_speed_235_avg', power_col='power_2_avg', label_col='label'):
    """
    Visualizes wind turbine power anomalies and their distributions.

    This function:
    - Separates DBSCAN-labeled 'normal' and 'anomaly' points.
    - Filters out saturated power points (power ≥ 0.99) from the anomaly set.
    - Plots scatter plots of wind vs. power and their KDE distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing wind speed, power, and anomaly labels.
    wind_col : str
        Name of the wind speed column.
    power_col : str
        Name of the power output column.
    label_col : str
        Column name containing 'normal' and 'anomaly' labels (e.g., from DBSCAN).

    Returns
    -------
    df_refined : pd.DataFrame
        Filtered anomaly DataFrame where power < 0.99.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from scipy.stats import gaussian_kde

    df = df.copy()


    # KDE on normal power values
    normal_powers = df[df[label_col] == 'normal'][power_col]
    kde = gaussian_kde(normal_powers, bw_method=0.2)
    df['kde_density'] = kde(df[power_col])

    # Extract DBSCAN groups
    df_norm = df[df[label_col] == 'normal']
    df_anom = df[df[label_col] == 'anomaly']

    # Only remove saturated power values from anomaly set
    df_refined = df_anom[df_anom[power_col] < 0.99]

    # --- Plot
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    # 1. Original Anomalies
    axs[0, 0].scatter(df_anom[wind_col], df_anom[power_col], color='red', s=10, alpha=0.5)
    axs[0, 0].set_title("Original DBSCAN Anomalies")
    axs[0, 0].set_xlabel("Wind Speed (m/s)")
    axs[0, 0].set_ylabel("Power")
    sns.kdeplot(df_anom[power_col], bw_adjust=0.3, fill=True, color='red', ax=axs[0, 1])
    axs[0, 1].set_title("KDE of DBSCAN Anomalies")
    axs[0, 1].set_xlabel("Power")
    axs[0, 1].set_ylabel("Density")

    # 2. Normals
    axs[1, 0].scatter(df_norm[wind_col], df_norm[power_col], color='blue', s=10, alpha=0.5)
    axs[1, 0].set_title("DBSCAN Normals")
    axs[1, 0].set_xlabel("Wind Speed (m/s)")
    axs[1, 0].set_ylabel("Power")
    sns.kdeplot(df_norm[power_col], bw_adjust=0.3, fill=True, color='blue', ax=axs[1, 1])
    axs[1, 1].set_title("KDE of DBSCAN Normals")
    axs[1, 1].set_xlabel("Power")
    axs[1, 1].set_ylabel("Density")

    # 3. Cleaned Anomalies (just removed saturated power points)
    axs[2, 0].scatter(df_refined[wind_col], df_refined[power_col], color='green', s=10, alpha=0.5)
    axs[2, 0].set_title("Cleaned Anomalies (Power < 0.99)")
    axs[2, 0].set_xlabel("Wind Speed (m/s)")
    axs[2, 0].set_ylabel("Power")
    sns.kdeplot(df_refined[power_col], bw_adjust=0.3, fill=True, color='green', ax=axs[2, 1])
    axs[2, 1].set_title("KDE of Cleaned Anomalies")
    axs[2, 1].set_xlabel("Power")
    axs[2, 1].set_ylabel("Density")

    plt.tight_layout()
    plt.show()
    return df_refined


##################################################################

# Final refined version with annotation box placed in the top-left corner of the plot

 

def plot_prediction_with_leadtime_and_annotation(
    y_scores,
    y_pred,
    y_true,
    timestamps,
):
    
    """
    Plot anomaly scores, predictions, and ground truth over time with annotation of lead time and confidence.

    Parameters:
        y_scores (array-like): Anomaly scores
        y_pred (array-like): Binary predictions
        y_true (array-like): Ground truth binary labels
        timestamps (array-like): Corresponding time stamps
        threshold (float): Threshold used to generate y_pred
        min_event_length (int): Minimum number of consecutive predictions to count as meaningful

    Returns:
        dict: Info about the detection including timestamps, lead time, and confidence
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    def explain_lead_time(lead_time_minutes):
        days = lead_time_minutes // (60 * 24)
        hours = (lead_time_minutes % (60 * 24)) // 60
        minutes = lead_time_minutes % 60
        return f"{days}d {hours}h {minutes}m"

    # Prepare inputs
    timestamps = pd.to_datetime(timestamps).reset_index(drop=True)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    gt_indices = np.where(y_true == 1)[0]
    pred_indices = np.where(y_pred == 1)[0]

    if len(gt_indices) == 0 or len(pred_indices) == 0:
        print("❌ Ground truth or prediction is empty.")
        return {}

    gt_start = gt_indices[0]
    pred_start = pred_indices[0]

    lead_steps = max(0, gt_start - pred_start)
    lead_minutes = lead_steps * 10
    confidence = y_scores[pred_start:gt_start].max() if gt_start > pred_start else 0.0
    readable_lead_time = explain_lead_time(lead_minutes)

    # Plotting
    plt.figure(figsize=(18, 6))
    plt.plot(timestamps, y_scores, label="Anomaly Score", color='blue', linewidth=1)
    plt.plot(timestamps, y_pred * max(y_scores), label="Prediction", color='red', linestyle='--', linewidth=1)
    plt.plot(timestamps, y_true * max(y_scores), label="Ground Truth", color='black', linestyle=':', linewidth=3)

    # Vertical lines and markers
    plt.axvline(timestamps[pred_start], color='darkgreen', linestyle='--', linewidth=4, label='First Prediction')
    plt.axvline(timestamps[gt_start], color='darkorange', linestyle='--', linewidth=4, label='GT Start')
    plt.plot(timestamps[pred_start], y_scores[pred_start], 'o', markersize=15, color='darkgreen', label='Prediction Marker')
    plt.plot(timestamps[gt_start], y_scores[gt_start], 'X', markersize=15, color='darkorange', label='GT Marker')

    # Annotation in top-left corner
    plt.gca().text(
        0.01, 0.95,
        f"Lead Time: {readable_lead_time}\nConfidence: {confidence:.2f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
        verticalalignment='top',
        horizontalalignment='left'
    )

    plt.title("Anomaly Detection with Lead Time & Key Markers", fontsize=14, fontweight='bold')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Score / Label", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    res = {
        'gt_start': str(timestamps[gt_start]),
        'pred_start': str(timestamps[pred_start]),
        'lead_time_steps': lead_steps,
        'lead_time_minutes': lead_minutes,
        'lead_time_readable': readable_lead_time,
        'peak_confidence': confidence
    }

    return print(res)

#########################################################################

