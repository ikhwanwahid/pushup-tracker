"""Temporal feature computation: velocities, acceleration, smoothing."""

import numpy as np
from scipy.signal import savgol_filter


def compute_velocity(sequence: np.ndarray, fps: float) -> np.ndarray:
    """Compute velocity (first derivative) of a time series.

    Args:
        sequence: Array of shape (T, D) — D-dimensional features over T frames.
        fps: Frames per second for proper time scaling.

    Returns:
        Array of shape (T, D). First frame velocity is set to zero.
    """
    if len(sequence) < 2:
        return np.zeros_like(sequence)

    dt = 1.0 / fps
    velocity = np.zeros_like(sequence)
    velocity[1:] = (sequence[1:] - sequence[:-1]) / dt
    return velocity


def compute_acceleration(sequence: np.ndarray, fps: float) -> np.ndarray:
    """Compute acceleration (second derivative) of a time series.

    Args:
        sequence: Array of shape (T, D).
        fps: Frames per second.

    Returns:
        Array of shape (T, D). First two frames are set to zero.
    """
    velocity = compute_velocity(sequence, fps)
    return compute_velocity(velocity, fps)


def smooth_sequence(
    sequence: np.ndarray,
    window: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a sequence.

    Args:
        sequence: Array of shape (T, D) or (T,).
        window: Window length for the filter (must be odd and > polyorder).
        polyorder: Polynomial order for the filter.

    Returns:
        Smoothed sequence with the same shape.
    """
    if len(sequence) < window:
        return sequence.copy()

    if sequence.ndim == 1:
        return savgol_filter(sequence, window, polyorder)

    result = np.zeros_like(sequence)
    for d in range(sequence.shape[1]):
        result[:, d] = savgol_filter(sequence[:, d], window, polyorder)
    return result
