import numpy as np
import matplotlib.pyplot as plt

def FITK_Curve_with_tail(Am1, Cen1, HWHM1, tail_type='exponential', scale=None, plot=False):
    """
    Generates a 1-dimensional Gaussian curve with a tail.
    The tail can either be an exponential decay or an increase in HWHM on the right side of the peak.

    Parameters:
    Am1 (float): Amplitude of the Gaussian.
    Cen1 (float): Center position of the peak.
    HWHM1 (float): Initial half-width at half-maximum.
    tail_type (str): Type of the tail ('exponential', 'power_law', 'increasing_HWHM').
    scale (float): Scale of the tail's effect. Default values if None are provided.
    plot (bool): If True, plot the Gaussian curve using matplotlib.

    Returns:
    np.ndarray: Array of y-values representing the Gaussian curve with a tail.
    """
    Std1 = HWHM1 / 2.355
    x = np.arange(0, 256, 0.25)
    gaussian = Am1 * np.exp(-((x - Cen1) ** 2) / (2 * Std1 ** 2))

    # Tail implementation based on the chosen type
    if scale is None:
        scale = 50 if tail_type == 'exponential' else 3 if tail_type == 'power_law' else 0.18  # Default scales

    if tail_type == 'exponential':
        tail = np.where(x > Cen1, np.exp(-(x - Cen1) / scale), 0)
    elif tail_type == 'power_law':
        tail = np.where(x > Cen1, (1 / ((x - Cen1 + 1) ** scale)), 0)
    elif tail_type == 'increasing_HWHM':
        std_adjustment = np.where(x > Cen1, scale * (x - Cen1), 0)
        modified_std = Std1 + std_adjustment
        gaussian = Am1 * np.exp(-((x - Cen1) ** 2) / (2 * modified_std ** 2))
        tail = 0  # No additional tail calculation needed
    else:
        raise ValueError("Invalid tail_type. Use 'exponential', 'power_law', or 'increasing_HWHM'.")

    y1 = gaussian * (1 + tail)

    # Plot the function if requested
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(x, y1, label=f'Gaussian with {tail_type} tail: Amplitude={Am1}, Center={Cen1}, Initial HWHM={HWHM1}, Scale={scale}')
        plt.title('1D Gaussian Distribution with Tail')
        plt.xlabel('X')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return y1

# Example usage
y1 = FITK_Curve_with_tail(44, 105, 2.355 * 6.5, tail_type='increasing_HWHM', plot=True)
