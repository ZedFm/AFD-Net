import numpy as np
import matplotlib.pyplot as plt

def FITK_Curve(Am1, Cen1, HWHM1, tail_type='exponential', scale=None, StandardGaussian=True, plot=False):
    """
    Generates a 1-dimensional Gaussian curve, with an optional tail.
    The tail can be an exponential decay or an increase in HWHM on the right side of the peak.

    Parameters:
    Am1 (float): Amplitude of the Gaussian.
    Cen1 (float): Center position of the peak.
    HWHM1 (float): Initial half-width at half-maximum.
    tail_type (str): Type of the tail ('exponential', 'power_law', 'increasing_HWHM').
    scale (float): Scale of the tail's effect. Default values if None are provided.
    StandardGaussian (bool): If True, generate a standard Gaussian curve. If False, generate a Gaussian curve with a tail.
    plot (bool): If True, plot the Gaussian curve using matplotlib.

    Returns:
    np.ndarray: Array of y-values representing the Gaussian curve.
    """
    Std1 = HWHM1 / 2.355
    x = np.arange(0, 256, 0.25)
    
    # Generate the standard Gaussian curve
    gaussian = Am1 * np.exp(-((x - Cen1) ** 2) / (2 * Std1 ** 2))

    # If not a standard Gaussian, apply the tail effect
    if not StandardGaussian:
        # Default scales for tail if not provided
        if scale is None:
            scale = 50 if tail_type == 'exponential' else 3 if tail_type == 'power_law' else 0.18

        if tail_type == 'exponential':
            tail = np.where(x > Cen1, np.exp(-(x - Cen1) / scale), 0)
            gaussian *= (1 + tail)
        elif tail_type == 'power_law':
            tail = np.where(x > Cen1, (1 / ((x - Cen1 + 1) ** scale)), 0)
            gaussian *= (1 + tail)
        elif tail_type == 'increasing_HWHM':
            std_adjustment = np.where(x > Cen1, scale * (x - Cen1), 0)
            modified_std = Std1 + std_adjustment
            gaussian = Am1 * np.exp(-((x - Cen1) ** 2) / (2 * modified_std ** 2))
        else:
            raise ValueError("Invalid tail_type. Use 'exponential', 'power_law', or 'increasing_HWHM'.")
        # Combine the standard Gaussian with the tail
        # gaussian *= (1 + tail)

    # Plot the function if requested
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(x, gaussian, label=f'{"Standard" if StandardGaussian else "Tail"} Gaussian: Amplitude={Am1}, Center={Cen1}, HWHM={HWHM1}')
        plt.title('1D Gaussian Distribution' + ('' if StandardGaussian else ' with Tail'))
        plt.xlabel('X')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    return gaussian

# Example usage: Standard Gaussian
y_standard = FITK_Curve(44, 105, 2.355 * 6.5, StandardGaussian=True, plot=False)

# Example usage: Gaussian with Tail
y_tail = FITK_Curve(44, 105, 2.355 * 6.5, tail_type='increasing_HWHM', StandardGaussian=False, plot=True)

# Using Multiple y_tail and  to form LiDAR full-waveform signals