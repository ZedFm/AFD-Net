import numpy as np
import matplotlib.pyplot as plt

def FITK_Curve(Am1, Cen1, HWHM1, plot=False):
    """
    Generates a 1-dimensional Gaussian curve.

    Parameters:
    Am1 (float): Amplitude of the Gaussian.
    Cen1 (float): Center position of the peak.
    HWHM1 (float): Half-width at half-maximum.
    plot (bool): If True, plot the Gaussian curve using matplotlib.

    Returns:
    np.ndarray: Array of y-values representing the Gaussian curve.
    """
    # Convert HWHM to standard deviation
    Std1 = HWHM1 / 2.355
    
    # Generate x values
    x = np.arange(0, 256, 0.25)
    
    # Calculate the Gaussian function
    y1 = Am1 * np.exp(-((x - Cen1) ** 2) / (2 * Std1 ** 2))
    
    # Plot the function if requested
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(x, y1, label=f'Gaussian: Amplitude={Am1}, Center={Cen1}, HWHM={HWHM1}')
        plt.title('1D Gaussian Distribution')
        plt.xlabel('X')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return y1

# Example usage
y1 = FITK_Curve(44, 105, 2.355 * 6.5, plot=True)
