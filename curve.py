import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

def optimize_stiffener_curve(L, h, n, t, plot=True):
    """
    Optimize the stiffener support curve while ensuring the slope does not exceed 45 degrees.

    Parameters:
        L (float): Total height of the box.
        h (float): Height of stiffener.
        n (int): Number of stiffeners.
        t (float): Extremity thickness of stiffeners.
        plot (bool): Whether to generate the plot (default: True).

    Returns:
        tuple: (t - y0, area_above_curve)
    """
    # Calculate pitch and curve width
    P = (L - h) / (n - 1)
    curve_width = (P - h) / 2  # Half-width of the curve

    # Start with initial height guess and reduce if needed
    y0 = t

    # Define x range
    x_range = np.linspace(-curve_width, curve_width, 1000)

    # Adjust y0 until all slopes are ≤ 45 degrees
    while True:
        # Define the parabolic curve: y = a*x^2 + y0
        a = -y0 / curve_width**2  # Ensuring parabola peaks at (0, y0)
        y_curve = a * x_range**2 + y0

        # Analytical derivative dy/dx = 2 * a * x
        derivative = 2 * a * x_range

        # Compute angles in degrees
        angles = np.abs(np.arctan(derivative) * (180 / np.pi))

        # If all angles are <= 45 degrees, break loop
        if np.all(angles <= 45):
            break

        # Reduce y0 slightly if any angle exceeds 45 degrees
        y0 -= 0.01

    # Bounding box area
    box_area = 2 * curve_width * y0

    # Integrate the area under the curve
    area_under_curve, _ = spi.quad(lambda x: a * x**2 + y0, -curve_width, curve_width)

    # Area above the curve
    area_above_curve = box_area - area_under_curve



    # Generate plot if requested
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(x_range, y_curve, label=f"Optimized Curve (y0={y0:.2f})", color="blue")
        plt.axhline(0, color='gray', linestyle='--', label="Stiffener Base")
        plt.axvline(-curve_width, color='red', linestyle='--', label="Extremity")
        plt.axvline(curve_width, color='red', linestyle='--')
        plt.fill_between(x_range, y_curve, y0, color='lightblue', alpha=0.5, label="Area Above Curve")
        plt.xlabel("Width (x)")
        plt.ylabel("Height (y)")
        plt.legend()
        plt.title("Optimized Support Curve with Area Calculation")
        plt.grid()
        plt.show()
        
        # Print results
        print(f"Optimized curve height (y0) to keep slope ≤ 45°: {y0:.4f}")
        print(f"Adjusted thickness (t - y0): {t - y0:.4f}")
        print(f"Area above the curve: {area_above_curve:.4f} [mm^2]")

    return t - y0, area_above_curve

# Example Usage
#adjusted_thickness, area_above = optimize_stiffener_curve(L=90, h=5, n=2, t=5, plot=True)
