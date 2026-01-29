import numpy as np
# Importing NumPy for numerical operations, particularly for matrix and vector operations, and for log, sqrt.
import matplotlib.pyplot as plot
# Importing Matplotlib to plot the 3D surface of option greeks.
from scipy.stats import norm
# Using SciPy's norm, which provides the CDF and the PDF (derivative of CDF) needed to calculate gamma and delta.


# Define the Black-Scholes function for gamma
def blsgamma(S, K, r, T, sigma):
    # Calculate d1
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    # Return gamma value
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


# Define the Black-Scholes function for delta
def blsdelta(S, K, r, T, sigma):
    # Calculate d1
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    # Return delta value
    return norm.cdf(d1)


# Setting the price range of the options, and the time range: one year divided into half-months.
# Generate a range of stock prices from $10 to $70
Range = np.arange(10, 71)
Span = len(Range)
# Define the time range for the option, from 0.5 to 12 months in half-month increments
j = np.arange(1, 12.5, 0.5)
Newj = np.tile(j, (Span, 1)).T / 12

# For each time period, creating a vector of prices from 10 to 70 and creating a matrix of all ones.
# (Creating matrices for repeated use of Range and time intervals)
JSpan = np.ones((len(j), 1))
NewRange = np.tile(Range, (len(j), 1))
Pad = np.ones(Newj.shape)

# Calculating gamma and delta values: for each combination of stock price and time
# The exercise price is $40, the risk-free interest rate is 10%, and volatility is 0.35
ZVal = blsgamma(NewRange, 40*Pad, 0.1*Pad, Newj, 0.35*Pad)
Color = blsdelta(NewRange, 40*Pad, 0.1*Pad, Newj, 0.35*Pad)

# Plotting setup
figure = plot.figure(figsize=(12, 8))
figure.canvas.manager.set_window_title('Call Option Price Sensitivity')
ax = figure.add_subplot(111, projection='3d')

# Creating a surface plot for gamma values colored by delta values
surf = ax.plot_surface(NewRange, Newj*12, ZVal, facecolors=plot.cm.viridis(Color))
ax.set_xlabel('Stock Price ($)')
ax.set_ylabel('Time (months)')
ax.set_zlabel('Gamma')
ax.set_title('Call Option Price Sensitivity')

# Adjusting view settings and adding colorbar, same as one from Matlab example
ax.view_init(elev=40., azim=50)
figure.colorbar(surf, orientation='horizontal')

# Inverting axes to look like plot from Matlab example
ax.invert_xaxis()
ax.invert_yaxis()

# Display the plot
plot.show()
