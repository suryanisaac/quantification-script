import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt

# -------
# Config
# -------


# Path to FIJI-Processed Image
image_path = "./sample.tif"

# Make True if last peak doesn't have a vertical line at the end. False if line is already present
AddFinalLimit = True

# Make True if first peak doesn't have a vertical line at the beginning. False if line is already present
AddInitialLimit = False

# Total number of peaks expected. Same as the number of spots on the image
numberOfSamples = 14

# Sensitivity for peak delineation. Lower is more sensitive. Between 0 and 1
sensitivity = 0.8

# Enter concentrations for the standards to automatically try a linear fit
# Disregard if non-linear fit is performed elsewhere
standards = [6, 4, 2, 1, 0.5, 0.1]

# Figure plotting styles (matplotlib.pyplot)
plt.style.use('default')
plt.rc('font', family='Andale Mono', size="20")

# ----------------
# Peak Delineation
# ----------------

# Open Image
img = Image.open(image_path)

# init image array and grayArray
image_array = np.array(img)
grayMean = []

for i in range(image_array.shape[1]): # from 0 to the end of the strip
    # all values from the 0 axis and one vertical slice from the 1 axis
    verticalSlice = image_array[:, i]
    verticalMean = np.mean(verticalSlice)
    grayMean.append(verticalMean)

# grayArray holds vertical means of pixels
grayArray = np.array(grayMean)
pixelArray = np.arange(0, image_array.shape[1])

# invert the signal and treat the minima as peaks, define a peak distance based on the number of expected peaks
peaks, [] = sp.signal.find_peaks(-grayArray, distance=int(sensitivity*grayArray.size/numberOfSamples))

# add a final line to close the last peak, if requested
if AddFinalLimit:
    if peaks.max() < grayArray.size:
        peaks = np.append(peaks, grayArray.size)
if AddInitialLimit:
    peaks = np.insert(peaks, 0, 0)

# Plot the mean gray value (Y axis) against horizontal pixel number (X axis)
plt.figure(1)
plt.plot(pixelArray, grayArray, color="black")
plt.xlabel("Distance (px)")
plt.ylabel("Mean Pixel Intensity")

# Add Vertical Lines where peaks are found
for i in range(len(peaks)):
    plt.axvline(peaks[i])
print(peaks)

# ----------------
# Peak Integration
# ----------------

# Find cumulative integrals for all X axis points, starting from 0
cumulativeIntegrals = sp.integrate.cumulative_trapezoid(grayArray, np.arange(0, grayArray.size))
#print(cumulativeIntegrals.size)

# Function to pull out individual peak integrals from the cumulative list
# Upper Cumulative - Lower Cumulative = Peak Integral
# Upper and lower come from the peaks list generated earlier


def cumulative_to_integral(cumulative_array, upper_lim, lower_lim):
    return cumulative_array[upper_lim] - cumulative_array[lower_lim]


integrals = []
for i in range(len(peaks)):
    try:
        integral = cumulative_to_integral(cumulativeIntegrals, peaks[i+1], peaks[i])
        print(f" Integral from {peaks[i]} to {peaks[i+1]} is {integral}")
        integrals.append(integral)
    except IndexError:
        # I don't know why you need [i+1]-2 but it works
        lastIntegral = cumulative_to_integral(cumulativeIntegrals, peaks[i+1]-2, peaks[i])
        integrals.append(lastIntegral)
        print(f" Last Integral from {peaks[i]} to {peaks[i+1]-2} is {lastIntegral}")
        print(f" Last Index of Cumulative Integral List is {cumulativeIntegrals.size-1}")
        break

print(f"Generated Integral List with {len(integrals)} elements. {numberOfSamples} peaks Expected.")

# -------------------------------
# Standards and Linear Regression
# -------------------------------

# Separate the Standards
standardIntegrals = np.array(integrals)[0:len(standards)]
print(standardIntegrals)

# Linear Regression
linear_model = sp.stats.linregress(np.array(standards), standardIntegrals)
print(f"R^2 is {np.square(linear_model.rvalue)}")

# Plot the line of best fit with standards
plt.figure(2)
plt.plot(standards, standardIntegrals, "o", label="Standards")
y = (linear_model.slope * np.array(standards)) + linear_model.intercept
plt.plot(standards, y, color="black", label=f"R² = {np.round(np.square(linear_model.rvalue), decimals=3)}")
plt.legend()
plt.grid()
plt.xlabel("Chlorogenic Acid Concentration (mg/ml)")
plt.ylabel("Peak Integral")

# Get quantified concentrations


def extrapolate_from_model(linearmodel, y_value):
    return (y_value-linearmodel.intercept)/linearmodel.slope


result = extrapolate_from_model(linear_model, integrals)
np.set_printoptions(precision=3)
print("List of Peak Integrals: ")
print(np.array(integrals))
print("Linear Regression Quantification Results: ")
print(result)
plt.show()
