# Aim: Compute mean, median, mode, variance, and standard deviation.
from statistics import mean, median, multimode, variance, stdev

data = [12, 15, 15, 18, 20, 20, 20, 25, 30]
print("Dataset:", data)
print("Mean:", mean(data))
print("Median:", median(data))
print("Mode(s):", multimode(data))
print("Sample variance:", round(variance(data), 4))
print("Sample standard deviation:", round(stdev(data), 4))
