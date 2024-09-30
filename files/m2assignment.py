# Your name: Dhruv Shrivastava
# Your PSU Email: dvs6026@psu.edu

# Assignment name: M2 Assignment: Review Math and use the related Python Packages
# Module number: 2

# Please note: 
# 1. use print function instead of assert to print the results
# 2. upload a screenshot of your console to include your test results


from typing import List

##############   From Chpater 4. Linear Algebra    #############  

import numpy as np

#1.

#def add(v: Vector, w: Vector) -> Vector:
#    """Adds corresponding elements"""
#    assert len(v) == len(w), "vectors must be the same length"
#
#    return [v_i + w_i for v_i, w_i in zip(v, w)]
#
#assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

###### Your codes to replace above cell of codes: 

def add(v, w):
    """Adds corresponding elements"""
    return np.add(np.array(v), np.array(w))
add1 = add([1, 2, 3], [4, 5, 6])
print(f"1. Add : {add1} -> {np.array_equal(add1, [5, 7, 9])}")
print()

###########################

#2
#def subtract(v: Vector, w: Vector) -> Vector:
#    """Subtracts corresponding elements"""
#    assert len(v) == len(w), "vectors must be the same length"
#
#    return [v_i - w_i for v_i, w_i in zip(v, w)]
#
#assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

###### Your codes to replace above cell of codes:  

def subtract(v, w):
    """Subtracts corresponding elements"""
    return np.subtract(np.array(v), np.array(w))
sub1 = subtract([5, 7, 9], [4, 5, 6])
print(f"2. Subtract : {sub1} -> {np.array_equal(sub1, [1, 2, 3])}")
print()

###########################

#3

#def vector_sum(vectors: List[Vector]) -> Vector:
#    """Sums all corresponding elements"""
#    # Check that vectors is not empty
#    assert vectors, "no vectors provided!"
#
#    # Check the vectors are all the same size
#    num_elements = len(vectors[0])
#    assert all(len(v) == num_elements for v in vectors), "different sizes!"
#
#    # the i-th element of the result is the sum of every vector[i]
#    return [sum(vector[i] for vector in vectors)
#            for i in range(num_elements)]
#
#assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

###### Your codes to replace above cell of codes:  

def vector_sum(vectors):
    """Sums all corresponding elements"""
    if not vectors:
        print("no vectors provided!")
    vectors = np.array(vectors)
    if not all(len(vector) == len(vectors[0]) for vector in vectors):
        print("different sizes!")
    return np.sum(vectors, axis=0)
vecsum1 = vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]])
print(f"3. Vector Sum : {vecsum1} -> {np.array_equal(vecsum1, [16, 20])}")
print()
    
###########################

#4

#def scalar_multiply(c: float, v: Vector) -> Vector:
#    """Multiplies every element by c"""
#    return [c * v_i for v_i in v]
#
#assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

###### Your codes to replace above cell of codes: 

def scalar_multiply(c, v):
    """Multiplied every element by c"""
    if not v:
        print("no vectors provided!")
    return c*np.array(v)
scalmul1 = scalar_multiply(2, [1, 2, 3])
print(f"4. Scalar Multiply : {scalmul1} -> {np.array_equal(scalmul1, [2, 4, 6])}")
print()

###########################

#5

#def vector_mean(vectors: List[Vector]) -> Vector:
#    """Computes the element-wise average"""
#    n = len(vectors)
#    return scalar_multiply(1/n, vector_sum(vectors))
#
#assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


###### Your codes to replace above cell of codes:  

def vector_mean(vectors):
    """Computes the element-wise average"""
    if not vectors:
        print("no vectors provided!")
    vectors = np.array(vectors)
    return np.mean(vectors, axis=0)
vecavg1 = vector_mean([[1, 2], [3, 4], [5, 6]])
print(f"5. Vector Mean : {vecavg1} -> {np.array_equal(vecavg1, [3, 4])}")
print()

###########################

#6

#def dot(v: Vector, w: Vector) -> float:
#    """Computes v_1 * w_1 + ... + v_n * w_n"""
#    assert len(v) == len(w), "vectors must be same length"
#
#    return sum(v_i * w_i for v_i, w_i in zip(v, w))
#
#assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6
#

###### Your codes to replace above cell of codes:  

def dot(v, w):
    """Computes v_1 * w_1 + ... v_n * w_n"""
    if not v or not w:
        print("vector not provided for one of the arguments")
    v = np.array(v)
    w = np.array(w)
    if not len(v) == len(w):
        print("vectors must be the same length")
    return np.dot(v,w)
dot1 = dot([1, 2, 3], [4, 5, 6])
print(f"6. Dot : {dot1} -> {np.equal(dot1, 32)}")
print()
   
###########################

#7

#def sum_of_squares(v: Vector) -> float:
#    """Returns v_1 * v_1 + ... + v_n * v_n"""
#    return dot(v, v)
#
#assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3
#

###### Your codes to replace above cell of codes: 

def sum_of_squares(v):
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    if not v:
        print("vector not provided!")
    return np.sum(np.square(np.array(v)))
sumsqrs1 = sum_of_squares([1, 2, 3])
print(f"7. Sum of Squares : {sumsqrs1} -> {np.equal(sumsqrs1, 14)}")
print()

###########################

#8

#def magnitude(v: Vector) -> float:
#    """Returns the magnitude (or length) of v"""
#    return math.sqrt(sum_of_squares(v))   # math.sqrt is square root function
#
#assert magnitude([3, 4]) == 5

###### Your codes to replace above cell of codes: 

def magnitude(v):
    """Returns the magnitude (or length) of v"""
    if not v:
        print("vector not provided!")
    v = np.array(v)
    return np.linalg.norm(v)
mag1 = magnitude([3, 4])
print(f"8. Magnitude {mag1} : -> {np.equal(mag1, 5)}")
print()

###########################

#9

#def squared_distance(v: Vector, w: Vector) -> float:
#    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
#    return sum_of_squares(subtract(v, w))

###### Your codes to replace above cell of codes: 

def squared_distance(v, w):
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    if not v or not w:
        print("vector not provided for one of the arguments")
    return np.square(np.linalg.norm(subtract(v, w)))
sqrdist1 = squared_distance([1, 2], [4, 6])
print(f"9. Squared Distance : {sqrdist1} -> {np.equal(sqrdist1, 25.0)}")
print()

###########################

#10

#def distance(v: Vector, w: Vector) -> float:
#    """Computes the distance between v and w"""
#    return math.sqrt(squared_distance(v, w))
#
#
###### Your codes to replace above cell of codes: 

def distance(v, w):
    """Computes the distance between v and w"""
    if not v or not w:
        print("vector not provided for one of the arguments")
    return np.linalg.norm(subtract(v, w))
dist1 = distance([1, 4], [4, 8])
print(f"10. Distance : {dist1} -> {np.equal(dist1, 5.0)}")
print()

###########################

#11

#def shape(A: Matrix) -> Tuple[int, int]:
#    """Returns (# of rows of A, # of columns of A)"""
#    num_rows = len(A)
#    num_cols = len(A[0]) if A else 0   # number of elements in first row
#    return num_rows, num_cols
#
#assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 rows, 3 columns
    
###### Your codes to replace above cell of codes: 

def shape(A):
    """Returns (# of rows of A, # of columns of A)"""
    return np.shape(np.array(A))
shape1 = shape([[1, 2, 3], [4, 5, 6]])
print(f"11. Shape : {shape1} -> {np.array_equal(shape1, (2, 3))}")

###########################

#############  From Chpater 5. Statistics  #############
import pandas as pd

num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

#from collections import Counter
#import matplotlib.pyplot as plt
#
#friend_counts = Counter(num_friends)
#xs = range(101)                         # largest value is 100
#ys = [friend_counts[x] for x in xs]     # height is just # of friends
#plt.bar(xs, ys)
#plt.axis([0, 101, 0, 25])
#plt.title("Histogram of Friend Counts")
#plt.xlabel("# of friends")
#plt.ylabel("# of people")
#plt.show()
#
#num_points = len(num_friends)               # 204
#
#
#assert num_points == 204
#
#largest_value = max(num_friends)            # 100
#smallest_value = min(num_friends)           # 1
#
#
#assert largest_value == 100
#assert smallest_value == 1
#
#sorted_values = sorted(num_friends)
#smallest_value = sorted_values[0]           # 1
#second_smallest_value = sorted_values[1]    # 1
#second_largest_value = sorted_values[-2]    # 49
#
#
#assert smallest_value == 1
#assert second_smallest_value == 1
#assert second_largest_value == 49

import matplotlib.pyplot as plt

# plotting friends vs people in a histogram

friendCount_series = pd.Series(num_friends)
friendCounts = friendCount_series.value_counts().sort_index()
friendCounts_index = friendCounts.index.astype(int)

xs = range(friendCounts_index.min(), friendCounts_index.max() + 1)
ys = [friendCounts.get(x, 0) for x in xs]

plt.bar(xs, ys)
plt.axis([0, max(xs) + 1, 0, max(ys) + 1])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()

# statistics
min_friends = friendCounts_index.min()
max_friends = friendCounts_index.max()
sorted_friends = list(friendCounts_index.sort_values())

# print("Plot Statistics")
print()

# print("Series Statistics")
# print(f"Smallest Value : {min_friends} -> {np.equal(min_friends, 1)}")

#12 

# Hint: convert list to pandas object Series

#def mean(xs: List[float]) -> float:
#    return sum(xs) / len(xs)
#
#mean(num_friends)   # 7.333333
#
#
#assert 7.3333 < mean(num_friends) < 7.3334

###### Your codes to replace above cell of codes: 

def mean(xs):
    """Finds the average value of x"""
    if not xs:
        print("vector not provided!")
    series = pd.Series(xs)  # creating series 
    return series.mean()
friendsMean = mean(num_friends)
print(f"12. Mean : {friendsMean} -> {np.equal(int(mean(friendsMean)), 7)}")  # Mean = 7.333333
print()
###########################

#13

# The underscores indicate that these are "private" functions, as they're
# intended to be called by our median function but not by other people
# using our statistics library.
#def _median_odd(xs: List[float]) -> float:
#    """If len(xs) is odd, the median is the middle element"""
#    return sorted(xs)[len(xs) // 2]
#
#def _median_even(xs: List[float]) -> float:
#    """If len(xs) is even, it's the average of the middle two elements"""
#    sorted_xs = sorted(xs)
#    hi_midpoint = len(xs) // 2  # e.g. length 4 => hi_midpoint 2
#    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2
#
#def median(v: List[float]) -> float:
#    """Finds the 'middle-most' value of v"""
#    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)
#
#assert median([1, 10, 2, 9, 5]) == 5
#assert median([1, 9, 2, 10]) == (2 + 9) / 2
#
#
#assert median(num_friends) == 6

#def median(v: List[float]) -> float:
#    """Finds the 'middle-most' value of v"""
#    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


###### Your codes to replace above cell of codes: 

def median(v):
    """Finds the 'middle-most' value of v"""
    if not v:
        print("vector not provided!")
    v = pd.Series(v)
    return v.median()
medt1 = median([1, 10, 2, 9, 5])
medt2 = median([1, 9, 2, 10])
friendsMedian = median(num_friends)
print(f"13. Median   : {medt1} -> {np.equal(medt1, 5)}")
print(f"13.1. Median 2 : {medt2} -> {np.equal(medt2, ((2 + 9) / 2))}")
print(f"13.2. Friends Median : {friendsMedian} -> {np.equal(friendsMedian, 6.0)}")
print()

###########################

#14

#def quantile(xs: List[float], p: float) -> float:
#    """Returns the pth-percentile value in x"""
#    p_index = int(p * len(xs))
#    return sorted(xs)[p_index]
#
#assert quantile(num_friends, 0.10) == 1
#assert quantile(num_friends, 0.25) == 3
#assert quantile(num_friends, 0.75) == 9
#assert quantile(num_friends, 0.90) == 13
#
###### Your codes to replace above cell of codes: 

def quantiles(xs, p):
    """Returns the pth-percentile value in x"""
    if not xs:
        print("vector not provided!")
    series = pd.Series(xs)
    return series.quantile(p)
friends10Quantile = quantiles(num_friends, 0.10)
friends25Quantile = quantiles(num_friends, 0.25)
friends75Quantile = quantiles(num_friends, 0.75)
friends90Quantile = quantiles(num_friends, 0.90)
print("14. Quantiles of Number of Friends")
print(f"14.1 0.10 : {friends10Quantile}  -> {np.equal(friends10Quantile, 1)}")
print(f"14.2 0.25 : {friends25Quantile}  -> {np.equal(friends25Quantile, 3)}")
print(f"14.3 0.75 : {friends75Quantile}  -> {np.equal(friends75Quantile, 9)}")
print(f"14.4 0.90 : {friends90Quantile} -> {np.equal(friends90Quantile, 13)}")
print()

###########################

#15
#Hint You may use Pandas DataFrame: mode() function

#def mode(x: List[float]) -> List[float]:
#    """Returns a list, since there might be more than one mode"""
#    counts = Counter(x)
#    max_count = max(counts.values())
#    return [x_i for x_i, count in counts.items()
#            if count == max_count]
#
#assert set(mode(num_friends)) == {1, 6}
#
###### Your codes to replace above cell of codes: 

def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more than one mode"""
    if not x:
        print("vector not provided!")
    s = pd.Series(x)
    return s.mode()
friendsMode = set(mode(num_friends))
print(f"15. Mode: {friendsMode} -> {np.array_equal(friendsMode, {1, 6})}")
print()

###########################

# No related fuction in numpy or pandas, you can skip

## "range" already means something in Python, so we'll use a different name
#def data_range(xs: List[float]) -> float:
#    return max(xs) - min(xs)
#
#assert data_range(num_friends) == 99

#16 

# You can define variance directy by using pandas

#from scratch.linear_algebra import sum_of_squares
#
#def de_mean(xs: List[float]) -> List[float]:
#    """Translate xs by subtracting its mean (so the result has mean 0)"""
#    x_bar = mean(xs)
#    return [x - x_bar for x in xs]
#
#def variance(xs: List[float]) -> float:
#    """Almost the average squared deviation from the mean"""
#    assert len(xs) >= 2, "variance requires at least two elements"
#
#    n = len(xs)
#    deviations = de_mean(xs)
#    return sum_of_squares(deviations) / (n - 1)
#
#assert 81.54 < variance(num_friends) < 81.55

###### Your codes to replace above cell of codes: 

def variance(xs):
    """Almost the average squared deviation from the mean"""
    if not xs:
        print("vector not provided")
    df = pd.DataFrame(xs)
    return df.var()
friendsVar = variance(num_friends).iloc[0]
print(f"16. Variance : 81.54 < {friendsVar} < 81.55 -> {81.54 < friendsVar and friendsVar < 81.55}")
print()

###########################

#17

#import math
#
#def standard_deviation(xs: List[float]) -> float:
#    """The standard deviation is the square root of the variance"""
#    return math.sqrt(variance(xs))
#
#
###### Your codes to replace above cell of codes: 

def standard_deviation(xs):
    """The standard deviation is the square root of the variance"""
    if not xs:
        print("vector not provided!")
    df = pd.DataFrame(xs)
    return df.std()
friendsStDev = standard_deviation(num_friends).iloc[0]
print(f"17. Standard Deviation : 9.03 < {friendsStDev} < 9.04 -> {9.03 < friendsStDev and friendsStDev < 9.04}")
print()

###########################

# Skip this
#def interquartile_range(xs: List[float]) -> float:
#    """Returns the difference between the 75%-ile and the 25%-ile"""
#    return quantile(xs, 0.75) - quantile(xs, 0.25)
#
#assert interquartile_range(num_friends) == 6
#
#

daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

daily_hours = [dm / 60 for dm in daily_minutes]

#18

#from scratch.linear_algebra import dot
#
#def covariance(xs: List[float], ys: List[float]) -> float:
#    assert len(xs) == len(ys), "xs and ys must have same number of elements"
#
#    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)
#
#assert 22.42 < covariance(num_friends, daily_minutes) < 22.43
#assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60

###### Your codes to replace above cell of codes: 

def covariance(xs, ys):
    if not xs or not ys:
        print("vector not provided for one of the arguments")
    s1 = pd.Series(xs)
    s2 = pd.Series(ys)
    return s1.cov(s2)
friendsDminsCov = covariance(num_friends, daily_minutes)
friendsDhrsCov = covariance(num_friends, daily_hours)
print("18. Friends Covariance")
print(f"18.1 Minutes: 22.42 <  {friendsDminsCov} < 22.43 -> {22.42 < friendsDminsCov and friendsDminsCov < 22.43}")
print(f"18.2 Hours: 22.42/60 < {friendsDhrsCov} < 22.43/60 -> {22.42/60 < friendsDhrsCov and friendsDhrsCov < 22.43/60}")
print()

###########################

#19

#def correlation(xs: List[float], ys: List[float]) -> float:
#    """Measures how much xs and ys vary in tandem about their means"""
#    stdev_x = standard_deviation(xs)
#    stdev_y = standard_deviation(ys)
#    if stdev_x > 0 and stdev_y > 0:
#        return covariance(xs, ys) / stdev_x / stdev_y
#    else:
#        return 0    # if no variation, correlation is zero
#
#assert 0.24 < correlation(num_friends, daily_minutes) < 0.25
#assert 0.24 < correlation(num_friends, daily_hours) < 0.25

###### Your codes to replace above cell of codes: 

def correlation(xs, ys):
    if not xs or not ys:
        print("vector not provided for one of the arguments")
    s1 = pd.Series(xs)
    s2 = pd.Series(ys)
    return s1.corr(s2)

friendsDminsCorr = correlation(num_friends, daily_minutes)
friendsDhrsCorr = correlation(num_friends, daily_hours)
print("19. Friends Correlation")
print(f"19.1 Minutes: 0.24 < {friendsDminsCorr} < 0.25 -> {0.24 < friendsDminsCorr and friendsDminsCorr < 0.25}")
print(f"19.2 Hours: 0.24 < {friendsDhrsCorr} < 0.25 -> {0.24 < friendsDhrsCorr and friendsDhrsCorr < 0.25}")

print()
###########################

###############

# outlier = num_friends.index(100)    # index of outlier

# num_friends_good = [x
#                     for i, x in enumerate(num_friends)
#                     if i != outlier]

# daily_minutes_good = [x
#                       for i, x in enumerate(daily_minutes)
#                       if i != outlier]

# daily_hours_good = [dm / 60 for dm in daily_minutes_good]

# assert 0.57 < correlation(num_friends_good, daily_minutes_good) < 0.58
# assert 0.57 < correlation(num_friends_good, daily_hours_good) < 0.58

#############  From Chpater 6. Probability  #############

import scipy.stats as ss
import matplotlib.pyplot as plt

# 20
#def uniform_cdf(x: float) -> float:
#    """Returns the probability that a uniform random variable is <= x"""
#    if x < 0:   return 0    # uniform random is never less than 0
#    elif x < 1: return x    # e.g. P(X <= 0.4) = 0.4
#    else:       return 1    # uniform random is always less than 1
#
#SQRT_TWO_PI = math.sqrt(2 * math.pi)

###### Your codes to replace above cell of codes: 

def uniform_cdf(x):
    """Returns the probability that a uniform random variable is <= x
       if   x < 0: return 0 -> uniform random is never < 0
       elif x < 1: return x -> 0 <= x < 1
       else:       return 1 -> uniform random is never > 1
    """
    return ss.uniform.cdf(x, loc=0, scale=1)

print("20. Uniform CDF (No Graph)")
print()

###########################

# cdf(x, loc=0, scale=1)

# 21
#import math
#def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
#    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

###### Your codes to replace above cell of codes: 

def normal_pdf(x, mu=0, sigma=1):
    return ss.norm.pdf(x, loc=mu, scale=sigma)


###########################

print("21. Normal PDF")

### th following codes for drawing the graphs
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()

print()

# 22
#def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
#    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

###### Your codes to replace above cell of codes: 

def normal_cdf(x, mu=0, sigma=1):
    return ss.norm.cdf(x, loc=mu, scale=sigma)

###########################

print("22. Normal CDF")

### th following codes for drawing the graphs
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4) # bottom right
plt.title("Various Normal cdfs")
plt.show()