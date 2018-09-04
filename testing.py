import numpy as np

Data = np.array([1, 2, 3, 4, 5, 6, 6.1, 6.2, 6.25, 6.31, 6.7, 7, 8, 9])
sortedData,length = sorted(Data),len(Data)
Difference = [Data[i+1]-Data[i] for i in range(length) if i+1 < length]
MinDifference = np.min(Difference)
print('MinDifference',MinDifference)
Mode = (Data[np.argmin(Difference)]+Data[np.argmin(Difference)+1])/2
print('Mode',Mode)