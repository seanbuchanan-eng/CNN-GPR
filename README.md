# CNN-GPR

CNN-GPR model developed to predict battery SOH using PRIMED battery data.

The data in the Data folder are numpy arrays of shape (m, n, p). Where m is the number of charge cycles, n is the amount of data points in each charge cycle, and p is the data channel (voltage, current, either capacity or temperature or nothing). The naming is formated as follows:

  - The 'MG' prefix relates to the microgrid or 'dynamic' data that follows a solar cycle.
  - The 'features' data refers to the model input data.
  - The 'labels' data refers to the model output data.
