# import the required lib
import numpy as np  # linear algorithm
import pandas as pd  # data processing, CSV file I/O (eg: pd.read_csv)

# load the CSV file as data frame
df = pd.read_csv(
    r"/projects/AI2/AIprojects/machine/Rain-Prediction/weatherAUS.csv")
print('Size of weather data frame is :', df.shape)

# display data
print(df[0:5])

# checking null values
print(df.count().sort_values())

# as we can see the first four columns have less than 60% data, we can ignore these four columns
# we dont need the location column because
# we are going to find oif it will rain Australia(not location specifies)
# we are going to drop the data column too.
# we need to remove RISK_MM bcz we want to predict 'raintommorrow'and RISK_MM can leak

df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'Date'], axis=1)
print(df.shape)

# let us get rid of null values in df
df = df.dropna(how='any')
print(df.shape)

# its time to remove the outliers in our data - we are using 2-score to detect and remove the
from scipy import stats

z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
# lets deal with the categorical columns now
# simply change yes/no to 1/0 for Raintoday and raintomorrow
df['RainToday'].replace({'No': 0, 'Yes':1}, inplace=True)
df['RainTomorrow'].replace({'No':0, 'Yes':1}, inplace=True)

#see unique values and covert them to int using pd.getDummies()
categorical_column = [ 'WindGustDir','WindDir3pm','WindDir9am']
for col in categorical_column:
    print(np.unique(df[col]))
# transform the categorical column
df = pd.get_dummies(df, columns=categorical_column)
print(df.iloc[4:9])

#explaratory data analysis
from sklearn.feature_selection import SelectKBest, chi2
x=df.loc[:,df.columns != 'RainTomorrow']
y=df[['RainTomorrow']]
selector = SelectKBest(chi2, k=3)
selector.fit(x,y)
x_new = selector.transform(x)


#next step is to standardise our data - using MinMaxScaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit_transform(df)
df = pd.DataFrame(scaler.transform(df))
print(df.iloc[4:9])
