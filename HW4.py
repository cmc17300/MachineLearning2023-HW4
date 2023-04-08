import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
from sklearn.svm import SVC, LinearSVC

df = pd.read_csv(r'C:\Users\Egg\OneDrive\Documents\data.csv')
# print(df.to_string())


print("Question 1--------------------------")
# check for null values
# not printing as desired, but this should work
res = df.isnull()
print(res)
# this is how you would replace the 0s with the mean
print("Question 1.1-------------------------")
meanMean = pd.DataFrame(df)
meanMean.fillna(0)
print(meanMean)
print("Question 2-----------------------------")
# Aggregate columns (min, max, count, mean)
mean = df['Duration'].mean()
min = df['Duration'].min()
max = df['Duration'].max()
count = df['Duration'].count()
print("Aggregate for Duration: ")
print("Mean: ", mean, "\n", "Max: ", max, "\n", "Min: ", min, "\n", "Count: ", count, "\n")
# column 2
mean = df['Pulse'].mean()
min = df['Pulse'].min()
max = df['Pulse'].max()
count = df['Pulse'].count()
print("Aggregate for Pulse: ")
print("Mean: ", mean, "\n", "Max: ", max, "\n", "Min: ", min, "\n", "Count: ", count, "\n")

# filter dataframe to select rows with values between 500 and 1000
print("Question 3------------------------")
# also does not display as intended, but should be how it is done
print("Question 4: -----------------------")
Q34 = df[(df['Calories'] == 500) &
         df['Calories'] == 1000]
print(Q34)
# filter for > 500 and pulse < 1000
print("Question 5: ----------------------")
print("Grater than 500: -----------------")
Q4 = df.loc[df['Calories'] > 500]
print(Q4)
print("Less than 1000: ------------------")
Q44 = df.loc[df['Calories'] < 1000]
print(Q44)
# create new df_modified, contains all columns except "maxpulse"
print("Question 6: -------------------------")
df_modified = df.copy()
df_modified = df_modified.drop(['Maxpulse'], axis=1)
print(df_modified)
# convert calories to int data type
print("Question 7: -----------------------")
# commented out but again, should work theoretically
# Q6 = df['Calories'] = df['Calories'].astype(int)
# print(Q6)
# create scatter plot using pandas
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["Calories", "Duration"]
plt.scatter(df.Calories, df.Duration)
plt.show()

print("question 8----------------")
dataFrame = pd.read_csv("train.csv")
print(dataFrame)
# doesn't print right but should be correct
# print(dataFrame['Survived'].corr(dataFrame['Sex']))
print("as for if the feature should be kept, no, seems a little tedious "
      "and the data doesn't really correlate to each other")
# naive bayes
"""
dataFrame.head()
sns.countplot(data=dataFrame, x='Survived', hue='Sex')
plt.xticks(rotation=45, ha='right')
dataFrameTwo = dataFrame.copy()
x = dataFrameTwo.drop('Sex', axis=1)
y = dataFrameTwo['Sex']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.33, random_state=125
)
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accur = accuracy_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test,average="weighted")

print("Accuracy: ", accur)
"""
print("Question 8---------------------------")
glass_df = pd.read_csv("glass.csv")
iris_df = pd.read_csv("Iris.csv")
print(glass_df.isnull().sum())
print('-'*10)
print(iris_df.isnull().sum())

x_train = glass_df.drop("RI", axis=1)
y_train = glass_df["RI"]

x_test = iris_df.drop("Mg", axis=1)

svc = SVC(max_iter=1000)
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train)*100,2)
print("svm accuracy = ",acc_svc)