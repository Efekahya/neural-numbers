
import pandas as pd
import numpy as np
from sklearn_som.som import SOM


def readDataAndValuesFromXLSX(dataFile, resultFile):
    # Read data from excel file
    data = pd.read_excel(dataFile, index_col=0)
    result = pd.read_excel(resultFile, index_col=0)

    data.reset_index(drop=True, inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Add result column to data
    concat = pd.concat([data, result["label"]], axis=1)

    return concat

# print("Data and values from excel file: ", readDataAndValuesFromXLSX("dataset.xlsx","index.xlsx"))


def normalizeRows(data):
    # Drop result column
    dataWithoutResult = data.drop(["label"], axis=1)

    # Normalize data 0 to 1
    dataWithoutResult = dataWithoutResult.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
    # Add result column to data
    concat = pd.concat([dataWithoutResult, data["label"]], axis=1)

    return concat
# print("Normalize data: ", normalizeRows(readDataAndValuesFromXLSX("dataset.xlsx","index.xlsx")))


def splitDataTrainAndTest(data):
    # Split data to train and test
    train = data.sample(frac=0.5, random_state=42)
    test = data.drop(train.index)

    return train, test
# print(splitDataTrainAndTest(readDataAndValuesFromXLSX("dataset.xlsx","index.xlsx")))


data = normalizeRows(readDataAndValuesFromXLSX("dataset.xlsx", "index.xlsx"))

train, test = splitDataTrainAndTest(data)

trainToNumPy = train.to_numpy()
testToNumPy = test.to_numpy()

iris_som = SOM(m=1, n=4, dim=784, random_state=12)
# Train data
iris_som.fit(trainToNumPy)

# Predict test data
prediction = iris_som.predict(testToNumPy)

# Save predictions to a file
np.savetxt("kume-sonuc.txt", np.dstack((np.arange(1,
           prediction.size+1), prediction))[0], fmt="%d\t\tC%d")


def getEachAttributesAccuracy(trainToNumPy, predictions):
    uniqueAttributes, counts = np.unique(
        trainToNumPy[:, -1], return_counts=True)
    accuracyArray = []

    for i in range(len(uniqueAttributes)):
        correct = 0
        for j in range(len(predictions)):
            if trainToNumPy[j][-1] == uniqueAttributes[i] and predictions[j] == uniqueAttributes[i]:
                correct += 1
        accuracyArray.append(correct/counts[i])
    return accuracyArray


print('getEachAttributesAccuracy: ',
      getEachAttributesAccuracy(trainToNumPy, prediction))
correct = 0
convertPrediction = []
for i in range(len(prediction)):
    if prediction[i] == 0:
        convertPrediction.append(0)
    elif prediction[i] == 1:
        convertPrediction.append(3)
    elif prediction[i] == 2:
        convertPrediction.append(8)
    elif prediction[i] == 3:
        convertPrediction.append(9)


for i in range(len(testToNumPy)):
    if testToNumPy[i][-1] == convertPrediction[i]:
        correct += 1

print("Accuracy: ", correct/len(testToNumPy))

input("Press Enter to continue...")
