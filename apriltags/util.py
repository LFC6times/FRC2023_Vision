import numpy as np

def writeToFile(arr: np.ndarray, fileName: str):
    with open(fileName, 'wb') as f:
        np.save(f, arr)

def readFromFile(fileName: str) -> np.ndarray:
    with open(fileName, 'rb') as f:
        return np.load(f)