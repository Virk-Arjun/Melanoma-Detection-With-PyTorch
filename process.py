import os
import cv2
import numpy as np

# one-hot vectors
# [1,0] = benign
# [0,1] = melanoma

# 50x50 pixels
img_size = 50 

# locations of image files 
ben_training_folder = "melanoma_dataset/train/benign/"
mal_training_folder = "melanoma_dataset/train/malignant/"

ben_testing_folder = "melanoma_dataset/test/benign/"
mal_testing_folder = "melanoma_dataset/test/malignant/"

# Load and process images function
def load_and_process_images(folder, label):
    data = [[np.array(cv2.resize(cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE), (img_size, img_size))), np.array(label)] for filename in os.listdir(folder)]
    return data

# Load training and testing data
ben_training_data = load_and_process_images(ben_training_folder, [1, 0])
mal_training_data = load_and_process_images(mal_training_folder, [0, 1])
ben_testing_data = load_and_process_images(ben_testing_folder, [1, 0])
mal_testing_data = load_and_process_images(mal_testing_folder, [0, 1])

# Convert lists to NumPy arrays
ben_training_data = np.array(ben_training_data, dtype=object)
mal_training_data = np.array(mal_training_data, dtype=object)
ben_testing_data = np.array(ben_testing_data, dtype=object)
mal_testing_data = np.array(mal_testing_data, dtype=object)

# Combine and shuffle training and testing data
training_data = np.concatenate((ben_training_data, mal_training_data), axis=0)
np.random.shuffle(training_data)

testing_data = np.concatenate((ben_testing_data, mal_testing_data), axis=0)
np.random.shuffle(testing_data)

# Save the processed data
np.save("melanoma_training_data.npy", training_data)
np.save("melanoma_testing_data.npy", testing_data) 

print()
print(f"Benign training count: {len(ben_training_data)}")
print(f"Malignant training count: {len(mal_training_data)}")
print()
print(f"Benign testing count: {len(ben_testing_data)}")
print(f"Malignant testing count: {len(mal_testing_data)}")
