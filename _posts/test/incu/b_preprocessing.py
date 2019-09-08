import cv2
import numpy as np
def pre_processing(image_path,image_size,image_dimension ):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    data = gray.astype(np.float32)
    data = data[np.newaxis]
    data = data[np.newaxis]
    train_input = np.reshape(data, (1,image_size,image_size,image_dimension ))
    return train_input

