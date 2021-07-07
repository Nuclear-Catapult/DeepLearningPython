import network
import numpy as np

training_data = [
   (np.array([[-1],[-1]], dtype=np.float32), np.array([[1], [0]], dtype=np.float64)),
   (np.array([[-1],[1]], dtype=np.float32), np.array([[0], [1]], dtype=np.float64)),
   (np.array([[1],[-1]], dtype=np.float32), np.array([[0], [1]], dtype=np.float64)),
   (np.array([[1],[1]], dtype=np.float32), np.array([[1], [0]], dtype=np.float64))
]

test_data = [
    (np.array([[-1],[-1]], dtype=np.float32), np.int64(0)),
    (np.array([[-1],[1]], dtype=np.float32), np.int64(1)),
    (np.array([[1],[-1]], dtype=np.float32), np.int64(1)),
    (np.array([[1],[1]], dtype=np.float32), np.int64(0))
]

net = network.Network()
net.SGD(training_data, 10000, 4, 10, test_data=test_data)
