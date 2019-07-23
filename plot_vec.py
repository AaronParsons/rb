import vec
import sys
import pylab as plt
import numpy as np

vec_num = 0
filename = sys.argv[-1]
if filename.endswith('.vec'):
    data = vec.read_vec(sys.argv[-1], 32, 32)
else:
    npz = np.load(filename)
    data = npz['ball']
frame_plt = plt.imshow(data[vec_num], cmap='gray')
plt.title(vec_num)

def press(event):
    global vec_num
    if event.key == 'n':
        vec_num += 1
        frame_plt.set_data(data[vec_num])
        plt.title(vec_num)
        plt.draw()

plt.connect('key_press_event', press)
plt.show()
