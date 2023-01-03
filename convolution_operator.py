import numpy as np
import matplotlib.pyplot as plt

LEN = 100
y1 = np.sin(np.linspace(0, 2*np.pi, LEN))
x = list(range(len(y1)))
conv_vect = [0.1, .1, .1, .1, .1]
conv_len = len(conv_vect)
conv_vect_x = np.array(list(range(conv_len)))
y2 = np.convolve(y1, conv_vect)

plt.ion() # Enable interactive plotting so we can update the figure without blocking the code
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
line1, = ax.plot(x, y1, 'b')
line2, = ax.plot(x[:conv_len], y2[:conv_len], 'r')
line3, = ax.plot(conv_vect_x, conv_vect, 'g')
verticalLine1, = ax.plot([conv_vect_x[0], conv_vect_x[0]] , [-1, 1], 'k')
verticalLine2, = ax.plot([conv_vect_x[-1], conv_vect_x[-1]] , [-1, 1], 'k')

for i in range(LEN - conv_len):
    line2.set_xdata(x[:i+conv_len],)
    line2.set_ydata(y2[:i+conv_len],)
    conv_vect_x_updated = conv_vect_x+i
    line3.set_xdata(conv_vect_x_updated,)
    verticalLine1.set_xdata([conv_vect_x_updated[0], conv_vect_x_updated[0]],)
    verticalLine2.set_xdata([conv_vect_x_updated[-1], conv_vect_x_updated[-1]],)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.2)


plt.ioff()
plt.show()

plt.show()