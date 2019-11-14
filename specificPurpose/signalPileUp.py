import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import rc

# activate latex text rendering
#rc('text', usetex=True)

fig = plt.figure(1, figsize=[12, 4.5])
ax = plt.subplot(1, 2, 1)

x1 = np.arange(0, 1025, 25)
y1 = np.zeros(41)
y1[16] = 37

x2 = x1
y2 = np.sin(np.arange(-5, 5.25, 0.25))*5
np.random.shuffle(y2)
np.random.shuffle(y2)
np.random.shuffle(y2)
y2[22:28] = np.asanyarray([10, 15, 35, 47, 25, 5])
y2[20] = 0
y2[16] = 0

minX = 0
minY = -10
maxY = 50
maxX = 1000
proxies = []
h, = plt.plot(1, 1, '-ko')
proxies.append(h)
h, = plt.plot(1, 1, '--*', color='0.8')
proxies.append(h)
ax.legend(proxies, ['Target', 'Measured'], loc="upper left", ncol=1, shadow=True, fancybox=True)

markerline, stemlines, baseline = ax.stem(x1, y1, 'k', use_line_collection=True)
markerline2, stemlines2, baseline2 = ax.stem(x2, y2, linefmt='--', markerfmt="*", use_line_collection=True)
plt.setp(baseline, 'color', 'k', 'linewidth', 2)
plt.setp(baseline2, 'color', 'k', 'linewidth', 2)
plt.setp(markerline, 'color', 'k', 'linewidth', 2)
plt.setp(stemlines, 'color', 'k', 'linewidth', 2)
plt.setp(stemlines2, 'color', '0.8', 'linewidth', 2)
plt.setp(markerline2, 'color', '0.8', 'linewidth', 2)

plt.xlabel('Time (in ns)\n\n'+r'$\bf{(A)}$')
plt.ylabel('ADC counts')

ax.annotate('', (395, -4), (605, -4), ha="center", va="center", arrowprops=dict(arrowstyle='<|-|>', fc='k', ec='k'))
ax.annotate(r'$\tau$', (500, -5.5), ha="center", va="center")
ax.tick_params()
ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
ax.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 5))
ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 5))
ax.tick_params(axis='both', labelsize=10, which='both', direction='out', grid_linestyle='--')
ax.grid(which='major', alpha=0.7)


ay = plt.subplot(1, 2, 2)

x1 = np.arange(0, 1025, 25)
y1 = np.zeros(41)
y1[17] = 23
y1[19] = 29

x2 = x1
y2 = np.sin(np.arange(-5, 5.25, 0.25))*5
np.random.shuffle(y2)
np.random.shuffle(y2)
np.random.shuffle(y2)
y2[24:30] = np.asanyarray([8, 20, 30, 25, 48, 15])
y2[17] = 0
y2[19] = 0
y2[21] = 0
y2[23] = 0


minX = 0
minY = -10
mayY = 50
mayX = 1000
proxies = []
h, = plt.plot(1, 1, '-ko')
proxies.append(h)
h, = plt.plot(1, 1, '--*', color='0.8')
proxies.append(h)
ay.legend(proxies, ['Target', 'Measured'], loc="upper left", ncol=1, shadow=True, fancybox=True)

markerline, stemlines, baseline = ay.stem(x1, y1, 'k', use_line_collection=True)
markerline2, stemlines2, baseline2 = ay.stem(x2, y2, linefmt='--', markerfmt="*", use_line_collection=True)
plt.setp(baseline, 'color', 'k', 'linewidth', 2)
plt.setp(baseline2, 'color', 'k', 'linewidth', 2)
plt.setp(markerline, 'color', 'k', 'linewidth', 2)
plt.setp(stemlines, 'color', 'k', 'linewidth', 2)
plt.setp(stemlines2, 'color', '0.8', 'linewidth', 2)
plt.setp(markerline2, 'color', '0.8', 'linewidth', 2)

plt.xlabel('Time (in ns)\n\n'+r'$\bf{(B)}$')
plt.ylabel('ADC counts')

ay.annotate('', (470, -3), (680, -3), ha="center", va="center", arrowprops=dict(arrowstyle='<|-|>', fc='k', ec='k'))
ay.annotate(r'$\tau$', (575, -4.5), ha="center", va="center")
ay.annotate('', (420, -6), (630, -6), ha="center", va="center", arrowprops=dict(arrowstyle='<|-|>', fc='k', ec='k'))
ay.annotate(r'$\tau$', (525, -7.5), ha="center", va="center")

ay.tick_params()
ay.set_xlim(minX - (mayX / 100), mayX + (mayX / 100))
ay.set_ylim(minY - (mayY / 100), mayY + (mayY / 100))
ay.set_xticks(np.arange(minX, mayX + (mayX / 100), mayX / 5))
ay.set_yticks(np.arange(minY, mayY + (mayY / 100), mayY / 5))
ay.tick_params(axis='both', labelsize=10, which='both', direction='out', grid_linestyle='--')
ay.grid(which='major', alpha=0.7)




plt.show()