import matplotlib.pyplot as plt

def create_plot(n):
    figure = plt.figure(figsize=(6, 6))
    ax = figure.add_subplot()
    ax.set_autoscaley_on(True)
    ax.set_xlim(-1, n + 1)
    ax.set_ylim(-1, n + 1)
    return ax

def plotter(ax, v):
    plt.cla()
    ax.axis('off')
    ax.set_autoscaley_on(True)
    plt.matshow(v, fignum=0)
    plt.draw()
    plt.show()
    plt.pause(0.1)
