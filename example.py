# Name: Jiaming Yu
# Time:
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

nx = 150
ny = 50

fig = plt.figure()
data = np.zeros((nx, ny))
im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=1)

def init():
    im.set_data(np.zeros((nx, ny)))

def animate(i):
    xi = i // ny
    yi = i % ny
    data[xi, yi] = 1
    im.set_data(data)
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nx * ny,
                               interval=50)

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)

x = np.random.rand(15)
y = np.random.rand(15)
names = np.array(list("ABCDEFGHIJKLMNO"))
c = np.random.randint(1,5,size=15)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
sc = plt.scatter(x,y,c=c, s=100, cmap=cmap, norm=norm)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()

    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()

f"second_term0: ({second_term[0, 0]:.2f},{second_term[0, 1]:.2f}) \n" \
f"second_term1: ({second_term[1, 0]:.2f},{second_term[1, 1]:.2f}) \n" \
f"second_term2: ({second_term[2, 0]:.2f},{second_term[2, 1]:.2f}) \n" \
f"second_term3: ({second_term[3, 0]:.2f},{second_term[3, 1]:.2f}) \n" \
f"diff0: ({diff[0, 0]:.2f},{diff[0, 1]:.2f}) \n" \
f"diff1: ({diff[1, 0]:.2f},{diff[1, 1]:.2f}) \n" \
f"diff2: ({diff[2, 0]:.2f},{diff[2, 1]:.2f}) \n" \
f"diff3: ({diff[3, 0]:.2f},{diff[3, 1]:.2f}) \n" \