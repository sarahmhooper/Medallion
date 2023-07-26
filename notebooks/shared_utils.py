import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
import os
from tqdm import tqdm


# Create custom colormaps
ncolors = 256
color_array = plt.get_cmap("gist_rainbow")(range(ncolors))
color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
map_object = LinearSegmentedColormap.from_list(name="rainbow_alpha", colors=color_array)
plt.register_cmap(cmap=map_object)

color_array = plt.get_cmap("plasma")(range(ncolors))
color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
map_object = LinearSegmentedColormap.from_list(name="plasma_alpha", colors=color_array)
plt.register_cmap(cmap=map_object)

color_array = plt.get_cmap("Reds")(range(ncolors))
color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
color_array[:, 0] = 17 / 256.0
color_array[:, 1] = 71 / 256.0
color_array[:, 2] = 161 / 256.0
map_object = LinearSegmentedColormap.from_list(name="cm1", colors=color_array)
plt.register_cmap(cmap=map_object)

color_array[:, 0] = 0 / 256.0
color_array[:, 1] = 255 / 256.0
color_array[:, 2] = 255 / 256.0
map_object = LinearSegmentedColormap.from_list(name="cm2", colors=color_array)
plt.register_cmap(cmap=map_object)

color_array[:, 0] = 255 / 256.0
color_array[:, 1] = 131 / 256.0
color_array[:, 2] = 0 / 256.0
map_object = LinearSegmentedColormap.from_list(name="cm3", colors=color_array)
plt.register_cmap(cmap=map_object)



def plot_classes(
    img, tp, fp, fn, size=10, alpha=0.5, title=None, contour=False, x=None, y=None
):
    # Visualize all slices of image with prediction/ground truth comparison overlaid
    n_rows = int(np.ceil(img.shape[-1] / 4.0))
    n_cols = min(img.shape[-1], 4)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(size, size))
    if title != None:
        fig.suptitle(title, fontsize=24)

    if n_rows == 1 and n_cols == 1:
        for mysl in range(img.shape[-1]):
            axs.imshow(
                img[:, :, mysl], cmap="gray", vmin=np.min(img), vmax=np.max(img)
            )
            axs.axis("off")
            if np.sum(tp+fp+fn)>0:
                axs.imshow(tp[:, :, mysl], alpha=alpha, cmap="cm1", vmin=0, vmax=1)
                axs.imshow(fp[:, :, mysl], alpha=alpha, cmap="cm2", vmin=0, vmax=1)
                axs.imshow(fn[:, :, mysl], alpha=alpha, cmap="cm3", vmin=0, vmax=1)
    else:
        axs = axs.ravel()
        for mysl in range(img.shape[-1]):
            axs[mysl].imshow(
                img[:, :, mysl], cmap="gray", vmin=np.min(img), vmax=np.max(img)
            )
            axs[mysl].axis("off")
            if np.sum(tp+fp+fn)>0:
                axs[mysl].imshow(tp[:, :, mysl], alpha=alpha, cmap="cm1", vmin=0, vmax=1)
                axs[mysl].imshow(fp[:, :, mysl], alpha=alpha, cmap="cm2", vmin=0, vmax=1)
                axs[mysl].imshow(fn[:, :, mysl], alpha=alpha, cmap="cm3", vmin=0, vmax=1)

    all_patches = []
    all_keys = ["TP", "FP", "FN"]
    all_colors = [(17 / 256.0, 17 / 256.0, 161 / 256.0), (0, 1, 1), (1,131/256.,0)]
    for k, c in zip(all_keys, all_colors):
        all_patches += [mpatches.Patch(color=c, label=k)]
    if n_rows == 1 and n_cols == 1:
        lgnd = axs.legend(handles=all_patches, loc="upper left", bbox_to_anchor=(1.05, 1))
    else:
        for aind in range(len(axs)):
            if aind%4==3:
                lgnd = axs[aind].legend(handles=all_patches, loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.show()
    return

def plot3d(
    img1,
    img2=np.asarray([]),
    img3=np.asarray([]),
    size=10,
    alpha=0.3,
    title=None,
    contour=False,
    x=None,
    y=None,
):
    """Visualize a 3d image, optionally with a 2nd 3d image overlaid"""
    # Visualize all slices of contrast enhanced image
    n_rows = int(np.ceil(img1.shape[-1] / 4.0))
    n_cols = min(img1.shape[-1], 4)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(size, size))
    if title != None:
        fig.suptitle(title)

    if n_rows == 1 and n_cols == 1:
        axs.imshow(np.squeeze(img1), cmap="gray", vmin=np.min(img1), vmax=np.max(img1))
        axs.axis("off")
        if img2.size != 0:
            if contour:
                axs.contour(np.squeeze(img2), alpha=alpha, cmap="plasma", linewidths=3)
            else:
                axs.imshow(
                    np.squeeze(img2),
                    alpha=alpha,
                    cmap="rainbow_alpha",
                    vmin=np.min(img2),
                    vmax=np.max(img2),
                )
        if img3.size != 0:
            axs.imshow(
                np.squeeze(img3),
                alpha=alpha,
                cmap="plasma_alpha",
                vmin=np.min(img3),
                vmax=np.max(img3),
            )
        if x != None and y != None:
            axs.plot(y, x, "rx")
    else:
        axs = axs.ravel()
        for mysl in range(img1.shape[-1]):
            axs[mysl].imshow(
                img1[:, :, mysl], cmap="gray", vmin=np.min(img1), vmax=np.max(img1)
            )
            axs[mysl].axis("off")
            if img2.size != 0:
                if contour:
                    axs[mysl].contour(
                        img2[:, :, mysl], alpha=alpha, cmap="plasma", linewidths=3
                    )
                else:
                    axs[mysl].imshow(
                        img2[:, :, mysl],
                        alpha=alpha,
                        cmap="rainbow_alpha",
                        vmin=np.min(img2),
                        vmax=np.max(img2),
                    )
            if img3.size != 0:
                axs[mysl].imshow(
                    img3[:, :, mysl],
                    alpha=alpha,
                    cmap="plasma_alpha",
                    vmin=np.min(img3),
                    vmax=np.max(img3),
                )
            if x != None and y != None:
                axs[mysl].plot(y, x, "rx")
    plt.show()
    return

def compute_dice(mask1, mask2):
    assert np.all([u in [0,1] for u in np.unique(mask1)])
    assert np.all([u in [0,1] for u in np.unique(mask2)])
    area_overlap = np.sum(np.logical_and(mask1, mask2))
    total_pix = np.sum(mask1) + np.sum(mask2)
    if total_pix==0: return 1 
    return 2.0 * area_overlap / float(total_pix)

def write_values_to_text(fp, values):
    with open(fp, "a") as writer:
        writer.write("\n" + str(values))
