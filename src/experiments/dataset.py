import os

from matplotlib import pyplot as plt

from datasets.lin import LINNearestDataset, LINDataset
from utils.command_line_logger import CommandLineLogger
from utils.filesystems.local import LocalCapsule, LocalFolder

datasets = {}
datasets_nn = {}
datasets_nnr = {}


def plot_nns():
    alp14_ds_nn = datasets_nn['Alp14']
    plt.subplots(1, 7, figsize=(15, 2.4))
    plt.subplot(1, 7, 1)
    plt.imshow(alp14_ds_nn[0][0].reshape(48, 80), cmap="Reds")
    plt.title('Alp14')
    plt.axis('off')
    for i, class_name in zip(range(1, 7), LINDataset.Classes):
        plt.subplot(1, 7, i + 1)
        plt.imshow(alp14_ds_nn[0][i].reshape(48, 80), cmap="Greens")
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(list(alp14_ds_nn.nearest_neighbors_info.keys())[0])
    plt.show()


def plot_nns_r():
    alp14_ds_nnr = datasets_nnr['Alp14']
    plt.subplots(1, 7, figsize=(15, 2.4))
    plt.subplot(1, 7, 1)
    plt.imshow(alp14_ds_nnr[0][0].reshape(48, 80), cmap="Reds")
    plt.title('Alp14')
    plt.axis('off')
    for i, class_name in zip(range(0, 6), LINDataset.Classes):
        plt.subplot(1, 7, i + 2)
        plt.imshow(alp14_ds_nnr[0][i].reshape(48, 80), cmap="Reds")
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(list(alp14_ds_nnr.nearest_neighbors_info.keys())[0])
    plt.show()


def plot_nns_gr(which_class: str = 'Alp14'):
    fig, _ = plt.subplots(2, 6, figsize=(15, 5))
    plt.tight_layout()
    plt.subplot(2, 6, 1)
    # Plot red channels
    img_ds_nnr = datasets_nnr[which_class]
    for i, class_name in zip(range(0, 6), LINDataset.Classes):
        ax = plt.subplot(2, 6, i + 1)
        plt.imshow(img_ds_nnr[0][i].reshape(48, 80), cmap="Reds")
        plt.axis('off')
        # Add red rectangle around selected images
        if class_name == which_class:
            bbox = ax.get_tightbbox(fig.canvas.get_renderer())
            x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
            fig.add_artist(
                plt.Rectangle((x0, y0), width, height, edgecolor='red', linestyle='--', linewidth=1, fill=False)
            )
        plt.title(class_name)
    # Plot green channels
    img_ds_nn = datasets_nn[which_class]
    for i, class_name in zip(range(1, 7), LINDataset.Classes):
        ax = plt.subplot(2, 6, 6 + i)
        plt.imshow(img_ds_nn[0][i].reshape(48, 80), cmap="Greens")
        plt.axis('off')
        # Add red rectangle around selected images
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
        fig.add_artist(
            plt.Rectangle((x0, y0), width, height, edgecolor='red', linestyle='--', linewidth=1, fill=False)
        )
    plt.suptitle(list(img_ds_nnr.nearest_neighbors_info.keys())[0])
    plt.show()


def plot_images():
    alp14_ds = datasets['Alp14']
    alp14_ds_0 = alp14_ds[0]
    plt.imshow(alp14_ds_0[0].reshape(48, 80), cmap="Reds")
    plt.show()
    plt.imshow(alp14_ds_0[1].reshape(48, 80), cmap="Greens")
    plt.show()


if __name__ == '__main__':
    # Initialize Google Drive stuff
    _local_gdrive_root = '/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.gdrive_personal'
    _capsule = LocalCapsule(_local_gdrive_root)
    _groot = LocalFolder.root(capsule_or_fs=_capsule).subfolder_by_name('Datasets')
    # Initialize Datasets
    _logger = CommandLineLogger(log_level=os.getenv('LOG_LEVEL', 'info'), name='main')
    for _class in LINDataset.Classes:
        datasets[_class] = LINDataset(dataset_fs_folder_or_root=_groot, which_classes=_class, logger=_logger,
                                      return_path=False)
        datasets_nn[_class] = LINNearestDataset(dataset_fs_folder_or_root=_groot, which_classes=_class, logger=_logger)
        datasets_nnr[_class] = LINNearestDataset(dataset_fs_folder_or_root=_groot, which_classes=_class, logger=_logger,
                                                 reds_only=True)
    # Plot random images
    plot_images()
    # Plot k-NNs
    plot_nns()
    plot_nns_r()
    plot_nns_gr('Arp3')
