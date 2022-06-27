import os

import torch
from matplotlib import pyplot as plt

from datasets.lin import LINNearestDataset, LINDataset
from utils.command_line_logger import CommandLineLogger
from utils.filesystems.local import LocalCapsule, LocalFolder
from utils.pytorch import invert_transforms

plt.rcParams["font.family"] = 'JetBrains Mono'

datasets = {}
datasets_nn = {}
datasets_nnr = {}


def plot_nns():
    alp14_ds_nn = datasets_nn['Alp14']
    plt.subplots(1, 7, figsize=(15, 2.4))
    plt.subplot(1, 7, 1)
    it = invert_transforms(alp14_ds_nn.transforms)
    red_channel = it(alp14_ds_nn[0])[0].reshape(48, 80)
    red_channel = torch.concat((red_channel.unsqueeze(0),
                                torch.zeros_like(red_channel).unsqueeze(0),
                                torch.zeros_like(red_channel).unsqueeze(0)), dim=0)
    plt.imshow(red_channel.permute(1, 2, 0))
    # plt.imshow(alp14_ds_nn[0][0].reshape(48, 80), cmap="Reds")
    plt.title('Alp14')
    plt.axis('off')
    for i, class_name in zip(range(1, 7), LINDataset.Classes):
        plt.subplot(1, 7, i + 1)
        green_channel = it(alp14_ds_nn[0])[i].reshape(48, 80)
        green_channel = torch.concat((torch.zeros_like(green_channel).unsqueeze(0),
                                      green_channel.unsqueeze(0),
                                      torch.zeros_like(green_channel).unsqueeze(0)), dim=0)
        plt.imshow(green_channel.permute(1, 2, 0))
        # plt.imshow(alp14_ds_nn[0][i].reshape(48, 80), cmap="Greens")
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(list(alp14_ds_nn.nearest_neighbors_info.keys())[0])
    plt.show()


def plot_nns_r():
    alp14_ds_nnr = datasets_nnr['Alp14']
    plt.subplots(1, 7, figsize=(15, 2.4))
    plt.subplot(1, 7, 1)
    it = invert_transforms(alp14_ds_nnr.transforms)
    sample = it(alp14_ds_nnr[0])
    red_channel = sample[0].reshape(48, 80)
    red_channel = torch.concat((red_channel.unsqueeze(0),
                                torch.zeros_like(red_channel).unsqueeze(0),
                                torch.zeros_like(red_channel).unsqueeze(0)), dim=0)
    plt.imshow(red_channel.permute(1, 2, 0))
    # plt.imshow(alp14_ds_nnr[0][0].reshape(48, 80), cmap="Reds")
    plt.title('Alp14')
    plt.axis('off')
    for i, class_name in zip(range(0, 6), LINDataset.Classes):
        plt.subplot(1, 7, i + 2)
        red_channel = sample[i].reshape(48, 80)
        red_channel = torch.concat((red_channel.unsqueeze(0),
                                    torch.zeros_like(red_channel).unsqueeze(0),
                                    torch.zeros_like(red_channel).unsqueeze(0)), dim=0)
        plt.imshow(red_channel.permute(1, 2, 0))
        # plt.imshow(alp14_ds_nnr[0][i].reshape(48, 80), cmap="Reds")
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
    it = invert_transforms(img_ds_nnr.transforms)
    sample = it(img_ds_nnr[0])
    for i, class_name in zip(range(0, 6), LINDataset.Classes):
        ax = plt.subplot(2, 6, i + 1)
        red_channel = sample[i].reshape(48, 80)
        red_channel = torch.concat((red_channel.unsqueeze(0),
                                    torch.zeros_like(red_channel).unsqueeze(0),
                                    torch.zeros_like(red_channel).unsqueeze(0)), dim=0)
        plt.imshow(red_channel.permute(1, 2, 0))
        # plt.imshow(img_ds_nnr[0][i].reshape(48, 80), cmap="Reds")
        plt.axis('off')
        # Add red rectangle around selected images
        if class_name == which_class:
            bbox = ax.get_tightbbox(fig.canvas.get_renderer())
            x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
            fig.add_artist(
                plt.Rectangle((x0, y0), width, height, edgecolor='red', linestyle='--', linewidth=2, fill=False)
            )
        plt.title(class_name)
    # Plot green channels
    img_ds_nn = datasets_nn[which_class]
    it = invert_transforms(img_ds_nn.transforms)
    sample = it(img_ds_nn[0])
    for i, class_name in zip(range(1, 7), LINDataset.Classes):
        ax = plt.subplot(2, 6, 6 + i)

        green_channel = sample[i].reshape(48, 80)
        green_channel = torch.concat((torch.zeros_like(green_channel).unsqueeze(0),
                                      green_channel.unsqueeze(0),
                                      torch.zeros_like(green_channel).unsqueeze(0)), dim=0)
        plt.imshow(green_channel.permute(1, 2, 0))

        # plt.imshow(img_ds_nn[0][i].reshape(48, 80), cmap="Greens")
        plt.axis('off')
        # Add red rectangle around selected images
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
        fig.add_artist(
            plt.Rectangle((x0, y0), width, height, edgecolor='red', linestyle='--', linewidth=2, fill=False)
        )
    plt.suptitle(list(img_ds_nnr.nearest_neighbors_info.keys())[0])
    plt.savefig(f'dataset-2-{which_class.lower()}.pdf')
    plt.show()


def plot_images(n_samples_per_class: int = 1):
    # if os.path.exists('dataset-1.pdf'):
    #     subprocess.call(('xdg-open', 'dataset-1.pdf'))
    #     return
    rows = 3 * n_samples_per_class
    fig, _ = plt.subplots(rows, 6, figsize=(15, 5 * n_samples_per_class))
    for sample_i in range(n_samples_per_class):
        for i, class_name in zip(range(1, 7), LINDataset.Classes):
            # 1st row
            plt.subplot(rows, 6, i + sample_i * 18)
            ds = datasets[class_name]
            it = invert_transforms(ds.transforms)
            ds_0 = it(ds[sample_i])
            rgb_img = torch.concat((ds_0, torch.zeros((1, 48, 80))), dim=0)
            plt.imshow(rgb_img.permute(1, 2, 0))
            if sample_i == 0:
                plt.title(class_name)
            plt.axis('off')
            plt.tight_layout()
            # 2nd row
            plt.subplot(rows, 6, 6 + i + sample_i * 18)
            red_channel = torch.concat((ds_0[0].unsqueeze(0),
                                        torch.zeros_like(ds_0[0]).unsqueeze(0),
                                        torch.zeros_like(ds_0[0]).unsqueeze(0)), dim=0)
            plt.imshow(red_channel.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
            # 3rd row
            plt.subplot(rows, 6, 12 + i + sample_i * 18)
            green_channel = torch.concat((torch.zeros_like(ds_0[0]).unsqueeze(0),
                                          ds_0[1].unsqueeze(0),
                                          torch.zeros_like(ds_0[0]).unsqueeze(0)), dim=0)
            plt.imshow(green_channel.permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()
    plt.tight_layout()
    plt.suptitle('First 3 Images per Training Class Dataset')
    plt.savefig('dataset-1.pdf')
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
    # # Plot random images
    # plot_images(n_samples_per_class=3)
    # Plot k-NNs
    plot_nns()
    # plot_nns_r()
    plot_nns_gr('Alp14')
    plot_nns_gr('Arp3')
