import atexit
import io
import json
import os
import shutil
from typing import Optional, List

import matplotlib
import matplotlib.font_manager
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torchvision.transforms import transforms

from utils.filesystems.gdrive.remote import GDriveFolder


def create_img_grid(images: torch.Tensor,
                    ncols: Optional[int] = None,
                    nrows: Optional[int] = None, border: int = 2,
                    black: float = 0.5,
                    white_border_right: bool = False,
                    gen_transforms: Optional[transforms.Compose] = None,
                    invert: bool = False) -> torch.Tensor:
    """
    :param (torch.Tensor) images: torch.Tensor object of shape N x (CxHxW) (similar to training tensors)
    :param (int) ncols:
    :param (int) nrows:
    :param (int) border:
    :param (int) black:
    :param (optional) gen_transforms:
    :return:
    """
    # Inverse generator transforms
    if gen_transforms is not None:
        from utils.pytorch import invert_transforms
        gen_transforms_inv = invert_transforms(gen_transforms)
    else:
        from utils.pytorch import ToTensorOrPass
        gen_transforms_inv = ToTensorOrPass()

    # Check nrows
    # if nrows is None:
    #     nrows = int(images.shape[0] / ncols)
    # assert nrows * ncols == images.shape[0], 'nrows * ncols must be equal to the total number of images'

    for img_i in range(len(images)):
        images[img_i] = gen_transforms_inv(images[img_i])

    # Split image to channels
    images_c = []
    for img in images:
        images_c.append(img[0])  # red
    for j in range(1, images.shape[1]):
        for img in images:
            images_c.append(img[j])  # green
    nrows = 2 if nrows is None else nrows
    ncols = len(images) if ncols is None else ncols

    # Create a single (grouped) image for each row
    row_images = []
    for r in range(nrows):
        _rlist = []
        _rheight = None
        for c in range(ncols):
            # Apply inverse image transforms to given images
            image = images_c[(r * ncols + c) if not invert else (c * nrows + r)].float()
            if _rheight is None:
                _rheight = image.shape[0]

            tr, tc = (r, c) if not invert else (c, r)

            if c == 0:
                _rlist.append(black * torch.ones(3, _rheight, border).float())  # |
            # Real red channel
            if tr == 0:
                image = torch.concat((image.unsqueeze(0),
                                      torch.zeros_like(image).unsqueeze(0),
                                      torch.zeros_like(image).unsqueeze(0)), dim=0)
            else:
                image = torch.concat((torch.zeros_like(image).unsqueeze(0),
                                      image.unsqueeze(0),
                                      torch.zeros_like(image).unsqueeze(0)), dim=0)
            _rlist.append(image)  # |□
            if c == 0:
                _rlist.append(black * torch.ones(3, _rheight, border).float())  # |□|□...
        _rlist.append(black * torch.ones(3, _rheight, border).float())  # |□|□...□|
        if white_border_right:
            _rlist.append(torch.ones(3, _rheight, 4 * border).float())  # |
        row_images.append(torch.cat(_rlist, dim=2).cpu())

    # Join row-images to form the final image
    _list = []
    for rii, ri in enumerate(row_images):
        vb = black * torch.ones(3, border, ri.shape[2]).float()
        if white_border_right:
            vb[:, :, -4 * border:] = 1.0
        _list.append(vb)  # ___
        _list.append(ri)  # |□|
        _list.append(vb)  # ---
        if nrows == 3 and rii == 1:
            pass
        else:
            _list.append(1.0 * torch.ones(3, 4 * border, ri.shape[2]).float())  # (gap)
    return torch.cat(_list[:-1], dim=1).cpu()


def create_img_grid_6class(x0: list, g0: list, g1: list, border: int = 2, black: float = 0.5) -> torch.Tensor:
    grids = []
    for class_idx in range(6):
        grids.append(create_img_grid(images=torch.stack([
            x0[class_idx], g0[class_idx], g1[class_idx]
        ]), gen_transforms=None, border=border, black=black, white_border_right=class_idx < 5))
    return torch.cat(grids, dim=2)


def unpack_multichannel(inp: torch.Tensor) -> List[torch.Tensor]:
    out = []
    for g in range(1, 7):
        out.append(inp[[0, g]].detach().clone())
    return out


def create_img_grid_6class_joint(x0, g0, g1, border=2, black=0.5):
    return create_img_grid(images=torch.stack([x0, g0, g1]), gen_transforms=None, border=border, black=black,
                           nrows=3, ncols=7, invert=True)


def ensure_matplotlib_fonts_exist(groot: GDriveFolder, force_rebuild: bool = False) -> bool:
    """
    Downloads all TTF files from Google Drive's "Fonts" folder and places them in the directory `matplotlib` expects to
    find its .ttf font files.
    :param (GDriveFolder) groot: the parent of "Fonts" folder in Google Drive
    :param (bool) force_rebuild: set to True to forcefully rebuild fonts cache in matplotlib
    :return: a `bool` object set to `True` if fonts rebuilding was performed, `False` otherwise
    """
    # Get fonts gfolder
    fonts_gfolder = groot if 'Fonts' == groot.name else \
        groot.subfolder_by_name('Fonts')
    # Download all fonts from Google Drive
    fonts_gfolder.download(recursive=True, in_parallel=False, show_progress=True, unzip_after=False)
    # Define the matplotlib ttf font dir (destination dir)
    matplotlib_ttf_path = matplotlib.matplotlib_fname().replace('matplotlibrc', 'fonts/ttf')
    assert os.path.exists(matplotlib_ttf_path) and os.path.isdir(matplotlib_ttf_path)
    # Visit all subfolders and copy .ttf files to matplotlib fonts dir
    new_ttf_files = []
    for sf in fonts_gfolder.subfolders:
        sf_fonts_folder = f'/usr/share/fonts/truetype/{sf.name.replace(" ", "").lower()}'

        # Copy only JetBrains Mono font
        if not sf_fonts_folder.endswith('jetbrainsmono'):
            continue
        os.system(f'mkdir -p {sf_fonts_folder}')
        for f in sf.files:
            if not f.name.endswith('ttf'):
                continue
            # Copy file to matplotlib folder
            if not os.path.exists(f'{matplotlib_ttf_path}/{f.name}'):
                new_ttf_files.append(shutil.copy(f.path, matplotlib_ttf_path))
            # Copy to system fonts folder
            if not os.path.exists(f'{sf_fonts_folder}/{f.name}'):
                shutil.copy(f.path, sf_fonts_folder)
    # Inform and rebuild fonts cache
    rebuild = force_rebuild
    if len(new_ttf_files) > 0:
        print('Font files copied:')
        print(json.dumps(new_ttf_files, indent=4))
        rebuild = True
    if rebuild:
        # Rebuild system font cache
        os.system('apt install fontconfig -y')
        os.system('fc-cache -fv')
        # Rebuild matplotlib font cache
        os.system('rm ~/.cache/matplotlib -rf')
        os.system('mkdir -p ~/.cache/matplotlib')
        try:
            # noinspection PyProtectedMember,PyUnresolvedReferences
            matplotlib.font_manager._rebuild()
        except AttributeError as e:
            print('[ensure_matplotlib_fonts_exist]: ' + str(e))
            os.kill(os.getpid(), 9)
    return rebuild


def plot_grid(grid: torch.Tensor or np.ndarray, figsize=tuple, footnote_l: Optional[str] = None,
              footnote_r: Optional[str] = None, save_path: str or None = None) -> Image:
    """
    Plots an image grid (created with `utils.plot.create_img_grid`)
    :param (torch.Tensor or numpy.ndarray) grid: a torch.Tensor or NumPy array object holding the image grid
    :param (tuple) figsize: plt.figure's `figsize` parameter as a tuple object (width, height). A wise choice could be
                            (ncols, nrows) as those were given to create_img_grid function.
    :param (optional) footnote_l: left-aligned footnote (is printed at the bottom of the plot)
    :param (optional) footnote_r: right-aligned footnote (is printed at the bottom of the plot -
                                  independent positioning of left footnote)
    :return: a `PIL.Image.Image` instance capturing the currently-shown matplotlib figure
    """
    # Set matplotlib params
    matplotlib.rcParams["font.family"] = 'JetBrains Mono'
    # Create a new figure
    plt.figure(figsize=figsize, dpi=300, frameon=False, clear=True)
    # Remove annoying axes
    plt.axis('off')
    # Create image and return
    if footnote_l:
        plt.suptitle(footnote_l, y=0.03, fontsize=4, fontweight='light', horizontalalignment='left', x=0.001)
    plt.imshow(grid.permute(1, 2, 0) if isinstance(grid, torch.Tensor) else np.transpose(grid, (1, 2, 0)))
    plt.tight_layout()
    # Show the right footnote
    fig: matplotlib.figure.Figure
    fig = plt.gcf()
    if footnote_r:
        fig.text(x=1, y=0.01, s=footnote_r, fontsize=4, fontweight='light', horizontalalignment='right')
    if save_path is not None:
        plt.savefig(save_path)
    return pltfig_to_pil(fig)


def pltfig_to_pil(figure: Figure) -> Image:
    """
    Convert a matplotlib figure (e.g. from plt.cfg()) to a PIL image.
    :param (Figure) figure: a `matplotlib.figure.Figure` instance with the image we want to convert to PIL
    :return: a `PIL.Image` object
    """
    # Create a buffer to read the figure data
    _temp_buffer = io.BytesIO()
    atexit.register(_temp_buffer.close)
    # Save figure in buffer
    figure.savefig(_temp_buffer, format='jpg')
    # Read image from buffer and return
    _temp_buffer.seek(0)
    return Image.open(_temp_buffer)
