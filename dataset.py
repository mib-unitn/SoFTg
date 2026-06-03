import numpy as np
import random
import torchvision.transforms as Ttorch
import torch
from glob import glob
import cv2
from scipy.special import softmax
from scipy.ndimage import rotate
import time
from scipy.ndimage import gaussian_filter
import PIL
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _intensity_augment(
    img: torch.Tensor,
    brightness_range=(0.8, 1.2),
    gamma_range=(0.75, 1.35),
    noise_std_range=(0.005, 0.02),
    p_brightness: float = 0.7,
    p_gamma: float = 0.5,
    p_noise: float = 0.5,
) -> torch.Tensor:
    """
    Random photometric augmentation for single-channel solar intensity images.
    Each operation is applied independently. Input expected in [0, 1] range.
    """
    if random.random() < p_brightness:
        img = img * random.uniform(*brightness_range)
    if random.random() < p_gamma:
        img = img.clamp(min=1e-7).pow(random.uniform(*gamma_range))
    if random.random() < p_noise:
        img = img + torch.randn_like(img) * random.uniform(*noise_std_range)
    return img.clamp(0.0, 1.0)


def warp(x, flo):
    """Elastic warping deformation using flow grids."""
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flo = flo.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    mask = torch.ones(x.size(), device=x.device)
    mask = F.grid_sample(mask, vgrid, align_corners=True)

    mask[mask <0.9999] = 0
    mask[mask >0] = 1

    return output*mask


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return Ttorch.functional.rotate(x, angle)


class SRS_crop(torch.nn.Module):
    """Cropping with probability map weighting (Spatial Random Sampling)."""
    def __init__(self, size):
        super().__init__()
        self.size = int(size)

    def forward(self, img, pmap, ind):
        counter = np.arange(len(pmap))
        pos = np.random.choice(counter, 1, p=pmap)      
        c = (int(ind[pos[0],0])+int(self.size/2), int(ind[pos[0],1])+int(self.size/2))
        img_raw = img.cpu().detach().numpy()

        x = int(c[0] - self.size/2)
        y = int(c[1] - self.size/2)

        cr_image_0 = img_raw[0, y:y+self.size, x:x+self.size]  # raw image
        cr_image_1 = img_raw[1, y:y+self.size, x:x+self.size]  # raw mask

        return torch.Tensor(np.array([cr_image_0, cr_image_1]), device='cpu'), c


class Secuential_trasn(torch.nn.Module):
    """Generates a sequential transformation chain."""
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, img, pmap, ind):
        t_list = [img]      
        for t in range(len(self.transforms)):
            if t == 1:
                rotation, c = self.transforms[t](t_list[-1], pmap, ind)
                t_list.append(rotation)
            else:
                t_list.append(self.transforms[t](t_list[-1]))

        return t_list[-1], c


class segDataset(torch.utils.data.Dataset):
    """Solar granule training dataset loader with online augmentations (warping, cropping, flips)."""
    def __init__(self, root, l=1000, s=128, augment: bool = True):
        super(segDataset, self).__init__()
        self.root = root
        self.size = s
        self.l = l
        self.classes = {'Intergranular lane': 0, 'Granule': 1}
        self.bin_classes = ['Intergranular lane', 'Granule']

        self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                                SRS_crop(self.size),
                                                RotationTransform(angles=[0, 90, 180, 270]),
                                                Ttorch.RandomPerspective(0.3, p=0.5, interpolation=Ttorch.InterpolationMode.NEAREST),
                                                Ttorch.RandomHorizontalFlip(p=0.5),
                                                Ttorch.RandomVerticalFlip(p=0.5)
                                                ])
        self.augment = augment
        self._erasing = Ttorch.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.0)
        
        self.file_list = sorted(glob(self.root+'*.npz'))
        
        print(f"📂 Found {len(self.file_list)} images in {self.root}:")
        for f in self.file_list:
            print(f"   └── {f.split('/')[-1]}")

        print("Reading images...")
        self.smap = []
        self.mask_smap = []
        self.weight_maps = []
        self.index_list = []
        self.rot_angle = np.arange(0, 90, 5)
        n_augments = len(self.rot_angle)
        total_steps = len(self.file_list) * n_augments

        with tqdm(
            total=total_steps,
            desc="  ⚙  Augmenting train set",
            unit="aug",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour="cyan",
        ) as pbar:
            for f in self.file_list:
                short_name = f.split('/')[-1]
                pbar.set_postfix(file=short_name[:30])
                file = np.load(f)
                psmap = file['smap'].astype(np.float32)
                pmsmap = file['cmask_map'].astype(np.float32)
                psmap = psmap/psmap.max()

                pad_value = int(((np.sqrt(2*(psmap.shape[0]**2))-psmap.shape[0]))/2)

                # Padding for rotation
                pad_psmap = np.pad(psmap, ((pad_value, pad_value), (pad_value, pad_value)), mode='reflect')
                pad_pmsmap = np.pad(pmsmap, ((pad_value, pad_value), (pad_value, pad_value)), mode='reflect')

                for a in self.rot_angle:
                    # Continuum image
                    vis1 = PIL.Image.fromarray(pad_psmap)
                    p_map1 = rotate(vis1, a)
                    x01 = int(abs(p_map1.shape[0]/2) - (psmap.shape[0]/2))
                    x02 = int(abs(p_map1.shape[0]/2) + (psmap.shape[0]/2))
                    p_map1 = p_map1[x01:x02, x01:x02]
                    
                    # Mask image
                    vis2 = PIL.Image.fromarray(pad_pmsmap)
                    p_map2 = np.asarray(vis2.rotate(a))
                    x11 = int(abs(p_map2.shape[0]/2) - (pmsmap.shape[0]/2))
                    x12 = int(abs(p_map2.shape[0]/2) + (pmsmap.shape[0]/2))
                    p_map_mask = p_map2[x11:x12, x11:x12]
                    
                    # Deformation parameter
                    nx, ny = p_map1.shape
                    flo = 100 * np.random.randn(2, nx, ny)
                    flo[0, :, :] = gaussian_filter(flo[0, :, :], sigma=10)
                    flo[1, :, :] = gaussian_filter(flo[1, :, :], sigma=10)
                    flo = torch.tensor(flo.astype('float32'))[None, :, :, :]

                    # Deforming Continuum image
                    tmp1 = p_map1[None, None, :, :]
                    tmp1 = torch.tensor(tmp1.astype('float32'))
                    p_map_res = warp(tmp1, flo)

                    # Deforming mask image
                    u_lables = np.unique(p_map_mask).astype(int)
                    tmp2 = np.array([(p_map_mask == c).astype(int) for c in u_lables])
                    tmp3 = []
                    for im in tmp2:
                        t_im = im[None, None, :, :]
                        t_im = torch.tensor(t_im.astype('float32'))
                        t_res = warp(t_im, flo)
                        tmp3.append(torch.round(t_res))

                    p_map_mask_res = torch.stack([tmp3[i] * int(u_lables[i]) for i in range(len(u_lables))]).sum(dim=0)

                    p_map_res = p_map_res.cpu().detach().numpy()
                    p_map_mask_res = p_map_mask_res.cpu().detach().numpy()

                    self.smap.append(p_map_res[0, 0, :, :])
                    self.mask_smap.append(p_map_mask_res[0, 0, :, :])

                    weight_maps = np.ones_like(p_map_mask_res[0, 0, int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)]).astype(np.float32)
                    wm_blurred = gaussian_filter(weight_maps, sigma=14)

                    self.weight_maps.append(softmax(wm_blurred.flatten()))
                    self.index_list.append(np.array(list(np.ndindex(weight_maps.shape))))
                    pbar.update(1)

        print(f"  ✅ Train dataset ready — {len(self.smap)} augmented maps from {len(self.file_list)} file(s).")
            
    def __getitem__(self, idx):
        ind = np.random.randint(low=0, high=len(self.smap))
        smap = self.smap[ind]
        mask_smap = self.mask_smap[ind]

        # Full probability maps calculation
        weight_map = self.weight_maps[ind]
        index_l = self.index_list[ind]
        img_t, c = self.transform_serie(np.array([smap, mask_smap]).transpose(), weight_map, index_l)

        self.image = img_t[0].unsqueeze(0)
        self.mask = (img_t[1] > 0).type(torch.int64)  # binarize: granule class → 1, lane → 0

        if self.augment:
            self.image = _intensity_augment(self.image)
            self.image = self._erasing(self.image)

        return self.image, self.mask
      
    def __len__(self):
        return self.l


class segDataset_val(torch.utils.data.Dataset):
    """Solar granule validation dataset loader."""
    def __init__(self, root, l=1000, s=128, augment: bool = False):
        super(segDataset_val, self).__init__()
        self.augment = augment
        self.root = root
        self.size = s
        self.l = l
        self.classes = {'Intergranular lane': 0, 'Granule': 1}
        self.bin_classes = ['Intergranular lane', 'Granule']

        # Validation uses a deterministic pipeline: crop only, no augmentations.
        self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                                SRS_crop(self.size),
                                                ])
        
        self.file_list = sorted(glob(self.root+'*.npz'))

        print(f"📂 Found {len(self.file_list)} images in {self.root}:")
        for f in self.file_list:
            print(f"   └── {f.split('/')[-1]}")

        print("Reading images...")
        self.smap = []
        self.mask_smap = []
        self.weight_maps = []
        self.index_list = []

        with tqdm(
            self.file_list,
            desc="  ⚙  Loading val set  ",
            unit="file",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="green",
        ) as pbar:
            for f in pbar:
                short_name = f.split('/')[-1]
                pbar.set_postfix(file=short_name[:35])
                file = np.load(f)
                psmap = file['smap'].astype(np.float32)
                pmsmap = file['cmask_map'].astype(np.float32)
                psmap = psmap/psmap.max()

                self.smap.append(psmap)
                self.mask_smap.append(pmsmap)

                weight_maps = np.ones_like(pmsmap[int(self.size/2):-int(self.size/2), int(self.size/2):-int(self.size/2)]).astype(np.float32)
                wm_blurred = gaussian_filter(weight_maps, sigma=14)

                self.weight_maps.append(softmax(wm_blurred.flatten()))
                self.index_list.append(np.array(list(np.ndindex(weight_maps.shape))))

        print(f"  ✅ Val dataset ready  — {len(self.smap)} map(s) from {len(self.file_list)} file(s).")
            
    def __getitem__(self, idx):
        ind = np.random.randint(low=0, high=len(self.smap))
        smap = self.smap[ind]
        mask_smap = self.mask_smap[ind]

        # Full probability maps calculation
        weight_map = self.weight_maps[ind]
        index_l = self.index_list[ind]
        img_t, c = self.transform_serie(np.array([smap, mask_smap]).transpose(), weight_map, index_l)

        self.image = img_t[0].unsqueeze(0)
        self.mask = (img_t[1] > 0).type(torch.int64)  # binarize: granule class → 1, lane → 0

        return self.image, self.mask
      
    def __len__(self):
        return self.l
