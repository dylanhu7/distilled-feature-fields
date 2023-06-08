from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .colmap_sam import ColmapSamDataset
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'colmap_sam': ColmapSamDataset,
                'nerfpp': NeRFPPDataset,
                'rtmv': RTMVDataset}