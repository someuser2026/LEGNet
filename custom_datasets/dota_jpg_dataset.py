import glob
import os.path as osp

from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset


@ROTATED_DATASETS.register_module()
class DOTAJpgDataset(DOTADataset):
    """DOTA dataset variant that resolves image extension dynamically."""

    def __init__(self,
                 *args,
                 img_suffixes=('.jpg', '.png', '.jpeg', '.JPG', '.PNG',
                               '.JPEG'),
                 img_suffix=None,
                 **kwargs):
        normalized = []
        for suffix in img_suffixes:
            norm = suffix if suffix.startswith('.') else f'.{suffix}'
            if norm not in normalized:
                normalized.append(norm)

        if img_suffix is not None:
            preferred = img_suffix if img_suffix.startswith('.') else f'.{img_suffix}'
            if preferred in normalized:
                normalized.remove(preferred)
            normalized.insert(0, preferred)

        self.img_suffixes = tuple(normalized)
        super().__init__(*args, **kwargs)

    def _resolve_filename(self, img_id):
        for suffix in self.img_suffixes:
            candidate = img_id + suffix
            if osp.exists(osp.join(self.img_prefix, candidate)):
                return candidate
        return img_id + self.img_suffixes[0]

    def load_annotations(self, ann_folder):
        ann_files = glob.glob(ann_folder + '/*.txt')

        if ann_files:
            data_infos = super().load_annotations(ann_folder)
            for data_info in data_infos:
                stem = osp.splitext(data_info['filename'])[0]
                data_info['filename'] = self._resolve_filename(stem)
            self.img_ids = [osp.splitext(x['filename'])[0] for x in data_infos]
            return data_infos

        data_infos = []
        for suffix in self.img_suffixes:
            ann_files.extend(glob.glob(ann_folder + f'/*{suffix}'))
        for ann_file in sorted(set(ann_files)):
            img_name = osp.basename(ann_file)
            img_id = osp.splitext(img_name)[0]
            data_infos.append(dict(filename=img_name, ann=dict(bboxes=[], labels=[])))

        self.img_ids = [osp.splitext(x['filename'])[0] for x in data_infos]
        return data_infos
