import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class UnalignedDataset(BaseDataset):
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--dcum',
            type=util.str2bool,
            nargs='?',
            const=True,
            default=False,
            help=(
                'training-only domain-conditional unpaired marginal: sample B '
                'uniformly from the A domain while forbidding the same stem'
            ),
        )
        return parser
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self._dcum_enabled = bool(getattr(opt, 'dcum', False)) and bool(opt.isTrain)
        self._B_by_domain = {}
        if self._dcum_enabled:
            for path in self.B_paths:
                domain, _ = self._domain_and_stem(path)
                self._B_by_domain.setdefault(domain, []).append(path)
            if not self._B_by_domain:
                raise RuntimeError('DCUM requires domain-prefixed B filenames')

    @staticmethod
    def _domain_and_stem(path):
        """Read ``domain__stem`` from a materialized view without pairing it."""
        name = os.path.splitext(os.path.basename(path))[0]
        if '__' not in name:
            raise ValueError(
                'DCUM expects materialized filenames in domain__stem form: %s' % path
            )
        domain, stem = name.split('__', 1)
        return domain, stem

    def _sample_dcum_B(self, A_path):
        domain, a_stem = self._domain_and_stem(A_path)
        pool = self._B_by_domain.get(domain, [])
        eligible = [
            path for path in pool
            if self._domain_and_stem(path)[1] != a_stem
        ]
        if not eligible:
            raise RuntimeError(
                'DCUM has no different-stem B candidate for domain %s' % domain
            )
        # Exactly one Python-RNG draw, just like the official unaligned path.
        return eligible[random.randint(0, len(eligible) - 1)]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self._dcum_enabled:
            B_path = self._sample_dcum_B(A_path)
        elif self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
            B_path = self.B_paths[index_B]
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
