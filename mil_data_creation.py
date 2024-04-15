from dino.vision_transformer import DINOHead, VisionTransformer
from dino.vim.models_mamba import VisionMamba
from dino.config import configurations
from functools import partial

from torch import nn
import torch
from dino.utils import load_pretrained_weights

from MIL.models.resnet_custom import resnet50_baseline
import os
import glob
from natsort import os_sorted
import time
import openslide
from MIL.datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from MIL.utils.utils import print_network, collate_features
from MIL.utils.file_utils import save_hdf5
import h5py
from tqdm import tqdm
import pandas as pd
import argparse
from dino import utils
def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vim-s', type=str,
        choices=['vim-s', 'vim-s2', 'vit-s', 'vim-t', 'vit-t'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")


    parser.add_argument('--image_size', default=512, type=int, help='Image Size of global views.')
    parser.add_argument('--image_size_down', default=224, type=int, help='Image Size of local views.')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging. Enabled by default.')

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of Classes.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    # parser.add_argument("--gpu", default=0, type=int, help="GPU rank")
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--source_level", type=str)
    parser.add_argument("--target_level", type=str)
    return parser


def get_model():
    # args.image_size = 224
    args.patch_size = 16
    args.num_classes = 2
    args.n_last_blocks = 4
    args.avgpool_patchtokens = False

    if args.arch in configurations:
        config = configurations[args.arch]
        config['img_size'] = args.image_size
        config['patch_size'] = args.patch_size
        config['num_classes'] = args.num_classes

        if 'norm_layer' in config and config['norm_layer'] == "nn.LayerNorm":
            config['norm_layer'] = partial(nn.LayerNorm, eps=config['eps'])
        config['drop_path_rate'] = 0
        if args.arch.startswith('vim'):
            model = VisionMamba(return_features=True, **config)
            embed_dim = model.embed_dim
        elif args.arch.startswith('vit'):
            model = VisionTransformer(**config)
            embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        print('EMBEDDED DIM:', embed_dim)
    elif args.arch == 'resnet':
        model = resnet50_baseline(pretrained=True)
    else:
        print(f"Unknown architecture: {args.arch}")

    model.cuda()
    model.eval()
    # args.pretrained_weights = '/home/ubuntu/checkpoints/camelyon16_224_10x/vit-s_224-96/checkpoint.pth'
    args.checkpoint_key = 'teacher'
    load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    return model

def get_feaures(model, inp):
    with torch.no_grad():
        if "vit" in args.arch:
            intermediate_output = model.get_intermediate_layers(inp, args.n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if args.avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1),
                                    torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = model(inp)
    return output

def compute_w_loader(file_path, output_path, wsi, model,
    batch_size = 8, verbose = 0, print_every=20, pretrained=True,
    custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
        custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    x, y = dataset[0]
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device, non_blocking=True)

            features = get_feaures(model, batch)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'

    return output_path

def create_data_split(feat_dir, classes, slide_folder, h5_dir):
    os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(feat_dir, 'pt_files'))

    for class_name in classes:
        print(f"Processing for class {class_name}")
        args.data_h5_dir = os.path.join(h5_dir, class_name)
        slide_paths = glob.glob(os.path.join(slide_folder, class_name, '*tif'))
        for slide_file_path in tqdm(slide_paths):
            slide_id = os.path.splitext(os.path.basename(slide_file_path))[0]
#             if os.path.exists(os.path.join(feat_dir, 'pt_files', slide_id+'.pt')):
#                 continue
            bag_name = slide_id+'.h5'
            h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
            output_path = os.path.join(feat_dir, 'h5_files', bag_name)
            wsi = openslide.open_slide(slide_file_path)
            args.batch_size=16
            output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                                model = model, batch_size = args.batch_size, verbose = 0,
                                                print_every = 20, custom_downsample=1,
                                                target_patch_size=args.image_size)


            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                features = torch.from_numpy(features)
                torch.save(features, os.path.join(feat_dir, 'pt_files', slide_id+'.pt'))

def process_training(args):
    classes = ['normal', 'tumor']

    split = 'training'

    print(f"Processing split: {split}")
    slide_folder = f'/home/ubuntu/Downloads/Camelyon16/{split}/'
    feat_dir = f'{args.output_dir}/{args.image_size}_{args.source_level}at{args.target_level}/{args.arch}/{split}'
    h5_dir = f'dataset/Camelyon16/{split}/{args.image_size}_{args.target_level}x/h5/'

    create_data_split(feat_dir, classes, slide_folder, h5_dir)

    # create csv
    slide_ids = os_sorted(os.listdir(os.path.join(feat_dir, 'h5_files')))
    slide_ids = [i.split('.')[0] for i in slide_ids]
    label = [i.split('_')[0] for i in slide_ids]
    df = pd.DataFrame([slide_ids, slide_ids, label]).T
    df.columns = ['case_id', 'slide_id', 'label']
    df.to_csv(os.path.join(feat_dir, 'tumor_vs_normal.csv'))

def process_testing(args):
    split = 'testing'
    print(f"Processing split: {split}")
    slide_folder = f'/home/ubuntu/Downloads/Camelyon16/{split}/'
    feat_dir = f'{args.output_dir}/{args.image_size}_{args.source_level}at{args.target_level}/{args.arch}/{split}'
    h5_dir = f'dataset/Camelyon16/{split}/{args.image_size}_{args.target_level}x/h5/'

    classes = ['images']
    create_data_split(feat_dir, classes, slide_folder, h5_dir)

    # create csv
    df = pd.read_csv('/home/ubuntu/Downloads/Camelyon16/testing/reference.csv', header=None)
    df = df.drop([2, 3], axis=1)
    df[2] = df[0]
    df = df[[0, 2, 1]]
    df.columns = ['case_id', 'slide_id', 'label']
    df['label'] = df['label'].apply(str.lower)
    df.to_csv(os.path.join(feat_dir, 'tumor_vs_normal.csv'))

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    parser = get_args_parser()
    args = parser.parse_args()
    args.output_dir = f'clam_data/'
    model = get_model()
    process_training(args)
    process_testing(args)
