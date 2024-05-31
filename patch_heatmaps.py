from glob import glob
import os
from natsort import os_sorted
from dino.vision_transformer import DINOHead, VisionTransformer
from dino.vim.models_mamba import VisionMamba
from dino.config import configurations
from dino.main import get_args_parser
from functools import partial
from dino.utils import load_pretrained_weights
from torchvision import transforms
from torch import nn
import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


from tqdm import tqdm
import random
import matplotlib.gridspec as gridspec
import cv2


def get_model(args):

    config = configurations[args.arch]
    config['img_size'] = args.image_size
    config['patch_size'] = args.patch_size
    config['num_classes'] = args.num_classes
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
    else:
        print(f"Unknown architecture: {args.arch}")
    return model

    
dataset_dir = 'path_to_test_candidate_images'
parser = get_args_parser()
args = parser.parse_known_args()[0]

val_transform = transforms.Compose([
    transforms.Resize(args.image_size, interpolation=3),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_vim(tensor, height=14, width=14, token_position=98):
    hidden_state = tensor
    hidden_state = torch.cat((hidden_state[:, 1:token_position, :], hidden_state[:, token_position+1:, :]), dim=1)
    result = hidden_state.reshape(hidden_state.size(0), height, width, hidden_state.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

args.image_size = 224
args.patch_size = 16
args.num_classes = 2
args.n_last_blocks = 4
args.avgpool_patchtokens = False

args.checkpoint_key = 'teacher'

args.arch = 'vim-s'
args.pretrained_weights = '/home/ubuntu/checkpoints/camelyon16_224_10x/vim-s_224-96/checkpoint.pth'
model_vim_s = get_model(args)
model_vim_s.cuda()
model_vim_s.eval()
load_pretrained_weights(model_vim_s, args.pretrained_weights, 
                        args.checkpoint_key, args.arch, args.patch_size)

args.arch = 'vit-s'
args.pretrained_weights = '/home/ubuntu/checkpoints/camelyon16_224_10x/vit-s_224-96/checkpoint.pth'
model_vit_s = get_model(args)
model_vit_s.cuda()
model_vit_s.eval()
load_pretrained_weights(model_vit_s, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)


models = {
    'Vim-s':(model_vim_s, model_vim_s.layers[-1].drop_path),
    'ViT-s':(model_vit_s, model_vit_s.blocks[-1].norm1)
}

for class_name in ['tumor', 'normal']:

    img_paths = glob(os.path.join(dataset_dir, class_name, "*jpg"))
    img_paths = os_sorted(img_paths)
    target_image_idx = list(np.random.randint(0, len(img_paths), 60))

    os.makedirs(f'heatmaps/heatmaps_diverse/{class_name}', exist_ok=True)
    os.makedirs(f'heatmaps/heatmaps_diverse/{class_name}/raw', exist_ok=True)

    for i in tqdm(target_image_idx):
        img = Image.open(img_paths[i])
        img_transformed = val_transform(img).unsqueeze(0)
        img_show = img_transformed.cpu().squeeze().permute(1, 2, 0).numpy()
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())  # Normalize to [0,1]
        plt.figure(figsize=(50, 23))
        gs = gridspec.GridSpec(2, 5)
        
        # Original image
        final_img = np.array(img)/255
            
        ax0 = plt.subplot(gs[0:2, 0:2])
        ax0.imshow(final_img)
        ax0.set_title('Original Image', fontsize=40)
        ax0.axis('off')

        
        cams = []
        for idx, (model_name, (model, target_layer)) in enumerate(models.items()):
            cam = GradCAM(model=model, target_layers=[target_layer], 
                          reshape_transform=reshape_transform_vim if 'mamba' in model.__class__.__name__.lower() else reshape_transform_vit)
            grayscale_cam = cam(input_tensor=img_transformed)[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (final_img.shape[:2]))
            cam_image = show_cam_on_image(final_img, grayscale_cam, use_rgb=True)

            ax = plt.subplot(gs[idx // 3, idx % 3 + 2])
            ax.imshow(cam_image)
            Image.fromarray(cam_image).save(f'heatmaps/heatmaps_diverse/{class_name}/raw/{img_name}_{model_name}.png')
            ax.set_title(f'{model_name} Heatmap', fontsize=40)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'heatmaps/heatmaps_diverse/{class_name}/{img_name}.jpg', bbox_inches='tight', dpi=200)
        plt.close()

