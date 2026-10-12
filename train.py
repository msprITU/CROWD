from __future__ import division

import os
import warnings

from sklearn import logger

from CLTR.config import return_args, args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import torch.nn as nn
from torchvision import transforms
import CLTR.class_selection as dataset

print()
print("SELECTIVE METHOD IS USED - MAKE SURE RIGHT CLASSES ARE SELECTED")
print()

import math
from CLTR.utils import get_root_logger, setup_seed
import nni
from nni.utils import merge_parameter
import time
import CLTR.util.misc as utils
from CLTR.utils import save_checkpoint
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # add tensoorboard
from matplotlib import pyplot as plt

import os 
from CLTR.image import load_data

if args.backbone == 'resnet50' or args.backbone == 'resnet101':
    from CLTR.Networks.CDETR_Visdrone import build_model

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

def main(args):

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(repo_root, 'data', 'visdrone_data')

    if args['dataset'] == 'visdrone_dataset':
        train_file = os.path.join(
            data_root, 'train', 'visdrone_train.npy'
        )
        test_file = os.path.join(
            data_root, 'test', 'visdrone_test.npy'
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {args['dataset']!r}. "
            "Currently supported dataset: 'visdrone_dataset'."
        )

    for dataset_file in (train_file, test_file):
        if not os.path.isfile(dataset_file):
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_file}\n"
                "Run ./prepare_data.sh from the repository root first."
            )

    with open(train_file, 'rb') as outfile:
        train_data = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_data = np.load(outfile).tolist()
    
    # Creates necessary directories for creating attention maps if visualization is enabled.
    # Not implemented for Visdrone -Yagiz
    if args['y_visualize']:

        attention_maps_file = './npydata/am_dataset.npy'

        with open(attention_maps_file, 'rb') as outfile:
            attention_maps_data = np.load(outfile).tolist()


    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)

    model = model.cuda()
    if args['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args['local_rank']])
        path = './save_file/log_file_single/' + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        args['save_path'] = path
        if not os.path.exists(args['save_path']) and args['local_rank'] <= 0:
            os.makedirs(args['save_path'])
        if args['save']:
            logger = get_root_logger(path + '/1.log')
        else:
            logger = get_root_logger('./save_file/log_file/debug/debug.log')
        writer = SummaryWriter(path)

    else:
        args['train_patch'] = True
        return_args.train_patch = True
        model = nn.DataParallel(model, device_ids=[0])
        path = './save_file/log_file_single/' + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        args['save_path'] = path
        if not os.path.exists(args['save_path']) and args['local_rank'] <= 0:
            os.makedirs(args['save_path'])
        if args['save']:
            logger = get_root_logger(path + '/1.log')
        else:
            logger = get_root_logger('./save_file/log_file/debug/debug.log')
        writer = SummaryWriter(path)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("model params:", num_params )
    logger.info("model params: = {:.3f}\t".format(num_params))

    logger.info(f"Using Dataset in path: {train_file}")
    
    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])
    if args['local_rank'] <= 0:
        logger.info(args)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args['lr_step']], gamma=0.1, last_epoch=-1)

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            logger.info("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
            
            #DEBUG, below should be commented -Yagiz
            #attention_maps(attention_maps_data, model, 77777, args, logger)
            #logger.info(f'Saved attention maps for epoch 77777')   

        else:
            logger.info("=> no checkpoint found at '{}'".format(args['pre']))

    print('best result:', args['best_pred'])
    logger.info('best result = {:.3f}'.format(args['best_pred']))
    torch.set_num_threads(args['workers'])

    if args['local_rank'] <= 0:
        logger.info('best result={:.3f}\t start epoch={:.3f}'.format(args['best_pred'], args['start_epoch']))


    if args['local_rank'] <= 0:
        logger.info('start training!')

    eval_epoch = 0
    for epoch in range(args['start_epoch'], args['epochs'] + 1):

        train(train_data, model, criterion, optimizer, epoch, scheduler, logger, writer, args)

        '''inference '''

        if args['y_visualize'] and epoch % (args['y_save_epochs']) == 0:

                ## Attention Maps, Yagiz
                attention_maps(attention_maps_data, model, epoch, args, logger)
                logger.info(f'Saved attention maps for epoch {epoch}')

        if epoch % args['test_per_epoch'] == 0:

            pred_mae, pred_mse, visi = validate(test_data, model, criterion, epoch, logger, args)

            writer.add_scalar('Metrcis/MAE', pred_mae, eval_epoch)
            writer.add_scalar('Metrcis/MSE', pred_mse, eval_epoch)

            # save_result
            if args['save']:

                is_best = pred_mae < args['best_pred']
                args['best_pred'] = min(pred_mae, args['best_pred'])
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args['pre'],
                    'state_dict': model.state_dict(),
                    'best_prec1': args['best_pred'],
                    'optimizer': optimizer.state_dict(),
                }, visi, is_best, args['save_path'])

            ##ESAD - separate CKPT saving
            # epoch > 0 can be added -Yagiz
            if args['save']:

                if epoch > 0:
                    print("Saving...")
                    is_best = pred_mae < args['best_pred']
                    args['best_pred'] = min(pred_mae, args['best_pred'])
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args['pre'],
                        'state_dict': model.state_dict(),
                        'best_prec1': args['best_pred'],
                        'optimizer': optimizer.state_dict(),
                    }, visi, is_best, args['save_path'], filename='checkpoint_' + str(epoch) + '.pth')


            ##

            end = time.time()

            if args['local_rank'] <= 0:
                logger.info(
                    'Testing Epoch:[{}/{}]\t mae={:.3f}\t mse={:.3f}\t best_mae={:.3f}\t'.format(
                        epoch,
                        args['epochs'],
                        pred_mae, pred_mse,
                        args['best_pred']))


def collate_wrapper(batch):
    targets = []
    imgs = []
    fname = []

    for item in batch:

        #if return_args.train_patch:
        fname.append(item[0])

        for i in range(0, len(item[1])):
            imgs.append(item[1][i])

        for i in range(0, len(item[2])):
            targets.append(item[2][i])
        # else:
        # fname.append(item[0])
        # imgs.append(item[1])
        # targets.append(item[2])

    return fname, torch.stack(imgs, 0), targets


def train(Pre_data, model, criterion, optimizer, epoch, scheduler, logger, writer, args):
    # losses = AverageMeter()

    # Synchronize GPU and start time tracking, Bekir
    torch.cuda.synchronize()
    start = time.time()

     # Dataset Loading and DataLoader Creation, Bekir
    train_data = dataset.listDataset(Pre_data, args['save_path'],
                                     shuffle=True,
                                     transform=transforms.Compose([
                                         transforms.RandomGrayscale(p=args['gray_p'] if args['gray_aug'] else 0),
                                         transforms.ToTensor(),

                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),
                                     ]),
                                     train=True,
                                     args=args)

    # DataLoader Preparation (for distributed training), Bekir
    if args['distributed']:
        datasampler = DistributedSampler(train_data, num_replicas=dist.get_world_size(), rank=args['local_rank'])
        datasampler.set_epoch(epoch)
    else:
        datasampler = None

    # DataLoader initialization
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args['batch_size'],
        drop_last=False,
        collate_fn=collate_wrapper,
        sampler=datasampler,
        num_workers=16,
        prefetch_factor=2,
        pin_memory=True
    )

    # Model in training mode, Bekir
    model.train()
    loss_log = []

    # Training Loop - Iterating over the DataLoader, Bekir
    for i, (fname, img, targets) in enumerate(train_loader):
        
        # Move the images to GPU
        img = img.cuda()

        # Forward pass through the model, d6 holds prediction of model
        d6 = model(img)

        # Calculate the loss
        loss_dict = criterion(d6, targets)
        weight_dict = criterion.weight_dict

        # print()
        # print(loss_dict)
        # print()
      
        # #ESAD
        # for k in loss_dict.keys():
        #     if k in weight_dict:
        #         print(f"Key: {k}")

        # print()

        # print('DONEDONEDONE')


        # Aggregate losses with their respective weights
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # TensorBoard Logging
        writer.add_scalar('loss/total', loss, len(train_loader) * epoch + i)
        writer.add_scalar('loss/loss_ce', loss_dict['loss_ce'], len(train_loader) * epoch + i)
        writer.add_scalar('loss/loss_point', loss_dict['loss_point'], len(train_loader) * epoch + i)
        writer.add_scalar('lr/lr_backbone', optimizer.param_groups[0]['lr'], len(train_loader) * epoch + i)

        loss_log.append(loss.item())

        # Backpropagation and Optimizer Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # End of epoch - Log training time and adjust learning rate
    torch.cuda.synchronize()
    epoch_time = time.time() - start
    scheduler.step()
    if args['local_rank'] <= 0:
        logger.info('Training Epoch:[{}/{}]\t loss={:.5f}\t lr={:.6f}\t epoch_time={:.3f}'.format(epoch,
                                                                                                                args[
                                                                                                                    'epochs'],
                                                                                                                np.mean(
                                                                                                                    loss_log),
                                                                                                                args[
                                                                                                                    'lr'],
                                                                                                                epoch_time))


def validate(Pre_data, model, criterion, epoch, logger, args):
    if args['local_rank'] <= 0:
        logger.info('begin test')

    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1,
    )

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []

    ii_limit = 5
    ii = 0

    for i, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader):

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(kpoint.shape) == 5:
            kpoint = kpoint.squeeze(0)


        with torch.no_grad():
            img = img.cuda()
            outputs = model(img)

        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits = out_logits.view(1, -1, 2)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                               kpoint.shape[0] * args['num_queries'], dim=1)
        count = 0
        gt_count = torch.sum(kpoint).item()
        for k in range(topk_values.shape[0]):
            sub_count = topk_values[k, :]
            sub_count[sub_count < args['threshold']] = 0
            sub_count[sub_count > 0] = 1
            sub_count = torch.sum(sub_count).item()
            count += sub_count

        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)

        ii += 1

        if ii >= ii_limit:
          break

    #ESAD 5 IMAGE VALIDATION
    mae = mae / ii_limit
    mse = math.sqrt(mse / ii_limit) 

    # mae = mae / len(test_loader)
    # mse = math.sqrt(mse / len(test_loader))

    print('mae', mae, 'mse', mse)
    return mae, mse, visi

def attention_maps(attention_maps_data, model, epoch, args, logger):

    logger.info(f"Generating visualizations for epoch: {epoch}")

    import torch.nn.functional as F


    colors = ['red','green','blue','magenta']

    attention_data_loader = torch.utils.data.DataLoader(
        dataset.listDataset(attention_maps_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            args=args, train=False),
        batch_size=1,
    )


    for i, (fname, img, kpoint, targets, patch_info) in enumerate(attention_data_loader):

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(kpoint.shape) == 5:
            kpoint = kpoint.squeeze(0)

        #logger.info(f"Processing image {i+1}/{len(attention_data_loader)}: {fname}")
        #print(f"Image shape before processing: {img.shape}")
        #print(f"Kpoint shape: {kpoint.shape}")
        
        points = []

        for k_p in kpoint:
            k_p = k_p.cpu()
            k_p = k_p.numpy()
            n_zero_points = np.nonzero(k_p)
            n_zero_points = np.array(n_zero_points).T  # Transpose to get (num_points, 2) shape
            #print(n_zero_points)
            points.append(n_zero_points)


        image = img
        img = img.cuda()

        conv_features, enc_attn_weights, dec_attn_weights, dec_selfattn_weights = [], [], [], []

        hooks = [
            model.module.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
            ),

            model.module.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),

            model.module.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1]), 
            ),
        ]

        model.eval()

        with torch.no_grad():
            outputs = model(img)

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]
        # Extracting the feature maps
        f_map = conv_features['0']

        #print(f"Feature map shape: {f_map.tensors.shape}")  # Debugging line
        #print(f"Image {i} Encoder attention shape: {enc_attn_weights.shape}")
        #print(f"Image {i} Decoder attention shape: {dec_attn_weights.shape}")

        # Extract feature map tensors
        feature_map = f_map.tensors  # Shape: [batch_size, channels, height, width]

        # Create a directory to save feature map visualizations
        feature_map_dir = os.path.join(args['save_path'], f'epoch_{epoch}', 'feature_maps')
        if not os.path.exists(feature_map_dir):
            os.makedirs(feature_map_dir)

        encoder_sattn_dir = os.path.join(args['save_path'], f'epoch_{epoch}', 'encoder_selfattention')
        if not os.path.exists(encoder_sattn_dir):
            os.makedirs(encoder_sattn_dir)

        decoder_c_attn_dir = os.path.join(args['save_path'], f'epoch_{epoch}', 'decoder_crossattention')
        if not os.path.exists(decoder_c_attn_dir):
            os.makedirs(decoder_c_attn_dir)

        for patch_idx in range(feature_map.shape[0]):
            #print(f"Visualizing feature map for patch {patch_idx}...")

            # Get the feature map for this patch
            patch_feature_map = feature_map[patch_idx]  # [channels, height, width]

            # Average across channels for visualization
            avg_feature_map = patch_feature_map.mean(dim=0).cpu().numpy()  # [height, width]

            # Normalize for better visualization
            #avg_feature_map -= avg_feature_map.min()
            #avg_feature_map /= avg_feature_map.max()

            # Get the original patch image (assuming 'image' is your normalized input tensor)
            patch_img = image[patch_idx].permute(1, 2, 0).cpu()
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            patch_img = patch_img * std + mean
            patch_img = torch.clamp(patch_img, 0, 1).numpy()

            # Plot both the original patch and the feature map side by side
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(patch_img)
            axs[0].set_title(f'Original Patch {patch_idx}')
            axs[0].axis('off')

            im1 = axs[1].imshow(avg_feature_map, cmap='viridis')
            axs[1].set_title(f'Feature Map - File {i} - Patch {patch_idx} - Queries: {args["num_queries"]} - Epoch {epoch}')
            axs[1].axis('off')
            plt.colorbar(im1, ax=axs[1])

            plt.tight_layout()
            plt.savefig(os.path.join(feature_map_dir, f'image_{i}_patch_{patch_idx}_feature_map_with_original.png'), dpi=300)
            plt.close()
        
        #logger.info(f"Feature map visualizations saved for image {i} in epoch {epoch}")

        shape = f_map.tensors.shape[-2:]  # [8, 8]

        # This is needed for DETR50
        fact = 32

        for patch_idx in range(enc_attn_weights.shape[0]):
            #print(f"Patch {patch_idx} Encoder Attention shape: {enc_attn_weights[patch_idx].shape}")
            
            # Proper reshaping
            patch_attention = enc_attn_weights[patch_idx]  # [64, 64]
            sattn = patch_attention.reshape(shape[0], shape[1], shape[0], shape[1])  # [8, 8, 8, 8]

            #print(f"Patch {patch_idx} reshaped attention shape: {sattn.shape}")  # Debugging line

            sattn = sattn.cpu()

            #print(f"Patch {patch_idx} Keypoints: {points[patch_idx]}")  # Debugging line
            #print(f"Patch {patch_idx} Keypoints shape: {points[patch_idx].shape}")  # Debugging line

            if points[patch_idx].shape[0] < 4:
                #logger.info(f"Image {i} Patch {patch_idx} has less than 4 points, skipping self-attention visualization")
                continue

            else:
                np.random.seed(7)  # For reproducibility
                #print(points[patch_idx])
                np.random.shuffle(points[patch_idx])  # Shuffle the points
                #print(points[patch_idx])
                idxs = points[patch_idx][0:4]  # Get the first 4 points for visualization

            idxs = idxs[:, [2,1]]  # Keep only the last two columns (x, y coordinates)
            #print(f"Image {i} Selected points for visualization: {idxs}")  # Debugging line
            # Visualization
            scale = 0.5
            fig = plt.figure(constrained_layout=True, figsize=(20 * scale, 6 * scale))

            gs = fig.add_gridspec(2, 4)
            axs = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[0, -1]),
                fig.add_subplot(gs[1, -1]),
            ]

            for c_id, (x, y) in enumerate(idxs):
                if x < 256 and y < 256:
                    idx_row, idx_col = y // fact, x // fact
                    if idx_row < shape[0] and idx_col < shape[1]:
                        # Show attention FROM this position TO all other positions
                        axs[c_id].imshow(sattn[idx_row, idx_col], cmap='cividis', interpolation='nearest')
                        axs[c_id].set_title(f'patch{patch_idx}-attn({x},{y}) - {colors[c_id]}')

            # Show the patch
            fcenter_ax = fig.add_subplot(gs[:, 1:-1])
            patch_img = image[patch_idx].permute(1, 2, 0).cpu()
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            patch_img = patch_img * std + mean
            patch_img = torch.clamp(patch_img, 0, 1)

            fcenter_ax.imshow(patch_img)

            for color_id, (x, y) in enumerate(idxs):
                if x < 256 and y < 256:
                    fcenter_ax.add_patch(plt.Circle((x, y), fact // 8, color=colors[color_id]))
            fcenter_ax.set_title(f'Encoder Self Attention - Image {i} - Patch {patch_idx} - Queries: {args["num_queries"]} - Epoch {epoch}')
            fcenter_ax.axis('off')
            
            save_dir = os.path.join(args['save_path'], f'epoch_{epoch}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(os.path.join(encoder_sattn_dir, f'image_{i}_patch{patch_idx}_encoder_attention_{fname}_epoch_{epoch}_queries_{args["num_queries"]}.png'), dpi=300)
            plt.close(fig)

        #logger.info(f"Encoder self-attention visualizations saved for image {i} in epoch {epoch}")

        #print("Creating decoder attention visualizations...")

        h, w = conv_features['0'].tensors.shape[-2:]

        for patch_idx in range(dec_attn_weights.shape[0]):
            if points[patch_idx].shape[0] == 0:
                #print(f"Image {i} Patch {patch_idx} has no points, skipping decoder attention visualization")
                #logger.info(f"Image {i} Patch {patch_idx} has no points, skipping decoder attention visualization")
                continue
            #print(f"Processing decoder attention for patch {patch_idx}")
            # Get predictions for this patch to identify active queries
            patch_logits = outputs['pred_logits'][patch_idx]  # [700, 4]
            patch_points = outputs['pred_points'][patch_idx]  # [700, 3]

            #print(patch_points.shape)
            
            # Get probabilities and find queries with high confidence
            #print(f"Patch {patch_idx} Logits shape: {patch_logits.shape}, Patch Logits: {patch_logits}")  # Debugging line
            
            # Changed here to 1, -1 might be correct as well. - Yagiz
            probas = patch_logits.softmax(-1)[..., 1:]  # Remove background class

            confidences = probas.max(-1).values 

            keep = (confidences > args['threshold']) & (confidences < 0.99)

            confident_indices = keep.nonzero().squeeze(-1)

            #print(f"Confident indices for patch {patch_idx}: {confident_indices}")  # Debugging line

            #print(f"Patch {patch_idx} keep shape: {keep.shape}, Keeped: {keep}")  # Debugging line

            filtered_confidences = confidences[confident_indices]  # Keep only confident scores

            #print(f"Patch {patch_idx} Confident scores shape: {filtered_confidences.shape}, Confident Scores: {filtered_confidences}")  # Debugging line

            #topk = torch.topk(filtered_confidences, k=min(4, keep.sum().item()), dim=0)

            #print(f"Patch {patch_idx} Topk shape: {topk.indices.shape}, Topk indices: {topk.indices}")  # Debugging line

            #confident_queries = confident_indices[topk.indices]

            #print(f"Patch {patch_idx} Confident queries shape: {confident_queries.shape}, Confident Queries: {confident_queries}")  # Debugging line

            if confident_indices.shape[0] == 0:
                #logger.info(f"No confident predictions in patch {patch_idx}, skipping decoder attention")
                continue
                
            # Get decoder attention for this patch
            patch_dec_attn = dec_attn_weights[patch_idx]  # [700, 64] - query to query attention
            
            # We need cross-attention (query to encoder features) for spatial visualization
            # Since we only have self-attention, we'll visualize the most confident queries

            #print(f"Keep for patch {patch_idx}: {keep}")  # Debugging line
            #print(f"Patch {patch_idx} has {keep.shape} ")

            #confident_queries = keep.nonzero().squeeze(-1)[:min(4, keep.sum())]  # Top 4 queries

            #print(f"Patch {patch_idx} has {len(confident_queries)} confident queries: {confident_queries}")  # Debugging line

            #print(f"Confident queries for patch {patch_idx}: {confident_queries}")
                
            # Create visualization for decoder attention
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            ax_actual_image = axes[0]
            patch_img = image[patch_idx].permute(1, 2, 0).cpu()
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            patch_img = patch_img * std + mean
            patch_img = torch.clamp(patch_img, 0, 1)
            ax_actual_image.imshow(patch_img)
            ax_actual_image.set_title(f'Original Patch')
            ax_actual_image.axis('off')
            ax_queries = axes[1]
            ax_queries.set_facecolor('white')
            ax_queries.set_xticks([])
            ax_queries.set_yticks([])
            ax_queries.axis('on')
            ax_queries.invert_yaxis()  # makes (0,0) top-left
            ax_queries.set_xlim(0, 256)
            ax_queries.set_ylim(0, 256)
            ax_queries.set_aspect('equal', adjustable='box')
            ax_queries.set_title(f'Object Queries')

            
            
            points[patch_idx] = points[patch_idx][:, [2, 1]]  # Keep only the last two columns (x, y coordinates)
            #print(f"Patch {patch_idx} Points shape: {points[patch_idx].shape}, Points: {points[patch_idx]}")  # Debugging line
            # Plot the points on the query attention visualization
            for point in points[patch_idx]:
                if point[0] < 256 and point[1] < 256:
                    #print(f"Plotting point {point} in patch {patch_idx} for Image {i}")
                    ax_queries.scatter(point[0], 256-point[1], c='blue', s=50, alpha=0.5)

            def color_confidence(confidence):
                """Map confidence to color."""
                if confidence < 0.5:
                    return 'red'
                elif confidence < 0.75:
                    return 'orange'
                else:
                    return 'green'

            for q_idx, query_id in enumerate(confident_indices):

                #print(f"Processing query {query_id} in patch {patch_idx} for Image {i}")
                    
                query_id = query_id.item()

                #print(f"Visualizing decoder attention for query {query_id} in patch {patch_idx} in Image {i}")

                # For self-attention, show how this query relates to other queries
                #print(f"Patch {patch_idx} Query {query_id} attention shape: {patch_dec_attn[query_id].shape}")  # Debugging line
                query_attn = patch_dec_attn[query_id]  # [700] - attention from this query to all queries

                #print(query_attn.shape)
                
                # Reshape to approximate spatial layout (not perfect but gives an idea)
                # Since queries don't have inherent spatial structure, we'll show as 1D
                query_attn = query_attn.cpu().view(h, w)

                #print(f"Query attention shape: {query_attn.shape}")  # Debugging line
                #print(f"Query attention values: {query_attn}")  # Debugging line
                
                query_attn = query_attn.numpy()

                # Bottom row: Show patch with predicted point
                
                # Get predicted point for this query (denormalize to patch coordinates)
                #print(patch_points[query_id])
                pred_point = patch_points[query_id][:2].cpu()  # [x, y] normalized

                #print(f"Predicted point for query {query_id}: {pred_point}")  # Debugging line
                pred_x = pred_point[0] * 256  
                pred_y = pred_point[1] * 256
                
                # Get confidence
                confidence = probas[query_id].max().item()
                
                # Plot predicted point
                ax_queries.scatter(pred_y, 256 - pred_x, c=color_confidence(confidence), s=10, alpha=0.8)


            
            plt.suptitle(f'Object Query Predictions - Image {i} - Patch {patch_idx} - Queries: {args["num_queries"]} - Epoch {epoch}')
            plt.tight_layout()
            
            # Save decoder attention visualization
            fig.savefig(os.path.join(decoder_c_attn_dir, f'image_{i}_patch{patch_idx}_decoder_attention_epoch_{epoch}_queries_{args["num_queries"]}.png'), dpi=300)
            plt.close(fig)

        #logger.info(f"Decoder cross-attention visualizations saved for image {i} in epoch {epoch}")

    logger.info(f"Visualizations for epoch: {epoch} completed and saved.")
    




if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
