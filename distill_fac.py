import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, StyleTranslator 
import wandb
import copy
import random
from reparam_module import ReparamModule
from contrastive_loss import *

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange((args.Iteration // 10) * 9, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images' % (c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f' % (
        ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes, args.ipc, 1 if args.single_channel else 3, im_size[0], im_size[1]), dtype=torch.float)
    styles = nn.ModuleList([StyleTranslator(in_channel=1 if args.single_channel else 3, mid_channel=channel, out_channel=channel, kernel_size=3)
                            for _ in range(args.n_style)])
    sim_content_net = Extractor(num_classes)
    syn_lr = torch.tensor(args.lr_teacher)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            if args.single_channel:
                image_syn.data[c] = get_images(c, args.ipc).detach().data.mean(dim=1, keepdim=True)
            else:
                image_syn.data[c] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')

    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    styles = styles.to(args.device)
    sim_content_net = sim_content_net.to(args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.95)
    optimizer_style = torch.optim.SGD(styles.parameters(), lr=args.lr_style, momentum=0.95)
    optimizer_sim_content = torch.optim.SGD(sim_content_net.parameters(), lr=0.001, momentum=0.9)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.9)
    optimizer_img.zero_grad()

    if args.distributed:
        sim_content_net = torch.nn.DataParallel(sim_content_net)

    criterion = nn.CrossEntropyLoss().to(args.device)
    contrast = SupConLoss().to(args.device)
    print('%s training begins' % get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    for it in range(0, args.Iteration + 1):
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                        args.device)  # get a random model

                    images_syn_eval = copy.deepcopy(image_syn.detach())
                    styles_eval = copy.deepcopy(styles)

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval,
                                                             [images_syn_eval, styles_eval],
                                                             None, testloader, args, mode='style_translator')
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.flatten(0, 1).detach()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(styles.state_dict(), os.path.join(save_dir, "styles_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
                    torch.save(styles.state_dict(), os.path.join(save_dir, "weights_best.pt"))

                wandb.log({"Pixels": wandb.Histogram(image_syn.detach().cpu())}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(grid.detach().cpu())}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(image_save.detach().cpu())}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(
                            grid.detach().cpu())}, step=it)

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(
            args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()
        sim_content_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [
            torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(num_classes * args.ipc, device=args.device)
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()
            label = these_indices // args.ipc
            base_idx = these_indices % args.ipc
            style_idx = random.randint(0, args.n_style - 1)
            base = syn_images[label, base_idx, :, :, :]
            style = styles[style_idx]
            x = style(base)
            this_y = label.long()

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        if not indices_chunks:
            indices = torch.randperm(num_classes * args.ipc, device=args.device)
            indices_chunks = list(torch.split(indices, args.batch_syn))

        these_indices = indices_chunks.pop()
        label = these_indices // args.ipc
        base_idx = these_indices % args.ipc
        style_idx = torch.randperm(args.n_style, device=args.device)[:2]
        base = syn_images[label, base_idx, :, :, :]
        style_0 = styles[style_idx[0]]
        x_0 = style_0(base)
        style_1 = styles[style_idx[1]]
        x_1 = style_1(base)
        x = torch.cat([x_0, x_1], dim=0)

        if args.dsa and (not args.no_aug):
            x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

        logits_c, embed_c = sim_content_net(x)
        club_content_loss = ((torch.nn.functional.cosine_similarity(embed_c[:base.shape[0]], embed_c[base.shape[0]:]) + 1.) / 2.).mean() * args.lambda_club_content

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss + club_content_loss

        optimizer_img.zero_grad()
        optimizer_style.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_img.step()
        optimizer_style.step()
        optimizer_lr.step()
        syn_lr.data = syn_lr.data.clip(min=0.001)

        if not indices_chunks:
            indices = torch.randperm(num_classes * args.ipc, device=args.device)
            indices_chunks = list(torch.split(indices, args.batch_syn))

        these_indices = indices_chunks.pop()
        label = these_indices // args.ipc
        base_idx = these_indices % args.ipc
        style_idx = torch.randperm(args.n_style, device=args.device)[:2]
        base = syn_images[label, base_idx, :, :, :]
        style_0 = styles[style_idx[0]]
        x_0 = style_0(base)
        style_1 = styles[style_idx[1]]
        x_1 = style_1(base)
        x = torch.cat([x_0, x_1], dim=0)

        if args.dsa and (not args.no_aug):
            x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

        logits_c, embed_c = sim_content_net(x)

        cls_content_loss = criterion(logits_c, torch.cat([label, label], dim=0)) * args.lambda_cls_content
        likeli_content_loss = ((1 - torch.nn.functional.cosine_similarity(embed_c[:base.shape[0]], embed_c[base.shape[0]:])) / 2.).mean() * args.lambda_likeli_content
        embed_c_0 = torch.nn.functional.normalize(embed_c[:base.shape[0]])
        embed_c_1 = torch.nn.functional.normalize(embed_c[base.shape[0]:])
        contrast_content_loss = contrast(torch.stack([embed_c_0, embed_c_1], dim=1), label) * args.lambda_contrast_content
        sim_content_loss = cls_content_loss + likeli_content_loss + contrast_content_loss

        optimizer_sim_content.zero_grad()
        sim_content_loss.backward()
        optimizer_sim_content.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
            "Param_Loss": param_loss.detach().cpu(),
            "Club_Content_Loss": club_content_loss.detach().cpu(),
            "Sim_Content_Loss": sim_content_loss.detach().cpu(),
            "Cls_Content_Loss": cls_content_loss.detach().cpu(),
            "Likeli_Content_Loss": likeli_content_loss.detach().cpu(),
            "Contrast_Content_Loss": contrast_content_loss.detach().cpu(),
            "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it % 10 == 0:
            print('%s iter = %04d, loss = %.4f, param_loss = %.4f, club_content_loss = %.4f, sim_content_loss = %.4f, cls_content_loss = %.4f, likeli_content_loss = %.4f, contrast_content_loss = %.4f' % (get_time(), it, grand_loss.item(), param_loss.item(), club_content_loss.item(), sim_content_loss.item(), cls_content_loss.item(), likeli_content_loss.item(), contrast_content_loss.item()))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette',
                        help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=15000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=100, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_style', type=float, default=100, help='learning rate for updating style translator')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=64, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true',
                        help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--max_files', type=int, default=None,
                        help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None,
                        help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--n_style', type=int, default=5, help='the number of styles')
    parser.add_argument('--single_channel', action='store_true', help="using single-channel but more basis")
    parser.add_argument('--lambda_club_content', type=float, default=0.1)
    parser.add_argument('--lambda_likeli_content', type=float, default=1.)
    parser.add_argument('--lambda_cls_content', type=float, default=1.)
    parser.add_argument('--lambda_contrast_content', type=float, default=1.)

    args = parser.parse_args()

    main(args)


