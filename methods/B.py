import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed, pil_loader
import copy
import os, math
import pickle
from PIL import Image
from itertools import chain
import shutil
from torchvision import transforms
import ipdb
import copy
import logging  # 导入logging模块
import datetime

class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, transform):
        """
        从给定的数据集中选取指定索引的数据，构造子数据集。

        参数：
            dataset (Dataset): 原始数据集
            indices (list): 需要提取的索引列表
        """
        
        self.images = dataset.images[indices]
        self.labels = dataset.labels[indices]
        self.transform = transform

    def __getitem__(self, idx):
        """
        根据新的索引获取数据
        """
        image = self.transform(Image.fromarray(self.images[idx]))  # 通过索引映射到原始数据集的索引
        label = self.labels[idx] 
        return idx, image, label

    def __len__(self):
        """
        返回子数据集的大小
        """
        return len(self.indices)

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

def label_distribution(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

class CustomConcatDataset(ConcatDataset):
    """自定义 ConcatDataset，支持 labels 属性"""
    def __init__(self, datasets, transform, args, logger):
        if args['dataset'] == 'tiny_imagenet':
            self.use_path = True
        else:
            self.use_path = False
        # 合并所有数据集的标签
        datasets = [*datasets[0], *datasets[1]] 

        self.labels = np.concatenate([ds.labels for ds in datasets]) 
        self.images = np.concatenate([ds.images for ds in datasets]) 
        self.transform = transform
        
        LD = label_distribution(self.labels)
        logger.info("Label Distribution: %s" % str(LD))
        print(LD)
    def __getitem__(self, idx):
        """
        根据新的索引获取数据
        """
        if self.use_path:
            image = self.transform(pil_loader(self.images[idx]))
        else:
            image = self.transform(Image.fromarray(self.images[idx]))  # 通过索引映射到原始数据集的索引
        label = self.labels[idx] 
        return idx, image, label
    
    def __len__(self):
        return len(self.labels)
    

class FLDataSelector:
    def __init__(self, num_clients, local_features, query_budget):
        """
        初始化联邦数据选择器
        :param num_clients: 客户端数量
        :param local_features: 每个客户端的特征矩阵 (列表, 每个元素形状为 (n_i, d))
        :param query_budget: 服务器总的查询预算 (nq)
        """
        self.num_clients = num_clients
        self.local_features = local_features
        self.query_budget = query_budget
        self.P = [self.generate_random_orthogonal_matrix(F.shape[0]) for F in local_features]  # 生成每个客户端的 P_i
        self.Q = self.generate_random_orthogonal_matrix(local_features[0].shape[1])  # 生成全局的 Q

    def generate_random_orthogonal_matrix(self, size):
        """
        生成随机正交矩阵
        :param size: 矩阵的大小 (size, size)
        :return: 随机正交矩阵
        """
        random_matrix = torch.randn(size, size)
        Q, _ = torch.linalg.qr(random_matrix)  # QR 分解生成正交矩阵
        return Q

    def mask_local_data(self):
        """
        对每个客户端的本地数据进行掩码
        :return: 掩码后的本地数据 (列表)
        """
        masked_features = []
        for i, F in enumerate(self.local_features):
            masked_F = torch.matmul(torch.matmul(self.P[i], F), self.Q)  # X'_i = P_i X_i Q
            masked_features.append(masked_F)
        return masked_features

    def compute_local_leverage_scores(self, masked_features):
        """
        计算每个客户端的杠杆分数
        :param masked_features: 掩码后的本地数据 (列表)
        :return: 每个客户端的杠杆分数 (列表)
        """
        leverage_scores = []
        for F in masked_features:
            U, _, _ = torch.svd(F)  # 计算 SVD，U 是左奇异矩阵
            tau_i = torch.norm(U, dim=1) ** 2  # τ_j = ||e_j^T U_i||^2
            tau_i /= tau_i.sum()  # 归一化，防止不同客户端的数值尺度差异
            leverage_scores.append(tau_i)
        return leverage_scores

    def aggregate_leverage_scores(self, leverage_scores):
        """
        服务器端聚合杠杆分数，同时考虑客户端均衡性，计算全局采样概率
        :return: 归一化的全局采样概率
        """
        client_weights = torch.tensor([1.0 / self.num_clients] * self.num_clients)  # 每个客户端的权重均衡

        # 计算每个客户端内部的归一化杠杆分数
        all_scores = [tau * client_weights[i] for i, tau in enumerate(leverage_scores)]
        
        all_scores = torch.cat(all_scores)  # 拼接所有客户端的杠杆分数
        p = all_scores / all_scores.sum()  # 归一化，计算采样概率
        return p

    def sample_data(self, p):
        """
        服务器端根据采样概率进行数据选择，并确保客户端间的均衡性
        :param p: 归一化的全局采样概率 (Tensor)
        :return: 每个客户端分别选中的数据索引
        """
        min_samples_per_client = self.query_budget // self.num_clients  # 确保每个客户端至少有一定数量
        client_selected_indices = {i: [] for i in range(self.num_clients)}  # 存储每个客户端选中的索引

        start_idx = 0
        for i in range(self.num_clients):
            end_idx = start_idx + len(self.local_features[i])
            client_p = p[start_idx:end_idx]
            sampled = torch.multinomial(client_p, min_samples_per_client, replacement=True)
            client_selected_indices[i] = sampled.cpu().numpy()
            start_idx = end_idx

        # 剩余的采样机会按照全局概率采样
        remaining_budget = self.query_budget - sum(len(v) for v in client_selected_indices.values())
        if remaining_budget > 0:
            extra_samples = torch.multinomial(p, remaining_budget, replacement=True).tolist()
            for idx in extra_samples:
                for i in range(self.num_clients):
                    start_idx = sum(len(self.local_features[j]) for j in range(i))
                    end_idx = start_idx + len(self.local_features[i])
                    if start_idx <= idx < end_idx:
                        client_selected_indices[i].append(idx)
                        break
        
        return client_selected_indices



class BBBB(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False) 
        self.class_order = torch.tensor(args["class_order"], device=args["gpu"])
        self.args = args
        self.mem_size = args['mem_size']
        self.r = args['r']
        self.ltc = args['ltc']
        self.transform, self.normalizer = self._get_norm_and_transform(self.args["dataset"])
        self.selected_data_indices = []  # 用于存储每个任务选择的数据索引
        self.retained_ds_all = [[] for _ in range(args['num_users'])]
        
    def _get_norm_and_transform(self, dataset):
        """根据数据集返回归一化和数据增强方法"""
        if dataset == "cifar100":
            data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        elif dataset == "cifar10":
            data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        
        elif dataset == "tiny_imagenet":
            data_normalize = dict(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
            train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        return train_transform, Normalizer(**dict(data_normalize))

    def _get_client_dataset(self, client_idx):
        """
        获取某个客户端的数据集
        :param client_idx: 客户端的索引
        :return: 客户端的数据集
        """       
        # 获取当前任务的数据集
        self.train_dataset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        
        client_indices = self.user_groups[client_idx]
        
        # 使用 DatasetSplit 获取客户端的数据集
        client_dataset = DatasetSplit(self.train_dataset, client_indices)
        return client_dataset

    def after_task(self):
        """在每个任务结束后，提取数据表征并进行数据选择"""
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        test_acc = self._compute_accuracy(self._old_network, self.test_loader)
        self.logger.info("After Task: %d,  Test ACC: %s" % (self._cur_task, str(test_acc))) 
        
        print("After Test Acc: %s" % test_acc)


    def _select_data_for_retention(self):
        """提取所有客户端的数据表征并进行数据选择"""
        all_client_features = []
        all_client_indices = []
        for client_idx in range(self.args["num_users"]):
            try:
                client_features, client_indices = self._extract_client_features(client_idx)
                if len(client_features) > 0:  # 只添加非空的特征
                    all_client_features.append(client_features)
                    all_client_indices.append(client_indices)
            except ValueError as e:
                print(f"Error extracting features for client {client_idx}: {e}")
                continue  # 跳过出错的客户端
        
        # 如果没有提取到任何特征，直接返回
        if len(all_client_features) == 0:
            print("No features extracted from any client. Skipping data selection.")
            return
#         ipdb.set_trace()
        # 在服务器端进行数据选择
        selector = FLDataSelector(self.args["num_users"], all_client_features, int((self._total_classes-self._known_classes) * self.mem_size))
        # 对本地数据进行掩码
        masked_features = selector.mask_local_data()
        # 计算杠杆分数（使用掩码后的数据）
        leverage_scores = selector.compute_local_leverage_scores(masked_features)
        # 聚合杠杆分数并计算采样概率
        p = selector.aggregate_leverage_scores(leverage_scores)
        # 采样数据
        selected_indices = selector.sample_data(p)
        # 保存选中的数据索引
        self.selected_data_indices = selected_indices
#         ipdb.set_trace()
    def _extract_client_features(self, client_idx):
        """提取客户端的数据表征"""
        # 确保 self._known_classes 和 self._total_classes 与本地训练时一致
        if not hasattr(self, "_known_classes") or not hasattr(self, "_total_classes"):
            raise ValueError("_known_classes or _total_classes is not defined.")
        
        # 获取客户端的数据集
        client_dataset = self._get_client_dataset(client_idx)
        local_train_loader = DataLoader(
            client_dataset,
            batch_size=self.args["local_bs"],
            shuffle=False,
            num_workers=self.args["num_worker"],
            pin_memory=True,
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
            drop_last=False
        )
        features = []
        indices = []
        self._network.eval()
        with torch.no_grad():
            for batch_idx, (_, images, labels) in enumerate(local_train_loader):
                images = images.cuda()
                output_list = self._network(images)
                feature = output_list["att"]  # 提取表征
                features.append(feature.cpu())
                start_idx = batch_idx * self.args["local_bs"]
                end_idx = start_idx + images.size(0)
                indices.extend(range(start_idx, end_idx))
               
        features = torch.cat(features, dim=0)
        return features, indices

    def _get_retained_dataset(self, client_idx):
        """获取保留的数据集"""       
        client_dataset = self._get_client_dataset(client_idx)
        retained_dataset = IndexedDataset(client_dataset, self.selected_data_indices[client_idx], self.transform)

        return retained_dataset         
    
    
    def incremental_train(self, data_manager, logger):
        self.logger = logger
        """增量训练过程"""
        setup_seed(self.seed)
        self.data_manager = data_manager  # 初始化 data_manager
        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
                
        self._network.update_fc(self._total_classes)
        self._network.cuda()
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 获取当前任务的数据
        train_dataset = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )          

        test_dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True)
      
        # 初始化 old_loader 和 new_loader
        if self._cur_task > 0:
            old_test_dataset = self.data_manager.get_dataset(
                np.arange(0, self._known_classes), source="test", mode="test"
            )
            self.old_loader = DataLoader(
                old_test_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True
            )
            new_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="test", mode="test"
            )
            self.new_loader = DataLoader(
                new_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True
            )
        
        # 继续训练
        self._fl_train(train_dataset, self.test_loader)
        # 训练完之后选取数据
#         self._select_data_for_retention()
# #         train_dataset = CustomConcatDataset([train_dataset, retained_dataset])
# #         self.retained_dataset = self._get_retained_dataset()
        

    def _fl_train(self, train_dataset, test_loader):
        """联邦学习训练过程"""
        self._network.cuda()
#         ipdb.set_trace()
        self.best_model = None
        self.lowest_loss = np.inf

        prog_bar = tqdm(range(self.args["com_round"]))
        optimizer = torch.optim.SGD(self._network.parameters(), lr=self.args['local_lr'], momentum=0.9, weight_decay=self.args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args["com_round"], eta_min=1e-3)
        # 划分客户端数据        
        user_groups, _ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        self.user_groups = user_groups

        for _, com in enumerate(prog_bar):
            local_weights = []
            local_models = {}
            # m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            # idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            idxs_users = range(self.args["num_users"])
            loss_weight = []
            local_p_list = []
            local_p_label_list = []
            for idx in idxs_users:
                local_train_ds_i = DatasetSplit(train_dataset, self.user_groups[idx])
                if self._cur_task > 0: 
                    print('xxx', local_train_ds_i.labels.shape)
                    local_train_ds_i = CustomConcatDataset([[local_train_ds_i], self.retained_ds_all[idx]], self.transform, self.args, self.logger)      
                    print('####', local_train_ds_i.labels.shape)    
                local_train_loader = DataLoader(local_train_ds_i, batch_size=self.args["local_bs"], shuffle=True, drop_last=False, num_workers=self.args["num_worker"], pin_memory=True, multiprocessing_context=self.args["mulc"], persistent_workers=True)
                # local_dataloader_list.append(local_train_loader)
                if self._cur_task == 0:
                    w, total_loss = self._local_update(copy.deepcopy(self._network), local_train_loader, scheduler.get_last_lr()[0])
                else:
                    w, total_loss = self._local_finetune(self._old_network, copy.deepcopy(self._network), local_train_loader, self._cur_task, idx, scheduler.get_last_lr()[0])

                local_weights.append(copy.deepcopy(w))
                loss_weight.append(total_loss)
                if com == self.args["com_round"] - 1:
                    local_models[idx] = copy.deepcopy(w)
                del local_train_loader, w
                torch.cuda.empty_cache()
            
            scheduler.step()
            sum_loss = sum(loss_weight)
            if sum_loss < self.lowest_loss:
                self.lowest_loss = sum_loss
                self.best_model = copy.deepcopy(self._network.state_dict())

            # 更新全局权重
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if self._cur_task == self.tasks - 1 and com == self.args["com_round"] - 1:
                save_path = "BBBB_model_"+self.args["dataset"]+"_users_"+str(self.args["num_users"])+"_beta_05_localep_2_task_"+str(self._cur_task)+"_memsize_"+str(self.args["mem_size"])+"_round_"+str(com+1)+".pth"
                torch.save(self._network.state_dict(), save_path)
                print(f"Saved final model for last task to {save_path}")

            
            if com % 1 == 0 and com < self.args["com_round"]:
                if self._cur_task > 0:
                    scale = self.args['scale']      
                else:
                    scale = False
                test_acc = self._compute_accuracy(self._network, test_loader, scale=scale, old_classes=self._known_classes)
                if self._cur_task > 0:
                    test_old_acc = self._compute_accuracy(copy.deepcopy(self._network), self.old_loader)
                    test_new_acc = self._compute_accuracy(copy.deepcopy(self._network), self.new_loader)
                    print("Task {}, Test_accy {:.2f} O {} N {}".format(self._cur_task, test_acc, test_old_acc, test_new_acc))
                info = ("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(self._cur_task, com + 1, self.args["com_round"], test_acc))
                self.logger.info(info)
                prog_bar.set_description(info)  
                
        self._select_data_for_retention()
        for idx in idxs_users:
            local_retained_ds = self._get_retained_dataset(idx)
            self.retained_ds_all[idx].append(local_retained_ds)    
        
        del self.best_model
        torch.cuda.empty_cache()
        # torch.save(local_models, f"BBBB_{self.args['dataset']}_local_models_task{self._cur_task}_round{self.args['com_round']}_{datetime.datetime.now().strftime('%Y-%m-%d-%H%M-%S')}.pth")
    
    
    def _local_update(self, model, train_data_loader, lr):
        """本地模型更新"""
        model.train()
        total_loss = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])
#         ipdb.set_trace()
        for it in range(self.args["local_ep"]):
            epoch_loss_collector = []
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):           
                images, labels = images.cuda(), labels.cuda()
                output_list = model(images)
                output = output_list["logits"]
                loss_ce = F.cross_entropy(output, labels)
                loss = loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_collector.append(loss.item())
                if it == 0:
                    total_loss += loss.detach()
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            self.logger.info('Epoch: %d Loss: %f' % (it, epoch_loss))
        
        return model.state_dict(), total_loss

    def _local_finetune(self, teacher, model, train_data_loader, task_id, client_id, lr):
        """本地微调（用于任务 > 0）"""
        model.train()
        teacher.eval()
        total_loss = 0
        class_temperature_dict = {}

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])

        loss_fn = TaskAwareTemperatureLoss(self._known_classes, self.args['tau_old'], self.args['tau_new'], self.args['w_old'], self.args['w_new'])
        for it in range(self.args["local_ep"]):
            epoch_lossce_collector = []
            epoch_losstcl_collector = []
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
#                 ipdb.set_trace()
                images, labels = images.cuda(), labels.cuda()
                output_list = model(images)
                output = output_list["logits"]
                # loss_ce = F.cross_entropy(output, labels)
                loss_ce = loss_fn(output, labels)
                
                loss = loss_ce# + self.args['lambda'] * loss_tcl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_lossce_collector.append(loss_ce.item())
                # epoch_losstcl_collector.append(loss_tcl.item())

                if it == 0:
                    total_loss += loss.detach()
            epoch_lossce = sum(epoch_lossce_collector) / len(epoch_lossce_collector)
            # epoch_losstcl = sum(epoch_losstcl_collector) / len(epoch_losstcl_collector)

            self.logger.info('Epoch: %d Loss CE: %f' % (it, epoch_lossce))    

        return model.state_dict(), total_loss

def generate_sample_per_class(local_data, num_classes):
    sample_per_class = torch.zeros(num_classes, dtype=torch.long)

    for _, data, target in local_data:
        counts = torch.bincount(target, minlength=num_classes)
        sample_per_class += counts

    # Replace zeros with ones to avoid zero counts
    sample_per_class = torch.where(
        sample_per_class > 0,
        sample_per_class,
        torch.ones_like(sample_per_class)
    )

    # Convert tensor to dictionary
    return {cls: count.item() for cls, count in enumerate(sample_per_class)}



class TaskAwareTemperatureLoss(nn.Module):
    def __init__(self, num_old_classes, T_old=0.5, T_new=1.5, w_old=10, w_new=1.0):
        """
        num_old_classes: 当前任务之前的类别数（旧类）
        T_old, T_new: 对旧类 / 新类列的 logits 进行列级缩放
        w_old, w_new: 对旧类 / 新类样本的损失加权
        """
        super().__init__()
        self.num_old_classes = num_old_classes
        self.T_old = T_old
        self.T_new = T_new
        self.w_old = w_old
        self.w_new = w_new

    def forward(self, logits, labels):
        """
        logits: (B, C) - 原始模型输出
        labels: (B,)   - 样本标签
        """
        device = logits.device
        B, C = logits.size()

        # --- Step 1: 按列调温 ---
        old_logits = logits[:, :self.num_old_classes] / self.T_old
        new_logits = logits[:, self.num_old_classes:] / self.T_new
        scaled_logits = torch.cat([old_logits, new_logits], dim=1)  # (B, C)

        # --- Step 2: 按样本进一步调温 ---
        is_old_sample = labels < self.num_old_classes  # (B,)
        sample_temp = torch.where(is_old_sample, self.T_old, self.T_new).to(device)  # (B,)
        sample_temp = sample_temp.view(-1, 1)  # (B, 1)

        scaled_logits = scaled_logits / sample_temp  # 每个样本整体再除一次温度

        # --- Step 3: 按样本加权 ---
        weights = torch.where(is_old_sample, self.w_old, self.w_new).to(device)  # (B,)
        losses = F.cross_entropy(scaled_logits, labels, reduction='none')  # (B,)
        weighted_loss = (losses * weights).mean()

        return weighted_loss



def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 