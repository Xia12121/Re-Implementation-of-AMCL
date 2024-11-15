import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightly
import torch.nn.functional as F
from src.util import knn_predict
from .amcl_components.adaptive_temperature import AdaptiveTemperature
from .amcl_components.multi_head_projector import MultiHeadProjector

knn_k = 200
knn_t = 0.1
classes = 10


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback
    At the end of every training epoch we create a feature bank by inferencing
    the backbone on the dataloader passed to the module. 
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the 
    feature_bank features from the train data.
    We can access the highest accuracy during a kNN prediction using the 
    max_accuracy attribute.
    """

    def __init__(self, dataloader_kNN, epochs):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.epochs = epochs

    def training_epoch_end(self, outputs):
        # print losses
        losses = [i['loss'].item() for i in outputs]
        loss_avg = sum(losses)/len(losses)
        print(f'Epoch {self.current_epoch+1}/{self.epochs}: train loss = {loss_avg:.2f}')

        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                if torch.cuda.is_available():
                    img = img.cuda()
                    target = target.cuda()
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(
            self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(
            self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(
                feature, self.feature_bank, self.targets_bank, classes, knn_k, knn_t)
            num = images.size(0)
            top1 = (pred_labels[:, 0] == targets).float().sum().item()
            return (num, top1)

    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_top1 = 0.
            for (num, top1) in outputs:
                total_num += num
                total_top1 += top1
            acc = float(total_top1 / total_num)
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            print(f'Epoch {self.current_epoch+1}/{self.epochs}: KNN acc = {100*acc:.2f}')
            # self.log('kNN_accuracy', acc * 100.0, prog_bar=True)

class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simclr model based on ResNet
        self.resnet_simclr = \
            lightly.models.SimCLR(self.backbone, num_ftrs=512)  # add a 2-layer projection head
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(x0, x1)
        # self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_simclr.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 更新学习率，通常在每个训练 step 中调用
        scheduler.step(self.current_epoch)

class AMCLSimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs, num_heads=3, input_dim=512, output_dim=128):
        super().__init__(dataloader_kNN, epochs)
        self.num_heads = num_heads

        # Create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # Create multi-head projection layers
        self.multi_head_projector = MultiHeadProjector(input_dim, output_dim, num_heads=self.num_heads)

        # Adaptive temperature module for scaling similarities
        self.adaptive_temp = AdaptiveTemperature(input_dim=output_dim, num_heads=self.num_heads)

        # Define contrastive loss (NT-Xent)
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        # Extract features using the ResNet backbone
        features = self.backbone(x)
        # Pass the features through the multi-head projector
        projections = self.multi_head_projector(features.flatten(start_dim=1))
        return projections

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        
        # Forward pass for both views (x0 and x1)
        proj_x0 = self(x0)
        proj_x1 = self(x1)

        # Compute adaptive temperature-scaled similarities for each head
        scaled_similarities, reg_term = self.adaptive_temp(proj_x0, proj_x1)

        # Compute the NT-Xent loss using the scaled similarities for each head
        total_loss = 0.0
        for head_idx in range(self.num_heads):
            loss = -torch.log(torch.exp(scaled_similarities[head_idx]) / (
                torch.exp(scaled_similarities[head_idx]) + torch.exp(-scaled_similarities[head_idx])))
            total_loss += loss.mean()

        # Add the regularization term for the temperature
        total_loss += reg_term

        self.log('train_loss_ssl', total_loss)
        return total_loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 更新学习率，通常在每个训练 step 中调用
        scheduler.step(self.current_epoch)

class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs, memory_bank_size=4096):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a moco model based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(self.backbone, num_ftrs=512,
                                m=0.99, batch_shuffle=True)
        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*self.resnet_moco(x0, x1))
        loss_2 = self.criterion(*self.resnet_moco(x1, x0))
        loss = 0.5 * (loss_1 + loss_2)
        # self.log('train_loss_ssl', loss)
        return loss
        
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_moco.parameters(),
            lr=0.08,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 更新学习率，通常在每个训练 step 中调用
        scheduler.step(self.current_epoch)

class AMCLMocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs, memory_bank_size=4096, num_heads=3, input_dim=512, output_dim=128):
        super().__init__(dataloader_kNN, epochs)
        self.num_heads = num_heads
        self.memory_bank_size = memory_bank_size

        # 创建Resnet backbone网络并且移除classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # 创建多头投影层
        self.multi_head_projector = MultiHeadProjector(input_dim, output_dim, num_heads=self.num_heads)

        # 自适应温度模块用于缩放相似度
        self.adaptive_temp = AdaptiveTemperature(input_dim=output_dim, num_heads=self.num_heads)

        # 基于 ResNet 创建带记忆库的 MoCo 模型
        self.resnet_moco = \
            lightly.models.MoCo(self.backbone, num_ftrs=512,
                                m=0.99, batch_shuffle=True)
        
        # 使用记忆库定义 NT-Xent 损失（对比损失）
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        # 使用 ResNet 主干提取特征
        features = self.backbone(x)
        # 将特征传入多头投影层
        projections = self.multi_head_projector(features.flatten(start_dim=1))
        return projections

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        
        # Forward pass for both views (x0 and x1)
        proj_x0 = self(x0)
        proj_x1 = self(x1)

        # Compute adaptive temperature-scaled similarities for each head
        scaled_similarities, reg_term = self.adaptive_temp(proj_x0, proj_x1)

        # Compute the NT-Xent loss using the scaled similarities for each head
        total_loss = 0.0
        for head_idx in range(self.num_heads):
            loss = -torch.log(torch.exp(scaled_similarities[head_idx]) / (
                torch.exp(scaled_similarities[head_idx]) + torch.exp(-scaled_similarities[head_idx])))
            total_loss += loss.mean()

        # Add the regularization term for the temperature
        total_loss += reg_term

        # self.log('train_loss_ssl', total_loss)
        return total_loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=0.08,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 更新学习率（在每个训练 step 中调用）
        scheduler.step(self.current_epoch)

class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(
                self.backbone, num_ftrs=512, num_mlp_layers=2)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        loss = self.criterion(x0, x1)
        # self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=0.05,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 更新学习率，通常在每个训练 step 中调用
        scheduler.step(self.current_epoch)

class AMCLSimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs, num_heads=3, input_dim=512, output_dim=128):
        super().__init__(dataloader_kNN, epochs)
        self.num_heads = num_heads
        
        # 创建 ResNet 主干并移除分类头
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # 创建多头投影层
        self.multi_head_projector = MultiHeadProjector(input_dim, output_dim, num_heads=self.num_heads)

        # 自适应温度模块，用于缩放相似度
        self.adaptive_temp = AdaptiveTemperature(input_dim=output_dim, num_heads=self.num_heads)

        # 基于 ResNet 创建 SimSiam 模型
        self.resnet_simsiam = lightly.models.SimSiam(
            self.backbone, num_ftrs=512, num_mlp_layers=2
        )

        # 使用 SymNegCosineSimilarityLoss 损失函数
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        # 使用 ResNet 主干提取特征
        features = self.backbone(x)
        # 将特征传入多头投影层
        projections = self.multi_head_projector(features.flatten(start_dim=1))
        return projections

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        
        # 对两个视图（x0 和 x1）进行前向传播
        proj_x0 = self(x0)
        proj_x1 = self(x1)

        # 计算每个头的自适应温度缩放相似度
        scaled_similarities, reg_term = self.adaptive_temp(proj_x0, proj_x1)

        # 使用缩放相似度计算对比损失
        total_loss = 0.0
        for head_idx in range(self.num_heads):
            loss = -torch.log(torch.exp(scaled_similarities[head_idx]) / (
                torch.exp(scaled_similarities[head_idx]) + torch.exp(-scaled_similarities[head_idx])))
            total_loss += loss.mean()

        # 添加温度的正则化项
        total_loss += reg_term

        return total_loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 更新学习率（在每个训练 step 中调用）
        scheduler.step(self.current_epoch)

class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        device = z_a.device
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD
        N = z_a.size(0)
        D = z_a.size(1)
        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD
        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        return loss

class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        # note that BarlowTwins has the same architecture
        self.resnet_simsiam = \
            lightly.models.SimSiam(
                self.backbone, num_ftrs=512, num_mlp_layers=3)
        self.criterion = BarlowTwinsLoss(lambda_param=5e-3)

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        # our simsiam model returns both (features + projection head)
        z_a, _ = x0
        z_b, _ = x1
        loss = self.criterion(z_a, z_b)
        # self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=0.11,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 更新学习率，通常在每个训练 step 中调用
        scheduler.step(self.current_epoch)

class AMCLBarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs, num_heads=3, input_dim=512, output_dim=128, lambda_param=5e-3):
        super().__init__(dataloader_kNN, epochs)
        self.num_heads = num_heads
        
        # 创建 ResNet 主干并移除分类头
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # 创建多头投影层
        self.multi_head_projector = MultiHeadProjector(input_dim, output_dim, num_heads=self.num_heads)

        # 自适应温度模块，用于缩放相似度
        self.adaptive_temp = AdaptiveTemperature(input_dim=output_dim, num_heads=self.num_heads)

        # 基于 ResNet 创建 SimSiam 模型（架构与 Barlow Twins 相同）
        self.resnet_simsiam = lightly.models.SimSiam(
            self.backbone, num_ftrs=512, num_mlp_layers=3
        )

        # 使用 Barlow Twins Loss 损失函数
        self.criterion = BarlowTwinsLoss(lambda_param=lambda_param)

    def forward(self, x):
        # 使用 ResNet 主干提取特征
        features = self.backbone(x)
        # 将特征传入多头投影层
        projections = self.multi_head_projector(features.flatten(start_dim=1))
        return projections

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        
        # 对两个视图（x0 和 x1）进行前向传播
        proj_x0 = self(x0)
        proj_x1 = self(x1)

        # 计算每个头的自适应温度缩放相似度
        scaled_similarities, reg_term = self.adaptive_temp(proj_x0, proj_x1)

        # 使用 Barlow Twins 的方式计算对比损失
        total_loss = 0.0
        for head_idx in range(self.num_heads):
            loss = self.criterion(scaled_similarities[head_idx], scaled_similarities[head_idx])
            total_loss += loss.mean()

        # 添加温度的正则化项
        total_loss += reg_term

        return total_loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=0.11,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # 更新学习率（在每个训练 step 中调用）
        scheduler.step(self.current_epoch)