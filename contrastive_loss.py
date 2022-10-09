import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def likelihood(mu, log_var, tgt_embed, eps=1e-5):
    return ((mu - tgt_embed) ** 2 / (log_var.exp() + eps) - log_var).mean()


def club(mu, log_var, tgt_embed, eps=1e-5):
    random_idx = torch.randperm(mu.size(0)).long().to(mu.device)
    positive = - (mu - tgt_embed) ** 2 / (log_var.exp() + eps)
    negative = - (mu - tgt_embed[random_idx]) ** 2 / (log_var.exp() + eps)
    return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2.


class ProbMLP(nn.Module):
    
    def __init__(self, in_feat, out_feat, mid_feat):
        super().__init__()
        self.log_var = nn.Sequential(nn.Linear(in_feat, mid_feat), nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(in_feat, mid_feat), nn.LeakyReLU())
        self.linear = nn.Linear(mid_feat, out_feat)
        
    @staticmethod
    def reparameterize(mu, log_var, factor=0.2):
        std = log_var.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + factor * std * eps
 
    def forward(self, x):
        log_var = self.log_var(x)
        mu = self.mu(x)
        if self.training:
            embed = self.reparameterize(mu, log_var)
        else:
            embed = mu
        result = self.linear(embed)
        return result, log_var, mu, embed


class Extractor(nn.Module):

    def __init__(self, n_classes, channel=3, image_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3) if channel == 1 else (1, 1)),
            nn.GroupNorm(128, 128, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(128, 128, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(128, 128, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(2048, n_classes)

    def forward(self, x):
        feat = self.model(x)
        logits = self.classifier(feat)
        return logits, feat


