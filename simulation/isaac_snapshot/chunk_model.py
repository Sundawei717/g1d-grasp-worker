import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.downsample = (nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False),
                                        nn.BatchNorm2d(outgoing))
                           if incoming != outgoing or stride != 1 else None)

    def forward(self, inputs):
        residual = self.downsample(inputs) if self.downsample is not None else inputs
        features = self.relu(self.bn1(self.conv1(inputs)))
        return self.relu(self.bn2(self.conv2(features)) + residual)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256))
        self.layer4 = nn.Sequential(ResidualBlock(256, 512, 2), ResidualBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)


class SplitPromptStem(nn.Module):
    """The RGB and prompt kernels share one conv1 operation and have distinct optimizer groups."""

    def __init__(self, pretrained, prompt_channels=2):
        super().__init__()
        self.rgb_weight = nn.Parameter(pretrained.weight.detach().clone())
        self.prompt_weight = nn.Parameter(pretrained.weight.new_zeros((64, prompt_channels, 7, 7)))
        self.in_channels = 3 + prompt_channels

    def forward(self, image):
        weight = torch.cat((self.rgb_weight, self.prompt_weight), dim=1)
        return F.conv2d(image, weight, stride=2, padding=3)


class SharedResNet18ChunkTransformer(nn.Module):
    def __init__(self, state_dim=None, prompt_meta_dim=6, horizon=8, weights_path=None, *,
                 fixture_random_init=False, wrist_channels=None, v3_final=False, freeze_bn=False,
                 prompt_fusion=False, point_queries=False, query_bbox=False, split_wrist_box=False):
        super().__init__()
        self.v3_final = v3_final
        self.freeze_bn = freeze_bn
        self.prompt_fusion = prompt_fusion
        self.point_queries = point_queries
        self.query_bbox = query_bbox
        if split_wrist_box and (not v3_final or wrist_channels != 4):
            raise ValueError("The split box kernel requires the V3-Final four-channel right wrist")
        if query_bbox and not point_queries:
            raise ValueError("Box-conditioned queries require point queries")
        if point_queries and not prompt_fusion:
            raise ValueError("Decoder point queries require explicit point geometry")
        if prompt_fusion:
            if not v3_final or prompt_meta_dim != 6:
                raise ValueError("Prompt fusion requires the V3-Final geometry contract")
            prompt_meta_dim = 10
        state_dim = (17 if v3_final else 8) if state_dim is None else state_dim
        if v3_final and (state_dim != 17 or wrist_channels not in (3, 4) or horizon != 8):
            raise ValueError("V3-Final requires a 17D state, three camera views and eight actions")
        if weights_path is None and not fixture_random_init:
            raise ValueError("ImageNet ResNet-18 weights are required; random weights are only for fixtures")
        base = ResNet18()
        self.weight_sha256 = None
        if weights_path is not None:
            path = Path(weights_path)
            self.weight_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            base.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        old = base.conv1

        def stem(channels):
            layer = nn.Conv2d(channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                              padding=old.padding, bias=False)
            with torch.no_grad():
                layer.weight[:, :3].copy_(old.weight)
                layer.weight[:, 3:].zero_()
            return layer

        self.wrist_channels = wrist_channels
        if v3_final:
            self.head_stem = SplitPromptStem(old) if prompt_fusion else stem(5)
            if prompt_fusion:
                self.head_prompt_fusion = nn.Conv2d(258, 256, 1)
                self.head_prompt_norm = nn.LayerNorm(256)
                with torch.no_grad():
                    # Preserve the feature routing and give both prompt maps a nonzero direct path.
                    self.head_prompt_fusion.weight[:, :256].copy_(torch.eye(256)[:, :, None, None])
                    nn.init.normal_(self.head_prompt_fusion.weight[:, 256:], std=0.05)
                    self.head_prompt_fusion.bias.zero_()
            with torch.random.fork_rng(devices=[]):
                self.left_wrist_stem = stem(3)
                self.right_wrist_stem = SplitPromptStem(old, prompt_channels=1) if split_wrist_box else stem(wrist_channels)
            self.backbone = nn.Sequential(base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3)
            self.position_grid = nn.Parameter(torch.empty(14, 14, 256))
            positions = (self.position_grid,)
        else:
            base.conv1 = stem(5)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.spatial_projection = nn.Conv2d(512, 256, 1)
            self.positions_row = nn.Parameter(torch.empty(7, 128))
            self.positions_column = nn.Parameter(torch.empty(7, 128))
            positions = (self.positions_row, self.positions_column)
        if not v3_final and wrist_channels is not None:
            if wrist_channels not in (3, 4):
                raise ValueError("Wrist input requires RGB with an optional mask")
            with torch.random.fork_rng(devices=[]):
                self.wrist_stem = stem(wrist_channels)
        self.view_embedding = nn.Parameter(torch.empty(2 if wrist_channels is None else 3, 256))
        self.state_projection = nn.Sequential(nn.Linear(state_dim * 2, 128), nn.GELU(), nn.Linear(128, 256))
        self.prompt_projection = nn.Sequential(nn.Linear(prompt_meta_dim, 128), nn.GELU(), nn.Linear(128, 256))
        encoder = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.1, activation="gelu",
                                             batch_first=True, norm_first=True)
        decoder = nn.TransformerDecoderLayer(256, 4, 1024, dropout=0.1, activation="gelu",
                                             batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 2, nn.LayerNorm(256), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder, 4, nn.LayerNorm(256))
        self.future_query = nn.Parameter(torch.empty(horizon, 256))
        if point_queries:
            self.query_point_projection = nn.Sequential(nn.Linear(8 if query_bbox else 4,128),nn.GELU(),nn.Linear(128,256))
        self.output = nn.Linear(256, 8 if v3_final else state_dim)
        for value in (*positions, self.view_embedding, self.future_query):
            nn.init.normal_(value, std=0.02)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("rgb_scale", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.state_dim = state_dim
        self.prompt_meta_dim = prompt_meta_dim
        self.horizon = horizon
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def visual_tokens(self, global_rgb_prompt, local_rgb_prompt, right_wrist=None):
        batch = global_rgb_prompt.shape[0]
        if global_rgb_prompt.shape != (batch, 5, 224, 224):
            raise ValueError("Expected five channels for the head view")
        def pixels(view):
            return torch.cat(((view[:, :3] - self.rgb_mean) / self.rgb_scale, view[:, 3:]), dim=1)

        if self.v3_final:
            if (local_rgb_prompt.shape != (batch, 3, 224, 224) or right_wrist is None
                    or right_wrist.shape != (batch, self.wrist_channels, 224, 224)):
                raise ValueError("V3-Final accepts RGB on left and RGB with an optional mask on right")
            encoded = self.backbone(torch.cat((self.head_stem(pixels(global_rgb_prompt)),
                                               self.left_wrist_stem(pixels(local_rgb_prompt)),
                                               self.right_wrist_stem(pixels(right_wrist))), dim=0))
        elif self.wrist_channels is None:
            if local_rgb_prompt.shape != global_rgb_prompt.shape or right_wrist is not None:
                raise ValueError("Expected two views of the legacy head camera")
            encoded = self.backbone(pixels(torch.cat((global_rgb_prompt, local_rgb_prompt), dim=0)))
        else:
            shape = (batch, self.wrist_channels, 224, 224)
            if local_rgb_prompt.shape != shape or right_wrist is None or right_wrist.shape != shape:
                raise ValueError("Wrist image shape differs from the checkpoint")
            head = self.backbone[0](pixels(global_rgb_prompt))
            wrists = self.wrist_stem(pixels(torch.cat((local_rgb_prompt, right_wrist), dim=0)))
            encoded = self.backbone[1:](torch.cat((head, wrists), dim=0))
        if self.v3_final:
            if self.prompt_fusion:
                head = encoded[:batch]
                grid = head.shape[-2:]
                bbox = F.adaptive_avg_pool2d(global_rgb_prompt[:, 3:4], grid)
                point = F.adaptive_max_pool2d(global_rgb_prompt[:, 4:5], grid)
                aligned = torch.cat((head, bbox.to(head.dtype), point.to(head.dtype)), dim=1)
                head_tokens = self.head_prompt_norm(self.head_prompt_fusion(aligned).relu().flatten(2).transpose(1, 2))
                wrists = encoded[batch:].flatten(2).transpose(1, 2)
                features = torch.cat((head_tokens, wrists), dim=0)
            else:
                features = encoded.flatten(2).transpose(1, 2)
            position = self.position_grid.reshape(1, 196, 256)
        else:
            features = self.spatial_projection(encoded).flatten(2).transpose(1, 2)
            position = torch.cat((self.positions_row[:, None, :].expand(7, 7, 128),
                              self.positions_column[None, :, :].expand(7, 7, 128)), dim=-1).reshape(1, 49, 256)
        return torch.cat([values + position + self.view_embedding[index] for index, values in enumerate(features.split(batch))], dim=1)

    def forward(self, global_rgb_prompt, local_rgb_prompt, recent_state, prompt_meta, right_wrist=None):
        batch = global_rgb_prompt.shape[0]
        if recent_state.shape != (batch, 2, self.state_dim) or prompt_meta.shape != (batch, self.prompt_meta_dim):
            raise ValueError("State history or prompt metadata shape mismatch")
        tokens = self.visual_tokens(global_rgb_prompt, local_rgb_prompt, right_wrist)
        state_tokens = self.state_projection(recent_state.reshape(batch, -1))[:, None, :]
        prompt_tokens = self.prompt_projection(prompt_meta)[:, None, :]
        memory = self.encoder(torch.cat((tokens, state_tokens, prompt_tokens), dim=1))
        queries = self.future_query[None].expand(batch,-1,-1)
        if self.point_queries:
            # Convert absolute coordinates to centered normalized coordinates.
            point = torch.cat((prompt_meta[:,6:8]*2-1, prompt_meta[:,8:]*2),dim=-1)
            if self.query_bbox:
                center = prompt_meta[:,:2]
                half_extent = prompt_meta[:,2:4] / 2
                box = torch.cat((center-half_extent,center+half_extent),dim=-1)
                point = torch.cat((box*2-1,point),dim=-1)
            queries = queries + self.query_point_projection(point)[:,None,:]
        decoded = self.decoder(queries,memory)
        return self.output(decoded)


def masked_chunk_loss(prediction, normalized_command, valid, width_weight=1.0):
    if (prediction.shape != normalized_command.shape or prediction.ndim != 3
            or valid.shape != prediction.shape[:2] or valid.dtype != torch.bool or width_weight < 0):
        raise ValueError("Invalid command loss dimensions or mask")
    if not valid.any():
        raise ValueError("Batch has no trainable commands")
    values = F.smooth_l1_loss(prediction[valid], normalized_command[valid], reduction="none")
    return values[:, :-1].mean() + width_weight * values[:, -1].mean()
