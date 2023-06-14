import torch
import torch.nn as nn
from torchvision.models import resnet18


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return(x.view(x.size(0), x.size(1)))

class visible_module(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim = 512):
        super(visible_module, self).__init__()

        self.visible = resnet18(pretrained=True)

        self.fusion_layer = fusion_layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)

        layer0 = [self.visible.conv1, self.visible.bn1, self.visible.relu, self.visible.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.visible.layer1,
                           "layer2": self.visible.layer2, "layer3": self.visible.layer3, "layer4": self.visible.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x, with_features = False):
        for i in range(0, self.fusion_layer):
            if i == 5:
                backbone_feat = x
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                if with_features:
                    return [x_pool, feat, backbone_feat]
                return [x_pool, feat]
            if i < 5:
                x = self.layer_dict["layer" + str(i)](x)

        return x


    def count_params(self):
        s = 0
        for layer in self.layer_dict["layer0"]:
            s += count_parameters(layer)
        for i in range(1, self.fusion_layer):
            s += count_parameters(self.layer_dict["layer" + str(i)])
        return s


class thermal_module(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim = 512):
        super(thermal_module, self).__init__()

        self.thermal = resnet18(pretrained=True)

        self.fusion_layer = fusion_layer

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)

        layer0 = [self.thermal.conv1, self.thermal.bn1, self.thermal.relu, self.thermal.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.thermal.layer1,
                           "layer2": self.thermal.layer2, "layer3": self.thermal.layer3, "layer4": self.thermal.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x, with_features = False):
        for i in range(0, self.fusion_layer):
            if i == 5:
                backbone_feat = x
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                if with_features :
                    return[x_pool, feat, backbone_feat]
                return [x_pool, feat]
            if i < 5:
                x = self.layer_dict["layer" + str(i)](x)
        return x

    def count_params(self):
        s = 0
        for layer in self.layer_dict["layer0"]:
            s += count_parameters(layer)
        for i in range(1, self.fusion_layer):
            s += count_parameters(self.layer_dict["layer" + str(i)])
        return s


class shared_resnet(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim = 512):
        super(shared_resnet, self).__init__()

        self.fusion_layer = fusion_layer

        model_base = resnet18(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.base = model_base

        layer0 = [self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.base.layer1,
                           "layer2": self.base.layer2, "layer3": self.base.layer3, "layer4": self.base.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x):

        for i in range(self.fusion_layer, 6):
            if i < 5 :
                x = self.layer_dict["layer" + str(i)](x)
            else :
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                return [x_pool, feat]

    def count_params(self):
        s = 0
        for i in range(self.fusion_layer, 5):
            s += count_parameters(self.layer_dict["layer" + str(i)])
        return s

class Global_network(nn.Module):
    def __init__(self, class_num, fusion_layer=4, model="", attention_needed=False):
        super(Global_network, self).__init__()

        pool_dim = 512
        self.pool_dim = pool_dim

        self.visible_module = visible_module(fusion_layer=fusion_layer, pool_dim = pool_dim)
        self.thermal_module = thermal_module(fusion_layer=fusion_layer, pool_dim = pool_dim)

        self.model = model

        nb_modalities = 2

        self.fusion_layer = fusion_layer

        self.shared_resnet = shared_resnet(fusion_layer=fusion_layer, pool_dim = pool_dim)

        if model == "concatenation" :
            pool_dim = 2*pool_dim

        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

        self.nb_modalities = nb_modalities

    def forward(self, X, model="concatenation", modality="BtoB"):
        if model == "unimodal":
            if modality == "VtoV" :
                x_pool, feat = self.visible_module(X[0])
            elif modality == "TtoT":
                x_pool, feat= self.thermal_module(X[1])
        else :
            X[0] = self.visible_module(X[0]) # X[0] = (X_pool, feat)
            X[1] = self.thermal_module(X[1]) # X[1] = (X_pool, feat)

            x_pool, feat = torch.cat((X[0][0], X[1][0]), 1), torch.cat((X[0][1], X[1][1]), 1)

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

    def count_params(self, model):
        global s
        # Not adapted to GAN_mmodal
        if model != "unimodal":

            s = self.visible_module.count_params() + self.thermal_module.count_params()
            s += self.shared_resnet.count_params()

            s += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)

            # In this case : Unimodal
        elif model == "unimodal":

            s = self.visible_module.count_params() + self.shared_resnet.count_params() + sum(
                p.numel() for p in self.fc.parameters() if p.requires_grad)
        return s

class MMSF(nn.Module):
    def __init__(self, class_num, mid_stream_block_loc,  pool_dim = 512):
        super(MMSF, self).__init__()

        self.mid_stream_block_loc = mid_stream_block_loc

        self.l2norm = Normalize(2)

        self.modal1 = resnet18(pretrained=True)
        self.center = resnet18(pretrained=True)
        self.modal2 = resnet18(pretrained=True)

        #### Modal1 stream
        layer0 = [self.modal1.conv1, self.modal1.bn1, self.modal1.relu, self.modal1.maxpool]
        self.block0_m1 = nn.Sequential(*layer0)
        self.block1_m1 = self.modal1.layer1
        self.block2_m1 = self.modal1.layer2
        self.block3_m1 = self.modal1.layer3
        self.block4_m1 = self.modal1.layer4

        self.avgpool_m1 = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten_m1 = Flatten()
        self.bottleneck_m1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck_m1.bias.requires_grad_(False)
        self.fc_m1 = nn.Linear(pool_dim, class_num, bias=False)


        #### Center stream
        layer0 = [self.center.conv1, self.center.bn1, self.center.relu, self.center.maxpool]
        # center_sum_pos == 0 means center layers all are used. Otherwise, only after 1 conv block or after 2 etc..
        if not mid_stream_block_loc in [1,2,3,4] :
            self.block0_c = nn.Sequential(*layer0)
        if not mid_stream_block_loc in [2,3,4] :
            self.block1_c = self.center.layer1
        if not mid_stream_block_loc in [3,4] :
            self.block2_c = self.center.layer2
        if not mid_stream_block_loc in [4]:
            self.block3_c = self.center.layer3

        self.block4_c = self.center.layer4 # Last center block is always here

        self.avgpool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten_c = Flatten()
        self.bottleneck_c = nn.BatchNorm1d(pool_dim)
        self.bottleneck_c.bias.requires_grad_(False)
        self.fc_c = nn.Linear(pool_dim, class_num, bias=False)

        #### Modal2 stream
        layer0 = [self.modal2.conv1, self.modal2.bn1, self.modal2.relu, self.modal2.maxpool]
        self.block0_m2 = nn.Sequential(*layer0)
        self.block1_m2 = self.modal2.layer1
        self.block2_m2 = self.modal2.layer2
        self.block3_m2 = self.modal2.layer3
        self.block4_m2 = self.modal2.layer4

        self.avgpool_m2 = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten_m2 = Flatten()
        self.bottleneck_m2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck_m2.bias.requires_grad_(False)
        self.fc_m2 = nn.Linear(pool_dim, class_num, bias=False)

    def forward(self, X):

        # BLOCK 0
        if self.mid_stream_block_loc == 0:
            X[1] = X[0] + X[2] # Center block is the sum of each feature map
        X[0] = self.block0_m1(X[0]) # Modality 1
        X[2] = self.block0_m2(X[2]) # Modality 2
        if hasattr(self, "block0_c"):
            X[1] = self.block0_c(X[1])

        # BLOCK 1
        if self.mid_stream_block_loc == 1 :
            X[1] = X[0] + X[2] # Center block is the sum of each feature map after block 0
        X[0] = self.block1_m1(X[0])
        X[2] = self.block1_m2(X[2])
        if hasattr(self, "block1_c"):
            X[1] = self.block1_c(X[1])

        # BLOCK 2
        if self.mid_stream_block_loc == 2 :
            X[1] = X[0] + X[2] # Center block is the sum of each feature map after block 0
        X[0] = self.block2_m1(X[0])
        X[2] = self.block2_m2(X[2])
        if hasattr(self, "block2_c"):
            X[1] = self.block2_c(X[1])

        # BLOCK 3
        if self.mid_stream_block_loc == 3 :
            X[1] = X[0] + X[2]
        X[0] = self.block3_m1(X[0])
        X[2] = self.block3_m2(X[2])
        if hasattr(self, "block3_c"):
            X[1] = self.block3_c(X[1])

        # BLOCK 4
        if self.mid_stream_block_loc == 4 :
            X[1] = X[0] + X[2]
        X[0] = self.block4_m1(X[0])
        X[1] = self.block4_c(X[1])
        X[2] = self.block4_m2(X[2])

        # AVG POOL + FLATTEN
        x_pool_m1 = self.avgpool_m1(X[0])
        x_pool_c = self.avgpool_c(X[1])
        x_pool_m2 = self.avgpool_m2(X[2])

        x_pool_m1 = self.Flatten_m1(x_pool_m1)
        x_pool_c = self.Flatten_c(x_pool_c)
        x_pool_m2 = self.Flatten_m2(x_pool_m2)

        # Bottleneck
        feat_m1 = self.bottleneck_m1(x_pool_m1)
        feat_c = self.bottleneck_c(x_pool_c)
        feat_m2 = self.bottleneck_m2(x_pool_m2)

        if self.training:
            return  x_pool_m1, self.fc_m1(feat_m1), \
                    x_pool_c, self.fc_c(feat_c), \
                    x_pool_m2, self.fc_m2(feat_m2)
        else:
            return  self.l2norm(torch.concat((x_pool_m1, x_pool_c, x_pool_m2), dim=1)), \
                    self.l2norm(torch.concat((feat_m1, feat_c, feat_m2), dim=1)), \
                    torch.concat((feat_m1, feat_c, feat_m2), dim=1)