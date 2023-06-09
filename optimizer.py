import torch.optim as optim
from TransREID.solver.make_optimizer import make_optimizer as make_optimizer_vit
from LMBN.optim.make_optimizer import make_optimizer as make_optimizer_LMBN

def optimizer_and_params_management(model, net, lr):

    if model == "transreid":
        optimizer = make_optimizer_vit(net)
    elif model == "LMBN":
        optimizer = make_optimizer_LMBN(net)
    elif "MMSF" in model:
        ignored_params = list(map(id, net.bottleneck_m1.parameters())) \
                         + list(map(id, net.bottleneck_c.parameters())) \
                         + list(map(id, net.bottleneck_m2.parameters())) \
                         + list(map(id, net.fc_m1.parameters())) \
                         + list(map(id, net.fc_c.parameters())) \
                         + list(map(id, net.fc_m2.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, net.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': net.bottleneck_m1.parameters(), 'lr': lr},
            {'params': net.bottleneck_c.parameters(), 'lr': lr},
            {'params': net.bottleneck_m2.parameters(), 'lr': lr},
            {'params': net.fc_m1.parameters(), 'lr': lr},
            {'params': net.fc_c.parameters(), 'lr': lr},
            {'params': net.fc_m2.parameters(), 'lr': lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    else: # unimodal / concatenation
        ignored_params = list(map(id, net.visible_module.layer_dict["layer7"].parameters())) \
                         + list(map(id, net.thermal_module.layer_dict["layer7"].parameters())) \
                         + list(map(id, net.fc.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': net.visible_module.layer_dict["layer7"].parameters(), 'lr': lr},
            {'params': net.thermal_module.layer_dict["layer7"].parameters(), 'lr': lr},
            {'params': net.fc.parameters(), 'lr': lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    return optimizer