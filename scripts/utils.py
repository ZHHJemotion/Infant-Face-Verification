import math
import torch
import torch.nn as nn
import torch.utils.data as data


def accuracy(output, target, topk=(1,)):
    """
        Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0]


def init_weights(model, pre_trained=None):
    """
        The initialization for network:
            step 1: initializing the whole net --> weight: gaussian of std = 0.01, bias = 0
            step 2: initializing the base net by using the weights from pre-trained model
            step 3: initializing the cls_layer of subnet in RetinaNet --> bias = - log((1-pi)/pi)
    """

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            # nn.init.constant(m.bias, 0.1)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 0.0)
            nn.init.constant(m.bias, 0.0)

    if pre_trained is not None:  # initializing the base net by pre-trained model
        pre_weight = torch.load(pre_trained)
        # prefix = "module.fpn.base_net."

        model_dict = model.state_dict()
        model_dict_tolist = list(model_dict.items())
        count = 0
        for key, value in pre_weight.items():
            if "feature" in key:
                layer_name, weights = model_dict_tolist[count]
                model_dict[layer_name] = value
                count += 1

        # pretrained_dict = {(prefix + k): v for k, v in pre_weight.items() if (prefix + k) in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


def get_mean_and_std(dataset, max_load=10000):
    '''Compute the mean and std value of dataset.'''
    # dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i in range(N):
        print(i)
        im, _, _ = dataset.load(1)
        for j in range(3):
            mean[j] += im[:, j, :, :].mean()
            std[j] += im[:, j, :, :].std()
    mean.div_(N)
    std.div_(N)
    return mean, std


if __name__ == "__main__":
    dataset = '/home/pingguo/ril-server/PycharmProject/database/faceDatasetCV/face_31&11/train/*.png'

    # mean, std = get_mean_and_std(dataset)
    # print(mean, std)