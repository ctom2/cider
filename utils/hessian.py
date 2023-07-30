import torch

def forward_diff(data, step, dim):
    # data --- input image (gpu array!!!)
    # step
    # dim --- determine which is the dimension to calculate derivate
    # dim = 0 --> z axis
    # dim = 1 --> y axis
    # dim = 2 --> x axis

    assert dim <= 2
    r, n, m = data.shape
#     size = torch.tensor((r, n, m))
    size = (r,n,m)
    size_ = (r + 1,n + 1,m + 1)

    position = torch.zeros(3, dtype = torch.float32).cuda()

    temp1 = torch.zeros(size_, dtype = torch.float32).cuda()
    temp2 = torch.zeros(size_, dtype = torch.float32).cuda()

    size = torch.tensor(size)

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp1[int(position[0]):int(size[0]), int(position[1]):int(size[1]), int(position[2]):int(size[2])] = data
    temp2[int(position[0]):int(size[0]), int(position[1]):int(size[1]), int(position[2]):int(size[2])] = data

    size[dim] = size[dim] - 1
    temp2[0:size[0], 0:size[1], 0:size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] + 1

    out = temp1[int(position[0]):int(size[0]), int(position[1]):int(size[1]), int(position[2]):int(size[2])]
    return -out


def back_diff(data, step, dim):
    # data --- input image(gpu array!!!)
    # step
    # dim --- determine which is the dimension to calculate derivate
    # dim = 0 --> z axis
    # dim = 1 --> y axis
    # dim = 2 --> x axis
    assert dim <= 2
    r, n, m = data.shape
#     size = torch.tensor((r, n, m))
    size = (r,n,m)
    size_ = (r + 1,n + 1,m + 1)
    position = torch.zeros(3, dtype=torch.float32).cuda()
    temp1 = torch.zeros(size_, dtype=torch.float32).cuda()
    temp2 = torch.zeros(size_, dtype=torch.float32).cuda()

    temp1[int(position[0]):int(size[0]), int(position[1]):int(size[1]), int(position[2]):int(size[2])] = data
    temp2[int(position[0]):int(size[0]), int(position[1]):int(size[1]), int(position[2]):int(size[2])] = data

    size = torch.tensor(size)

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp2[int(position[0]):int(size[0]), int(position[1]):int(size[1]), int(position[2]):int(size[2])] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] - 1
    out = temp1[0:int(size[0]), 0:int(size[1]), 0:int(size[2])]
    return out


def get_gxx(g):
    return back_diff(forward_diff(g, 1, 1), 1, 1).cuda()


def get_gxy(g):
    return forward_diff(forward_diff(g, 1, 1), 1, 2).cuda()


def get_gyy(g):
    return back_diff(forward_diff(g, 1, 2), 1, 2).cuda()


def hessian_loss(data):

    gxx = get_gxx(data)
    gxy = get_gxy(data)
    gyy = get_gyy(data)

    return gxx.abs().sum() + gyy.abs().sum() + 2*gxy.abs().sum()