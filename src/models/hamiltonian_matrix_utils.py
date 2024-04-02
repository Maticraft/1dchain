import torch


def get_strip(x: torch.Tensor, offset: int, sites_num: int, fill_mode: str = 'zeros', block_size: int = 4):
    strip = torch.zeros((x.shape[0], x.shape[1], block_size, sites_num*block_size)).to(x.device)
    x_off = abs(offset)*block_size
    for i in range(sites_num):
        idx0 =  i*block_size
        idx1 = idx0 + block_size
        if offset >= 0:
            idx0_off = (idx0 + x_off) % (sites_num * block_size)
            idx1_off = idx0_off + block_size
        else:
            idx0_off = (idx0 - x_off) % (sites_num * block_size)
            idx1_off = idx0_off + block_size
        strip[:, :, :, idx0: idx1] = x[:, :, idx0: idx1, idx0_off: idx1_off]
    if fill_mode == 'zeros':
        if offset > 0:
            strip[:, :, :, -x_off:] = 0.
        elif offset < 0:
            strip[:, :, :, :x_off] = 0.
        return strip
    elif fill_mode == 'circular':
        if offset > 0:
            strip[:, :, :, -x_off:] = strip[:, :, :, :x_off]
        elif offset < 0:
            strip[:, :, :, :x_off] = strip[:, :, :, -x_off:]
        return strip
    elif fill_mode == 'hamiltonian':
        return strip
    else:
        raise ValueError(f'Fill mode: {fill_mode} not implemented')
    

def get_matrix_from_strips(strips: torch.Tensor, sites_num: int, block_size: int = 4):
    matrix = torch.zeros((strips.shape[0], 2, sites_num*block_size, sites_num*block_size)).to(strips.device)
    strips_split = torch.tensor_split(strips, strips.shape[1] // 2, dim=1)
    for i, strip in enumerate(strips_split):
        offset = i - (len(strips_split) // 2)
        matrix_off = abs(offset)*block_size
        for j in range(sites_num):
            idx0 =  j*block_size
            idx1 = (j+1)*block_size
            if offset >= 0:
                matrix_idx0 = (idx0 + matrix_off) % (sites_num*block_size)
                matrix_idx1 = matrix_idx0 + block_size
            else:
                matrix_idx0 = (idx0 - matrix_off) % (sites_num*block_size)
                matrix_idx1 = matrix_idx0 + block_size
            matrix[:, :, idx0: idx1, matrix_idx0: matrix_idx1] = strip[:, :, :, idx0: idx1]
    return matrix
