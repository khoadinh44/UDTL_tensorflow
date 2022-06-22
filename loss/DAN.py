import tensorflow as tf

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.shape[1]) + int(target.shape[1])
    total = tf.concat([source, target], axis=0)
    
    total0 = tf.expand_dims(total, axis=1)
    total0 = tf.tile(total0, (1, total.shape[1], 1, 1))
    total1 = tf.expand_dims(total, axis=2)
    total1 = tf.tile(total1, (1, 1, total.shape[1], 1))
    
    L2_distance = ((total0-total1)**2) + 2.
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.math.reduce_sum(L2_distance) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [tf.math.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return tf.math.reduce_sum(kernel_val, axis=0)#/len(kernel_val)


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.shape[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss
