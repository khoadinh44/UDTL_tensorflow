import tensorflow as tf

def CORAL(source, target):
    d = tf.cast(source.shape[1], tf.float32)

    # source covariance
    xm = tf.math.reduce_mean(source, 0) - source
    xc = tf.linalg.matmul(xm, xm, transpose_a=True)

    # target covariance
    xmt = tf.math.reduce_mean(target, 0) - target
    xct = tf.linalg.matmul(xmt, xmt, transpose_a=True)

    # frobenius norm between source and target
    loss = tf.math.reduce_mean(tf.linalg.matmul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return 
