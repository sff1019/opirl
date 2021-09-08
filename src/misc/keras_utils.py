import tensorflow as tf


def my_reset_states(metric):
    """Resets metric states.

  Args:
    metric: A keras metric to reset states for.
  """
    for var in metric.variables:
        var.assign(0)


def orthogonal_regularization(model, reg_coef=1e-4):
    """Orthogonal regularization v2.

  See equation (3) in https://arxiv.org/abs/1809.11096.

  Args:
    model: A keras model to apply regualization for.
    reg_coef: Orthogonal regularization coefficient.

  Returns:
    A regularization loss term.
  """

    reg = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            prod = tf.matmul(tf.transpose(layer.kernel), layer.kernel)
            reg += tf.reduce_sum(
                tf.math.square(prod * (1 - tf.eye(prod.shape[0]))))
    return reg * reg_coef


def save_keras_model(model, path):
    model.save(path)


def load_keras_model(path):
    return tf.keras.models.load_model(path)
