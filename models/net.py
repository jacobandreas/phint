def _linear(t_in, n_out):
    assert len(t_in.get_shape()) == 2
    v_w = tf.get_variable(
            "w",
            shape=(t_in.get_shape()[1], n_out),
            initializer=tf.uniform_unit_scaling_initializer(
                factor=INIT_SCALE))
    v_b = tf.get_variable(
            "b",
            shape=n_out,
            initializer=tf.constant_initializer(0))
    return tf.einsum("ij,jk->ik", t_in, v_w) + v_b

def _embed(t_in, n_embeddings, n_out):
    v = tf.get_variable(
            "embed", shape=(n_embeddings, n_out),
            initializer=tf.uniform_unit_scaling_initializer())
    t_embed = tf.nn.embedding_lookup(v, t_in)
    return t_embed

