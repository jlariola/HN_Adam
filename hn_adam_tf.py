import tensorflow as tf

class HN_Adam(tf.keras.optimizers.Optimizer):
    """
    Implementation of the HN_Adam algorithm proposed by Reyad, M., Sarhan, A. & Arafa, M.
    A modified Adam algorithm for deep neural network optimization. 
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, name="HN_Adam", **kwargs):
        # FIX: Keras 3 strictly requires learning_rate passed directly to the base class
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables."""
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        
        self._m = []
        self._v = []
        self._v_hat = []
        self._lambda_t0 = []
        
        for var in var_list:
            # FIX: Updated variable naming arguments for Keras 3 compatibility
            self._m.append(self.add_variable_from_reference(reference_variable=var, name="m"))
            self._v.append(self.add_variable_from_reference(reference_variable=var, name="v"))
            self._v_hat.append(self.add_variable_from_reference(reference_variable=var, name="v_hat"))
            
            lambda_init = tf.random.uniform(shape=var.shape, minval=2.0, maxval=4.0, dtype=var.dtype)
            base_name = var.name.split(':')[0] if hasattr(var, 'name') and var.name else "param"
            self._lambda_t0.append(tf.Variable(lambda_init, name=f"{base_name}/lambda_t0", trainable=False))

    # FIX: Keras 3 passes learning_rate dynamically directly into the update step
    def update_step(self, gradient, variable, learning_rate):
        """Performs a single optimization step."""
        if isinstance(gradient, tf.IndexedSlices):
            raise NotImplementedError("HN_Adam does not currently support sparse gradients.")

        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)
        epsilon = tf.cast(self.epsilon, variable.dtype)
        lr = tf.cast(learning_rate, variable.dtype)

        var_key = self._get_variable_index(variable)
        m = self._m[var_key]
        v = self._v[var_key]
        v_hat = self._v_hat[var_key]
        lambda_t0 = self._lambda_t0[var_key]

        m_t_minus_1 = m.value()

        m_t = beta_1 * m_t_minus_1 + (1.0 - beta_1) * gradient
        m.assign(m_t)

        abs_g_t = tf.abs(gradient)
        m_max = tf.maximum(m_t_minus_1, abs_g_t)

        safe_m_max = tf.where(tf.equal(m_max, 0.0), tf.ones_like(m_max), m_max)
        ratio = tf.where(tf.equal(m_max, 0.0), tf.zeros_like(m_t_minus_1), m_t_minus_1 / safe_m_max)
        lambda_t = lambda_t0 - ratio

        pow_grad = tf.pow(abs_g_t, lambda_t)
        v_t = beta_2 * v.value() + (1.0 - beta_2) * pow_grad
        v.assign(v_t)

        abs_v_t = tf.abs(v_t)
        v_hat_t = tf.maximum(v_hat.value(), abs_v_t)
        v_hat.assign(v_hat_t)

        safe_lambda_t = tf.where(tf.equal(lambda_t, 0.0), tf.ones_like(lambda_t), lambda_t)
        inv_lambda_t = tf.where(tf.equal(lambda_t, 0.0), tf.zeros_like(lambda_t), 1.0 / safe_lambda_t)

        denom_amsgrad = tf.pow(v_hat_t, inv_lambda_t) + epsilon
        denom_adam = tf.pow(abs_v_t, inv_lambda_t) + epsilon

        amsgrad_mask = lambda_t < 2.0
        denom = tf.where(amsgrad_mask, denom_amsgrad, denom_adam)

        variable.assign_sub(lr * (m_t / denom))

    def get_config(self):
        """Allows the optimizer to be saved and loaded accurately."""
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
        })
        return config