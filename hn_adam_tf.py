import tensorflow as tf

class HN_Adam(tf.keras.optimizers.Optimizer):
    """
    Implementation of the HN_Adam algorithm proposed by Reyad, M., Sarhan, A. & Arafa, M.
    A modified Adam algorithm for deep neural network optimization. 
    Neural Comput & Applic 35, 17095-17112 (2023).
    https://doi.org/10.1007/s00521-023-08568-z
    
    This implementation strictly adheres to Algorithm 2 and Section 5 of the paper.
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, name="HN_Adam", **kwargs):
        super().__init__(name=name, **kwargs)
        # 1: Initialized parameter \theta_0, step size \eta
        # 2: Exponential decay rates \beta_1, \beta_2, \epsilon
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        
        self._m = []
        self._v = []
        self._v_hat = []
        self._lambda_t0 = []
        
        for var in var_list:
            # Initialize: m_0 = 0, v_0 = 0, amsgrad = False, v_hat(0) = 0
            self._m.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))
            self._v.append(self.add_variable_from_reference(model_variable=var, variable_name="v"))
            self._v_hat.append(self.add_variable_from_reference(model_variable=var, variable_name="v_hat"))
            
            # Section 5: "The threshold value of the norm (\Lambda_{t0}) is randomly 
            # chosen in the range from 2 to 4". Initialized once per parameter tensor.
            lambda_init = tf.random.uniform(shape=var.shape, minval=2.0, maxval=4.0, dtype=var.dtype)
            
            # Extract base name cleanly for variable naming
            base_name = var.name.split(':')[0] if hasattr(var, 'name') and var.name else "param"
            self._lambda_t0.append(tf.Variable(lambda_init, name=f"{base_name}/lambda_t0", trainable=False))

    def update_step(self, gradient, variable):
        """Performs a single optimization step."""
        if isinstance(gradient, tf.IndexedSlices):
            raise NotImplementedError("HN_Adam does not currently support sparse gradients.")

        # Cast hyperparameters to match variable dtype
        lr = tf.cast(self.learning_rate, variable.dtype)
        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)
        epsilon = tf.cast(self.epsilon, variable.dtype)

        # Retrieve tracked states for the current variable
        var_key = self._get_variable_index(variable)
        m = self._m[var_key]
        v = self._v[var_key]
        v_hat = self._v_hat[var_key]
        lambda_t0 = self._lambda_t0[var_key]

        # Save m_{t-1} before updating to compute m_max and Lambda(t)
        m_t_minus_1 = m.value()

        # 6: m_t <- \beta_1 * m_{t-1} + (1 - \beta_1) * g_t  // moving Average
        m_t = beta_1 * m_t_minus_1 + (1.0 - beta_1) * gradient
        m.assign(m_t)

        # 7: m_max <- Max(m_{t-1}, |g_t|)
        abs_g_t = tf.abs(gradient)
        m_max = tf.maximum(m_t_minus_1, abs_g_t)

        # 8: \Lambda(t) <- \Lambda_{t0} - (m_{t-1} / m_max)
        # Handle edge case: zero values in m_max to avoid division by zero
        safe_m_max = tf.where(tf.equal(m_max, 0.0), tf.ones_like(m_max), m_max)
        ratio = tf.where(tf.equal(m_max, 0.0), tf.zeros_like(m_t_minus_1), m_t_minus_1 / safe_m_max)
        lambda_t = lambda_t0 - ratio

        # 9: v_t <- \beta_2 * v_{t-1} + (1 - \beta_2) * (|g_t|)^{\Lambda(t)}
        pow_grad = tf.pow(abs_g_t, lambda_t)
        v_t = beta_2 * v.value() + (1.0 - beta_2) * pow_grad
        v.assign(v_t)

        # 10: If \Lambda(t) < 2: // Switching Between Adam and AMSgrad
        # 12: v_{hat(t)} <- Max(v_{hat(t-1)}, |v_t|)
        abs_v_t = tf.abs(v_t)
        v_hat_t = tf.maximum(v_hat.value(), abs_v_t)
        v_hat.assign(v_hat_t)

        # Calculate safe inverse of Lambda(t) to prevent division by zero edge case
        safe_lambda_t = tf.where(tf.equal(lambda_t, 0.0), tf.ones_like(lambda_t), lambda_t)
        inv_lambda_t = tf.where(tf.equal(lambda_t, 0.0), tf.zeros_like(lambda_t), 1.0 / safe_lambda_t)

        # Denominator for AMSGrad branch: (v_{hat(t)}^{1/\Lambda(t)}) + \epsilon
        denom_amsgrad = tf.pow(v_hat_t, inv_lambda_t) + epsilon
        
        # Denominator for Adam branch: (v_t^{1/\Lambda(t)}) + \epsilon
        denom_adam = tf.pow(abs_v_t, inv_lambda_t) + epsilon

        # 14: else (amsgrad = False)
        # Select denominator based on switching condition element-wise
        amsgrad_mask = lambda_t < 2.0
        denom = tf.where(amsgrad_mask, denom_amsgrad, denom_adam)

        # 13 & 16: \theta_t <- \theta_{t-1} - \eta * (m_t / denom)
        variable.assign_sub(lr * (m_t / denom))

    def get_config(self):
        """Allows the optimizer to be saved and loaded accurately."""
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
        })
        return config