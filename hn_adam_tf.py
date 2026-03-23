import tensorflow as tf

class HN_Adam(tf.keras.optimizers.Optimizer):
    """
    Implementation of the HN_Adam (Hybrid and Adaptive Norming of Adam with AMSGrad) algorithm.
    
    Paper: Reyad, M., Sarhan, A. & Arafa, M. (2023)
    "A modified Adam algorithm for deep neural network optimization"
    Neural Comput & Applic 35, 17095–17112
    https://doi.org/10.1007/s00521-023-08568-z
    
    HN_Adam implements Algorithm 2 from the paper, which combines adaptive exponent learning
    with conditional switching between AMSGrad and standard Adam optimization modes.
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, name="HN_Adam", **kwargs):
        """
        Initialize HN_Adam optimizer.
        
        Args:
            learning_rate (float): Step size η in Algorithm 2. Default: 0.001
            beta_1 (float): Exponential decay rate for first moment. Default: 0.9
            beta_2 (float): Exponential decay rate for second moment. Default: 0.999
            epsilon (float): Small constant ε for numerical stability. Default: 1e-8
            name (str): Name of the optimizer. Default: "HN_Adam"
            **kwargs: Additional keyword arguments for base Optimizer class.
        """
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """
        Initialize optimizer state variables for Algorithm 2.
        
        Per Algorithm 2 pseudocode:
        - m: First moment estimate (initialized to 0)
        - v: Second moment estimate (initialized to 0)
        - v_hat: Maximum of second moment (initialized to 0, used in AMSGrad mode)
        - lambda_t0: Adaptive exponent parameter (initialized uniformly in [2.0, 4.0])
        """
        if hasattr(self, "_built") and self._built:
            return
            
        super().build(var_list) 
        
        self._m = []
        self._v = []
        self._v_hat = []
        self._lambda_t0 = []
        
        for var in var_list:
            self._m.append(self.add_variable_from_reference(reference_variable=var, name="m"))
            self._v.append(self.add_variable_from_reference(reference_variable=var, name="v"))
            self._v_hat.append(self.add_variable_from_reference(reference_variable=var, name="v_hat"))
            
            l_t0 = self.add_variable_from_reference(reference_variable=var, name="lambda_t0")
            l_t0.assign(tf.random.uniform(shape=var.shape, minval=2.0, maxval=4.0, dtype=var.dtype))
            self._lambda_t0.append(l_t0)
            
        self._built = True

    def update_step(self, gradient, variable, learning_rate):
        """Performs a single optimization step following Algorithm 2."""
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

        # Algorithm 2, Step 6: Compute first moment (moving average)
        # m_t ← β₁ · m_{t-1} + (1-β₁) · g_t
        m_t_minus_1 = m
        m_t = beta_1 * m_t_minus_1 + (1.0 - beta_1) * gradient
        m.assign(m_t)

        # Algorithm 2, Step 7: Compute maximum of previous first moment and absolute gradient
        # m_max ← Max(m_{t-1}, |g_t|)
        abs_g_t = tf.abs(gradient)
        m_max = tf.maximum(m_t_minus_1, abs_g_t)

        # Algorithm 2, Step 8: Compute adaptive exponent Λ(t)
        # Λ(t) ← Λ₁₀ - (m_{t-1} / m_max)
        # Handle edge case where m_max = 0 to avoid division by zero
        safe_m_max = tf.where(tf.equal(m_max, 0.0), tf.ones_like(m_max), m_max)
        ratio = tf.where(tf.equal(m_max, 0.0), tf.zeros_like(m_t_minus_1), m_t_minus_1 / safe_m_max)
        lambda_t = lambda_t0 - ratio

        # Algorithm 2, Step 9: Compute second moment with adaptive exponent
        # v_t ← β₂ · v_{t-1} + (1-β₂) · (|g_t|)^Λ(t)
        pow_grad = tf.pow(abs_g_t, lambda_t)
        v_t = beta_2 * v + (1.0 - beta_2) * pow_grad
        v.assign(v_t)

        # Algorithm 2, Step 10-16: Conditional update based on Λ(t) < 2.0
        # Switching between AMSGrad (Λ(t) < 2.0) and Adam (Λ(t) ≥ 2.0) modes
        amsgrad_mask = lambda_t < 2.0
        
        # Algorithm 2, Step 12: Update v_hat only in AMSGrad mode (when Λ(t) < 2.0)
        # v_hat(t) ← Max(v_hat(t-1), |v_t|)
        abs_v_t = tf.abs(v_t)
        v_hat_conditional = tf.where(amsgrad_mask, tf.maximum(v_hat, abs_v_t), v_hat)
        v_hat.assign(v_hat_conditional)

        # Compute denominators for both modes
        # Handle edge case where λ_t = 0 to avoid 1/0 issues
        safe_lambda_t = tf.where(tf.equal(lambda_t, 0.0), tf.ones_like(lambda_t), lambda_t)
        inv_lambda_t = tf.where(tf.equal(lambda_t, 0.0), tf.zeros_like(lambda_t), 1.0 / safe_lambda_t)

        # Algorithm 2, Step 13: AMSGrad denominator
        # (v_hat(t)^(1/Λ(t))) + ε
        denom_amsgrad = tf.pow(v_hat_conditional, inv_lambda_t) + epsilon
        
        # Algorithm 2, Step 16: Adam denominator
        # (v_t^(1/Λ(t))) + ε
        denom_adam = tf.pow(abs_v_t, inv_lambda_t) + epsilon

        # Select denominator based on Λ(t) < 2.0 condition
        denom = tf.where(amsgrad_mask, denom_amsgrad, denom_adam)

        # Parameter update: θ_t ← θ_{t-1} - η · m_t / denom
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