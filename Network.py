import numpy as np
import time
from Functions import function
from Layers import Layer


class network:
    def __init__(
        self, n_layers, nodes_per_layer, function_list
    ):  # nodes_per_layer and function_list must be lists
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer
        self.function_list = function_list

    def initialize(self):
        layers = np.full(self.n_layers, None, dtype=object)
        layers[0] = Layer(self.nodes_per_layer[0], 0, 0, "ReLU")

        for i in range(1, self.n_layers):
            prev_nodes = self.nodes_per_layer[i - 1]
            current_nodes = self.nodes_per_layer[i]
            activation = self.function_list[i - 1]

            # Select initialization scheme based on activation
            if activation == "ReLU":
                # He initialization for ReLU
                scale = np.sqrt(2.0 / prev_nodes)
            elif activation == "sigmoid" or activation == "tanh":
                # Xavier/Glorot initialization
                scale = np.sqrt(1.0 / prev_nodes)
            elif activation == "LeakyReLU":
                # He initialization variant for LeakyReLU
                alpha = function(activation).alpha
                scale = np.sqrt(2.0 / (1 + alpha**2)) / np.sqrt(
                    prev_nodes
                )  # alpha=0.01
            else:
                # Default: He initialization
                scale = np.sqrt(2.0 / prev_nodes)

            weight_matrix = (
                np.random.randn(current_nodes, prev_nodes)
            ) * scale  # Shape (current_layer_nodes, previous_layer_nodes)
            bias_matrix = (
                np.random.randn(current_nodes, 1) * 0.01
            )  # Shape (current_layer_nodes, 1)

            layers[i] = Layer(
                current_nodes,
                weight_matrix,
                bias_matrix,
                activation,
            )

        self.layers = layers

    def compute_network(self, Layer1_Value):  # Layer1_Value column numpy array
        # initialize before using
        value = Layer1_Value

        for i in range(1, self.n_layers):
            value = self.layers[i].compute(value, 0)

        return value

    def cost(self, Layer1_Value, Wanted_outut, function_name):
        value = self.compute_network(Layer1_Value)
        return function(function_name).cost_normal(value, Wanted_outut)

    def forward_pass(self, Layer1_Value):
        a = np.full(self.n_layers, None, dtype=object)
        z = np.full(self.n_layers, None, dtype=object)
        value = Layer1_Value
        a[0] = Layer1_Value

        for i in range(1, self.n_layers):
            a[i], z[i] = self.layers[i].compute(value, 1)
            value = a[i]
        return a, z

    def back_prop(
        self, Layer1_Values, Wanted_Output, cost_function
    ):  # Layer1_values and Wanted_output column numpy arrays
        if cost_function != "cross_entropy" and self.function_list[-1] == "SoftMax":
            print(
                "Only cross_entropy can be used when final layer activation is through SoftMax"
            )

        elif cost_function == "cross_entropy":
            a, z = self.forward_pass(Layer1_Values)
            delta = function("cross_entropy").cost_der(z[-1], Wanted_Output)

            delta_w = np.full(self.n_layers, None, dtype=object)
            delta_b = np.full(self.n_layers, None, dtype=object)

            for i in range(self.n_layers - 1):
                delta_w[-1 - i] = np.dot(delta, a[-2 - i].T)
                delta_b[-1 - i] = delta.copy()
                if i != self.n_layers - 2:
                    delta = np.dot(delta.T, self.layers[-1 - i].w).T * self.layers[
                        -2 - i
                    ].function.derivative(z[-2 - i])

            return delta_w, delta_b

        elif cost_function == "MSE" or cost_function == "MAE":
            a, z = self.forward_pass(Layer1_Values)
            delta_a = function(cost_function).cost_der(a[-1], Wanted_Output)
            delta_b = function(self.function_list[-1]).derivative(z[-1])
            delta = delta_a * delta_b

            delta_w = np.full(self.n_layers, None, dtype=object)
            delta_b = np.full(self.n_layers, None, dtype=object)

            for i in range(self.n_layers - 1):
                delta_w[-1 - i] = np.dot(delta, a[-2 - i].T)
                delta_b[-1 - i] = delta.copy()
                if i != self.n_layers - 2:
                    delta = np.dot(delta.T, self.layers[-1 - i].w).T * self.layers[
                        -2 - i
                    ].function.derivative(z[-2 - i])

            return delta_w, delta_b

        else:
            print("No cost function found(Backprop)")

    def batch_prop(
        self, Inputs, Outputs, size, batch, cost_function
    ):  # list of column arrays, Inputs and Outputs
        # safeguards
        assert len(Inputs) == size, "Inputs size mismatch"
        assert len(Outputs) == size, "Outputs size mismatch"
        if batch == 0:
            raise ValueError("Batch size must be greater than 0")

        w_sum = [None] * self.n_layers
        b_sum = [None] * self.n_layers

        for i in range(1, self.n_layers):
            w_sum[i] = np.zeros(
                (self.nodes_per_layer[i], self.nodes_per_layer[i - 1]), dtype=np.float64
            )  # Fixed
            b_sum[i] = np.zeros((self.nodes_per_layer[i], 1), dtype=np.float64)  # Fixed

        for i in range(batch):
            w, b = self.back_prop(Inputs[i], Outputs[i], cost_function)
            for j in range(1, self.n_layers):
                if w[j] is not None:  # Skip input layer
                    w_sum[j] += w[j]
                if b[j] is not None:  # Skip input layer
                    b_sum[j] += b[j]

        return [w / batch if w is not None else None for w in w_sum], [
            b / batch if b is not None else None for b in b_sum
        ]

    def hardmax(self, x):
        if self.function_list[-1] != "SoftMax":
            print("HardMax can only be used when the last layer is SoftMax")
            return None
        hardmax_output = [None] * len(x)
        for i in range(len(x)):
            k = self.compute_network(x[i])
            hardmax_output[i] = np.zeros_like(k)
            (hardmax_output[i])[np.argmax(k)] = 1
        return hardmax_output

    def total_cost(self, input, wanted_output, cost_function):
        if len(input) != len(wanted_output):
            raise ValueError(
                "For cost calculation, the input and wanted putput must have same number of elements"
            )

        if (
            cost_function == "MSE"
            or cost_function == "MAE"
            or cost_function == "cross_entropy"
        ):
            net_cost = 0

            for i in range(len(input)):
                net_cost += self.cost(input[i], wanted_output[i], cost_function)

            return net_cost / len(input)

        elif cost_function == "HardMax":
            hardmax_output = self.hardmax(input)
            percentage = 0
            for i in range(len(input)):
                percentage += int(np.array_equal(hardmax_output[i], wanted_output[i]))

            return percentage / len(input)

    def train(
        self,
        X_train,  # List/array of input samples (column vectors)
        y_train,
        x_valid,
        y_valid,  # List/array of corresponding outputs
        epochs=100,  # Number of training passes
        batch_size=32,  # Mini-batch size
        learning_rate=0.01,
        cost_function="cross_entropy",
        verbose=True,
        update=10,
        cost_display="None",  # Print progress(CHECK LATER)
        clip_gradients=False,  # Enable gradient clipping (CHECK LATER)
        beta1=0.9,  # Adam: exponential decay rate for 1st moment estimates
        beta2=0.999,  # Adam: exponential decay rate for 2nd moment estimates
        epsilon=1e-8,  # Adam: small constant for numerical stability
        lambda_=1e-4,
    ):
        zipped = list(zip(X_train, y_train))

        m_w = [None] * self.n_layers  # 1st moment (mean) for weights
        v_w = [None] * self.n_layers  # 2nd moment (uncentered variance) for weights
        m_b = [None] * self.n_layers  # 1st moment for biases
        v_b = [None] * self.n_layers  # 2nd moment for biases

        for k in range(1, self.n_layers):
            m_w[k] = np.zeros_like(self.layers[k].w)
            v_w[k] = np.zeros_like(self.layers[k].w)
            m_b[k] = np.zeros_like(self.layers[k].b)
            v_b[k] = np.zeros_like(self.layers[k].b)

        bias_correction = 0

        if verbose:
            print("Training started....")

        for i in range(epochs):
            bias_correction += 1

            if i == 0:
                epoch_start = time.time()
            if verbose:
                if cost_display == "None":
                    cost_display = cost_function
                if (i) % update == 0:
                    total_cost_ = self.total_cost(X_train, y_train, cost_display)
                    total_cost_validation = self.total_cost(
                        x_valid, y_valid, cost_display
                    )
                    print(
                        f"Epoch: {i}; Cost_Training = {total_cost_:.4f}; Cost_Validation = {total_cost_validation:.4f}"
                    )

            zipped_copy = zipped.copy()
            np.random.shuffle(zipped_copy)
            shuffled_X, shuffled_y = zip(*zipped_copy)
            j = 0
            while j * batch_size < len(X_train):
                end_bound = min((j + 1) * batch_size, len(shuffled_X))
                X_used = shuffled_X[j * batch_size : end_bound]
                y_used = shuffled_y[j * batch_size : end_bound]
                batch_w, batch_b = self.batch_prop(
                    X_used, y_used, len(X_used), len(X_used), cost_function
                )
                for k in range(1, self.n_layers):
                    if batch_w[k] is not None:  # Skip input layer
                        # Add L2 regularization term to gradient
                        batch_w[k] = batch_w[k] + lambda_ * self.layers[k].w
                        m_w[k] = beta1 * m_w[k] + (1 - beta1) * batch_w[k]
                        v_w[k] = beta2 * v_w[k] + (1 - beta2) * (batch_w[k] ** 2)

                        # Bias-corrected moments
                        m_hat_w = m_w[k] / (1 - beta1**bias_correction)
                        v_hat_w = v_w[k] / (1 - beta2**bias_correction)

                        # Update weights
                        self.layers[k].w -= (
                            learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
                        )

                    if batch_b[k] is not None:
                        # Update moments for biases
                        m_b[k] = beta1 * m_b[k] + (1 - beta1) * batch_b[k]
                        v_b[k] = beta2 * v_b[k] + (1 - beta2) * (batch_b[k] ** 2)

                        # Bias-corrected moments
                        m_hat_b = m_b[k] / (1 - beta1**bias_correction)
                        v_hat_b = v_b[k] / (1 - beta2**bias_correction)

                        # Update biases
                        self.layers[k].b -= (
                            learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
                        )

                j += 1

            if i == 0:
                epoch_end = time.time()
                epoch_time = epoch_end - epoch_start
                est_total = epoch_time * epochs
                print(
                    f"Estimated time: {int(est_total / 60)} mins {est_total - (int(est_total / 60)) * 60} seconds"
                )

        if verbose:
            print("Training ended....")
