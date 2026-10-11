import numpy as np 



class LinearRegression:
    """Ordinary least squares linear regression.

    Fits a linear model ``y = theta_0 + theta_1 * x_1 + ... + theta_n * x_n``
    by minimizing the mean squared error. The model can be trained with a
    closed-form solution (normal equation) or with several flavors of
    gradient descent.

    Args:
        solver (str, optional): Optimization method. One of:

            - ``'normal'``: closed-form solution using the pseudo-inverse.
            - ``'batch'``: batch gradient descent (uses all samples per step).
            - ``'mini-batch'``: mini-batch gradient descent.
            - ``'stochastic'``: stochastic gradient descent (one sample per step).

            Defaults to ``'normal'``.
        alpha (float, optional): Learning rate used by the gradient descent
            solvers. Must be greater than 0. Ignored by ``'normal'``.
            Defaults to 0.01.
        epochs (int, optional): Number of passes over the training data for
            the gradient descent solvers. Must be a positive integer.
            Defaults to 2000.
        batch_size (int, optional): Number of samples per update when
            ``solver='mini-batch'``. Must be a positive integer.
            Defaults to 32.
        shufle (bool, optional): If True, shuffle the samples at the start of
            every epoch (``'stochastic'`` and ``'mini-batch'`` only).
            Defaults to False.
        random_state (int, optional): Seed for the random number generator
            used when shuffling. Defaults to 42.

    Attributes:
        cost_history_ (list): Cost (MSE / 2) recorded after each epoch.
            Empty for the ``'normal'`` solver.
        intercept_ (float): Learned intercept (available after ``fit``).
        coef_ (numpy.ndarray): Learned feature coefficients, one per feature
            (available after ``fit``).
        n_iter (int): Number of iterations/epochs performed (available after
            ``fit``).

    Raises:
        ValueError: If any constructor argument is invalid.

    Example:
        >>> import numpy as np
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([3, 5, 7, 9])
        >>> model = LinearRegression(solver='normal')
        >>> model.fit(X, y)
        >>> model.predict([[5]])
        array([11.])
        >>> model.score(X, y)
        1.0
    """
    
    # Construction
    def __init__(self, solver='normal',alpha=0.01,epochs=2000, batch_size=32, shufle=False, random_state=42):
        """Create and configure a LinearRegression model.

        Validates every argument and stores it internally. No training is
        done here; call ``fit`` to train the model.

        Args:
            solver (str, optional): ``'normal'``, ``'batch'``,
                ``'mini-batch'`` or ``'stochastic'``. Defaults to ``'normal'``.
            alpha (float, optional): Learning rate (> 0). Defaults to 0.01.
            epochs (int, optional): Number of epochs (positive integer).
                Defaults to 2000.
            batch_size (int, optional): Mini-batch size (positive integer).
                Defaults to 32.
            shufle (bool, optional): Shuffle samples every epoch.
                Defaults to False.
            random_state (int, optional): Seed for shuffling. Defaults to 42.

        Raises:
            ValueError: If ``solver`` is unknown, if ``alpha``, ``epochs`` or
                ``batch_size`` are not positive, if ``epochs``,
                ``batch_size`` or ``random_state`` are not integers, or if
                ``shufle`` is not a boolean.
        """
        # Parameters 
        self.__theta = None 
        self.__solver = solver
        if self.__solver != 'normal' and self.__solver != 'batch' and self.__solver != 'mini-batch' and self.__solver != 'stochastic': 
            raise ValueError(
                "Optimization method doesn't exist. "
                "choose 'normal', 'batch', 'stochastic', 'mini-batch' .")
        self.__alpha =  alpha
        if self.__alpha <= 0:
            raise ValueError("alpha must be greater than 0.")  
        self.__epochs = epochs
        if self.__epochs <= 0:
            raise ValueError("epochs must be greater than 0.")
        elif type(self.__epochs) != int:
            raise ValueError("epochs must have integer value")
        self.__batch_size = batch_size 
        # Validate batch
        if self.__batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0.")
        elif type(self.__batch_size) != int:
            raise ValueError("batch_size must have integer value.")
        self.__shufle = shufle
        if type(self.__shufle) != bool:
            raise ValueError('shufle must have a boolean value (True or False).')
        if type(random_state)!= int:
            raise ValueError('random_state must have a integer value.')
        self.__rng = np.random.default_rng(random_state)
        # Model information 
        self.__n_features_in_ = None 
        self.__n_iter_ = None 
        # Training history
        self.cost_history_ = []

    # =============================
    #      Validate the Input    
    # =============================
    
    @staticmethod
    def __validate_input(X,y):
        """Validate training data and convert it to NumPy arrays.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target values of length n_samples.

        Returns:
            tuple: ``(X, y)`` where ``X`` is a 2D NumPy array and ``y`` is a
            1D NumPy array.

        Raises:
            ValueError: If ``X`` is not 2D, if ``X`` and ``y`` have a
                different number of samples, if ``X`` has no samples or no
                features, or if either contains non-numeric or non-finite
                values.
        """
        
        # converts the feature matrix and target to NumPy array 
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        
        # X must be 2D
        if X.ndim != 2 :
            raise ValueError(
                "X must be a 2D array of shape "
                "(n_samples, n_features).")
            
         # Same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
            
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values.")

        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("y must contain numeric values.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values.")
 
        return X,y
        
    # ===============================================
    # CHECK FITTED 
    # ===============================================

    def __check_fitted(self):
        """Ensure the model has been trained.

        Raises:
            ValueError: If ``fit`` has not been called yet.
        """
        if self.__theta is None:
            raise ValueError(
                "Model must be fitted before"
                "accessing its parameters."
            )

    # ==================================================
    # FIT FUNCTION 
    # ==================================================
    
    def fit(self,X,y):
        """Train the model on the given data.

        Validates the input, adds an intercept column, resets the training
        history, and then optimizes the parameters with the solver chosen at
        construction time.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Target values of length n_samples.

        Returns:
            LinearRegression: The fitted model (``self``), so calls can be
            chained, e.g. ``model.fit(X, y).predict(X_new)``.

        Raises:
            ValueError: If the input data is invalid, or if gradient descent
                diverges (try a smaller ``alpha`` or scale your features).

        Example:
            >>> model = LinearRegression(solver='batch', alpha=0.05, epochs=1000)
            >>> model.fit([[1], [2], [3]], [2, 4, 6])
        """
        
        # Validate and convert inpu 
        X,y = self.__validate_input(X,y)

        # Store number of features before adding incept 
        self.__n_features_in_ =  X.shape[1]
        
        # Add the intercept column 
        X = self.__add_intercept(X)

        # reset Training history 
        self.cost_history_= []
        
        # confirm the optimization method 
        if self.__solver == 'normal' :
            self.__normal_equation(X,y)
            
        elif self.__solver == 'batch':
            self.__batch_gradient_descent(X,y)
            
        elif self.__solver == 'stochastic':
            self.__stochastic_gradient_descent(X,y)

        elif self.__solver == 'mini-batch':
            self.__mini_gradient_descent(X,y)
            
        else : 
            raise ValueError(
                "Optimization method doesn't exist. "
                "choose 'normal', 'batch', 'stochastic', 'mini-batch' .")
            
            
        return self 
    
    # ===============================
    # INITIALIZE THETA 
    # ===============================
    def __initialize_theta(self, X):
        """Initialize the parameter vector ``theta`` with zeros.

        Args:
            X (numpy.ndarray): Design matrix (with intercept column) whose
                number of columns sets the length of ``theta``.
        """
        self.__theta = np.zeros(X.shape[1])

    # ===============================
    # ADD INTERCEPT 
    #================================
    
    @staticmethod
    def __add_intercept(X):
        """Prepend a column of ones to ``X`` for the intercept term.

        Args:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Matrix of shape (n_samples, n_features + 1) whose
            first column is all ones.
        """
        return np.c_[np.ones(X.shape[0]),X]
        

    # ================================
    # PREDICT FUNCTION 
    # ================================
        
    def predict(self, X):
        """Predict target values for new samples.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
                The number of features must match the data used in ``fit``.

        Returns:
            numpy.ndarray: Predicted values of shape (n_samples,).

        Raises:
            ValueError: If the model is not fitted, if ``X`` is not 2D, or if
                the number of features differs from training.

        Example:
            >>> model = LinearRegression().fit([[1], [2], [3]], [2, 4, 6])
            >>> model.predict([[4]])
            array([8.])
        """
        
        # check the fitting 
        self.__check_fitted()
            
        # converts the feature matrix and target to NumPy array 
        X = np.asarray(X)
        
        # check the shape must be 2D
        if X.ndim != 2 :
            raise ValueError(
                    "X must be a 2D array of shape "
                    "(n_samples, n_features)."
                )
            
        # Check number of features
        if X.shape[1] != self.__n_features_in_:
            raise ValueError(
                f"X must contain {self.__n_features_in_} features, "
                f"but received {X.shape[1]}."
            )
            
        # add the intercept to first column X 
        X = np.c_[np.ones(X.shape[0]),X]
        
        return X @ self.__theta 

    
    # ==================================
    # check numerical stability 
    # ==================================
    def __check_numerical_stability(self, value):
        """Detect divergence during training.

        Args:
            value (numpy.ndarray): Array to check (usually ``theta``).

        Raises:
            ValueError: If ``value`` contains NaN or infinite entries.
        """
        if not np.all(np.isfinite(value)):
            raise ValueError(
                "Training diverged due to numerical instability. "
                "Try reducing alpha or scaling your features."
            )
        
    # ===================================
    # Calculate the Gradient 
    # ===================================
    
    def __gradient(self,X,y):
        """Compute the gradient of the cost function.

        Uses ``(1/m) * X.T @ (X @ theta - y)``.

        Args:
            X (numpy.ndarray): Design matrix (with intercept column) of the
                samples to use.
            y (numpy.ndarray): Target values for those samples.

        Returns:
            numpy.ndarray: Gradient with respect to ``theta``, same shape as
            ``theta``.
        """
        
        # calculate the prediction
        y_pred = X @ self.__theta 

        # measure the error 
        error = y_pred - y
        
        m = len(y)
        
        # calculate the gradient 
        dtheta = (1/m)* (X.T @ error)
        
        return dtheta

    # ===========================================
    # COST FUNCTION 
    # ===========================================
    def __cost(self,X,y):
        """Compute the cost (half of the mean squared error).

        Uses ``(1 / (2m)) * sum((X @ theta - y) ** 2)``.

        Args:
            X (numpy.ndarray): Design matrix (with intercept column).
            y (numpy.ndarray): Target values.

        Returns:
            float: The cost value.
        """
        
        y_pred = X @ self.__theta 
        error = y_pred - y
        m = len(y)

        return (1 / (2 * m)) * np.sum(error ** 2)


    # ==================================
    # NORMAL EQUATION 
    # ==================================

    def __normal_equation(self, X, y):
        """Solve for ``theta`` in closed form (normal equation).

        Uses the Moore-Penrose pseudo-inverse, so it still works when
        ``X.T @ X`` is singular (not invertible).

        Args:
            X (numpy.ndarray): Design matrix (with intercept column).
            y (numpy.ndarray): Target values.
        """
        
        # uses the formula theta = (X.T X)^(-1) X.T y
        # I won't use : np.linalg.solve(X.T @ X, X.T @ y)
        # I used pinv to prevent singular matrix error when X isn't invertible 
        self.__theta = np.linalg.pinv(X) @ y
        self.__n_iter_ = 1

        
    # ========================================
    # BATCH GRADDIENT DESCENT 
    # =========================================
    def __batch_gradient_descent(self, X, y):
        """Train with batch gradient descent.

        Every update uses all training samples. The cost is recorded in
        ``cost_history_`` after each epoch.

        Args:
            X (numpy.ndarray): Design matrix (with intercept column).
            y (numpy.ndarray): Target values.

        Raises:
            ValueError: If training diverges.
        """
        self.__initialize_theta(X)
     
        for epochs in range(self.__epochs):
            
            dtheta =  self.__gradient(X,y)
            self.__theta -= self.__alpha * dtheta

            self.__check_numerical_stability(self.__theta)

            cost = self.__cost(X,y)
            self.cost_history_.append(cost)

        self.__n_iter_ = self.__epochs
             
    
    # ===================================================
    # STOCHASTIC GRADDEINT DESCENT 
    # ===================================================
    def __stochastic_gradient_descent(self, X,y):
        """Train with stochastic gradient descent (SGD).

        Updates ``theta`` once per sample. If ``shufle`` is True, the samples
        are reshuffled at the start of every epoch. The cost on the full
        dataset is recorded in ``cost_history_`` after each epoch.

        Args:
            X (numpy.ndarray): Design matrix (with intercept column).
            y (numpy.ndarray): Target values.

        Raises:
            ValueError: If training diverges.
        """
        
        self.__initialize_theta(X)
        m = len(y)
        X_shuffled = X
        y_shuffled = y
        for epochs in range(self.__epochs):
            if self.__shufle:
                indices = self.__rng.permutation(m)
                X_shuffled = X_shuffled[indices]
                y_shuffled = y_shuffled[indices]
            for i in range(m):
                
                X_i = X_shuffled[i:i +1]
                y_i = y_shuffled[i: i+1]

                dtheta = self.__gradient(X_i, y_i)
                self.__theta -= self.__alpha * dtheta

            self.__check_numerical_stability(self.__theta)
            
            cost = self.__cost(X,y)
            self.cost_history_.append(cost)

        self.__n_iter_ = self.__epochs

    # ============================================
    # MINI BATCH GRADIENT ALGORITHM
    # ============================================
    
    def __mini_gradient_descent(self,X,y):
        """Train with mini-batch gradient descent.

        Updates ``theta`` once per mini-batch of ``batch_size`` samples. If
        ``shufle`` is True, the samples are reshuffled at the start of every
        epoch. The cost on the full dataset is recorded in ``cost_history_``
        after each epoch.

        Args:
            X (numpy.ndarray): Design matrix (with intercept column).
            y (numpy.ndarray): Target values.

        Raises:
            ValueError: If training diverges.
        """
        
        self.__initialize_theta(X)
        m = y.shape[0]
        X_shuffled = X
        y_shuffled = y
       
        for epochs in range(self.__epochs):
            if self.__shufle:
                indices = self.__rng.permutation(m)
                X_shuffled = X_shuffled[indices]
                y_shuffled = y_shuffled[indices]
            for start in range(0, m, self.__batch_size):
                end = min(start + self.__batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                dtheta = self.__gradient(X_batch,y_batch)
                self.__theta -= self.__alpha * dtheta
                
            self.__check_numerical_stability(self.__theta)
            
            cost =  self.__cost(X,y)
            self.cost_history_.append(cost)
            
        self.__n_iter_ = self.__epochs
    

                 
    # =======================================
    # MEASURE THE MODEL PERFORMANCE 
    # =======================================
    
    @staticmethod
    def r2_score(y, y_pred):
        """Compute the R² (coefficient of determination) score.

        ``R² = 1 - SS_res / SS_tot``. A score of 1.0 is a perfect fit, 0.0
        means the model is no better than predicting the mean of ``y``, and
        it can be negative for very poor models.

        Args:
            y (array-like): True target values.
            y_pred (array-like): Predicted target values.

        Returns:
            float: The R² score.

        Raises:
            ValueError: If ``y`` has no variance (R² is undefined).
        """
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            raise ValueError("R² score is undefined when y contains no variance.")
            
        return 1 - ss_res / ss_tot


    def score(self, X, y):
        """Return the R² score of the model on the given data.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): True target values of length n_samples.

        Returns:
            float: The R² score of ``predict(X)`` against ``y``.

        Raises:
            ValueError: If the model is not fitted, if ``X`` is invalid, or
                if ``y`` has no variance.
        """
        y_pred = self.predict(X)
        return self.r2_score(y, y_pred)
        
    # ========================================
    # INTERCEPT, COEFFICIENT, AND INTERATIONS 
    # ========================================
    
    @property
    def intercept_(self):
        """float: The learned intercept (bias) term.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        self.__check_fitted()
        return self.__theta[0]

    
    @property
    def coef_(self):
        """numpy.ndarray: The learned coefficients, one per feature.

        Does not include the intercept (see ``intercept_``).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        self.__check_fitted()
        return self.__theta[1:]
    @property
    def n_iter(self):
        """int: Number of iterations (epochs) performed during training.

        Equals 1 for the ``'normal'`` solver, and ``epochs`` for the
        gradient descent solvers.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        self.__check_fitted()
        return self.__n_iter_