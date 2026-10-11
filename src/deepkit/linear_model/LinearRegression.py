import numpy as np 



class LinearRegression:
    
    # Construction
    def __init__(self, solver='normal',alpha=0.01,epochs=2000, batch_size=32, shufle=False, random_state=42):
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
        if self.__theta is None:
            raise ValueError(
                "Model must be fitted before"
                "accessing its parameters."
            )

    # ==================================================
    # FIT FUNCTION 
    # ==================================================
    
    def fit(self,X,y):
        
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
        self.__theta = np.zeros(X.shape[1])

    # ===============================
    # ADD INTERCEPT 
    #================================
    
    @staticmethod
    def __add_intercept(X):
        return np.c_[np.ones(X.shape[0]),X]
        

    # ================================
    # PREDICT FUNCTION 
    # ================================
        
    def predict(self, X):
        
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
        if not np.all(np.isfinite(value)):
            raise ValueError(
                "Training diverged due to numerical instability. "
                "Try reducing alpha or scaling your features."
            )
        
    # ===================================
    # Calculate the Gradient 
    # ===================================
    
    def __gradient(self,X,y):
        
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
        
        y_pred = X @ self.__theta 
        error = y_pred - y
        m = len(y)

        return (1 / (2 * m)) * np.sum(error ** 2)


    # ==================================
    # NORMAL EQUATION 
    # ==================================

    def __normal_equation(self, X, y):
        
        # uses the formula theta = (X.T X)^(-1) X.T y
        # I won't use : np.linalg.solve(X.T @ X, X.T @ y)
        # I used pinv to prevent singular matrix error when X isn't invertible 
        self.__theta = np.linalg.pinv(X) @ y
        self.__n_iter_ = 1

        
    # ========================================
    # BATCH GRADDIENT DESCENT 
    # =========================================
    def __batch_gradient_descent(self, X, y):
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
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            raise ValueError("R² score is undefined when y contains no variance.")
            
        return 1 - ss_res / ss_tot


    def score(self, X, y):
        y_pred = self.predict(X)
        return self.r2_score(y, y_pred)
        
    # ========================================
    # INTERCEPT, COEFFICIENT, AND ITERATIONS 
    # ========================================
    
    @property
    def intercept_(self):
        self.__check_fitted()
        return self.__theta[0]

    
    @property
    def coef_(self):
        self.__check_fitted()
        return self.__theta[1:]
    @property
    def n_iter(self):
        self.__check_fitted()
        return self.__n_iter_


    
    
