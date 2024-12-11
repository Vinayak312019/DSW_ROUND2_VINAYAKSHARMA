HI i am vinayak sharma please feel from to view my file. I have explained every steps which i have done on my file with its code. model and visualization, performance matrix and all eda andsteps are executed in single file only. still for ur reference i have put my model code here too. 

class LoanDefaultModel:
    def __init__(self, data_path):
        """Initialize the model class with the path to the training data."""
        self.data = pd.read_excel(data_path)
        self.X = None
        self.y = None
        self.model = None

    def load(self):
        """Load the data and prepare it for modeling."""
        # Encode categorical features using LabelEncoder
        label_encoders = {}
        for col in ['sub_grade', 'home_ownership', 'purpose', 'verification_status', 'application_type']:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            label_encoders[col] = le

        # Drop non-feature columns and prepare feature matrix and target variable
        self.X = self.data.drop(['customer_id', 'transaction_date', 'loan_status'], axis=1)
        self.y = self.data['loan_status']
        self.X = pd.get_dummies(self.X, drop_first=True)

        return self

    def preprocess(self):
        """Scale numerical features and split the data into training and test sets."""
        # Standardize the numerical features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        # Split data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

    def train(self, model_type='logistic_regression'):
        """Train the model based on the specified type."""
        if model_type == 'logistic_regression':
            # Initialize and train a Logistic Regression model
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            # Initialize and train a Random Forest Classifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.model.fit(self.X_train, self.y_train)
        return self

    def test(self):
        """Evaluate the model on the test set and print metrics."""
        predictions = self.model.predict(self.X_test)
        print("Classification Report:")
        print(classification_report(self.y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, predictions))
        print("ROC AUC Score:")
        print(roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1]))

    def predict(self, new_data):
        """Predict using the trained model on new data."""
        return self.model.predict(new_data)
