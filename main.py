# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, confusion_matrix
# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # # Load dataset
# # def load_data(file_path):
# #     print("Loading dataset...")
# #     data = pd.read_csv(file_path)
# #     return data

# # # Preprocess dataset
# # def preprocess_data(data):
# #     print("Preprocessing data...")

# #     # Map loan_status to binary values
# #     data['loan_status'] = data['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# #     # Drop rows with missing target values
# #     data = data.dropna(subset=['loan_status'])

# #     # Select relevant columns
# #     relevant_columns = [
# #         'loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', 'annual_inc',
# #         'dti', 'revol_bal', 'revol_util', 'open_acc', 'total_acc', 'mort_acc', 'loan_status'
# #     ]
# #     data = data[relevant_columns]

# #     # One-hot encode categorical variables (grade and sub_grade)
# #     data = pd.get_dummies(data, columns=['grade', 'sub_grade'], drop_first=True)

# #     # Split features and target
# #     X = data.drop('loan_status', axis=1)
# #     y = data['loan_status']

# #     return X, y

# # # Train and evaluate the model
# # def train_and_evaluate(X, y):
# #     print("Training and evaluating the model...")

# #     # Split data into training and test sets
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Train Random Forest model
# #     model = RandomForestClassifier(random_state=42)
# #     model.fit(X_train, y_train)

# #     # Evaluate the model
# #     y_pred = model.predict(X_test)
# #     print("Classification Report:\n", classification_report(y_test, y_pred))
# #     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# #     # Plot confusion matrix
# #     sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
# #     plt.title("Confusion Matrix")
# #     plt.xlabel("Predicted")
# #     plt.ylabel("Actual")
# #     plt.show()

# # # Main execution
# # def main():
# #     file_path = 'lending_club_loan_two.csv'  # Ensure this file is in the same directory as the script

# #     # Load data
# #     data = load_data(file_path)

# #     # Preprocess data
# #     X, y = preprocess_data(data)

# #     # Train and evaluate the model
# #     train_and_evaluate(X, y)

# # if __name__ == "__main__":
# #     main()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# def load_data(file_path):
#     print("Loading dataset...")
#     data = pd.read_csv(file_path)
#     return data

# def preprocess_data(data):
#     print("Preprocessing data...")
#     # Map loan_status to binary
#     data['loan_status'] = data['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
#     data.dropna(subset=['loan_status'], inplace=True)
#     data = pd.get_dummies(data, columns=['grade', 'sub_grade'], drop_first=True)
#     data['revol_util'] = data['revol_util'].fillna(data['revol_util'].median())
#     return data

# def train_and_evaluate(X, y):
#     print("Training and evaluating the model...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print("Classification Report:\n", classification_report(y_test, y_pred))
#     sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()

# def visualize_data(data):
#     print("Visualizing data...")
    
#     # Loan Amount Distribution
#     sns.histplot(data['loan_amnt'], kde=True)
#     plt.title('Distribution of Loan Amounts')
#     plt.xlabel('Loan Amount')
#     plt.ylabel('Frequency')
#     plt.show()
    
#     # Loan Grade vs Loan Amount
#     loan_grade_avg = data.groupby('grade')['loan_amnt'].mean().sort_index()
#     plt.bar(loan_grade_avg.index, loan_grade_avg.values)
#     plt.title('Average Loan Amount by Grade')
#     plt.xlabel('Loan Grade')
#     plt.ylabel('Average Loan Amount')
#     plt.show()

#     # Word Cloud for Purpose
#     if 'purpose' in data.columns:
#         text = ' '.join(data['purpose'].dropna())
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#         plt.figure(figsize=(10, 5))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis('off')
#         plt.title('Most Common Loan Purposes')
#         plt.show()

# def main():
#     file_path = 'lending_club_loan_two.csv'
#     data = load_data(file_path)
#     visualize_data(data)
#     data = preprocess_data(data)
#     X, y = data.drop('loan_status', axis=1), data['loan_status']
#     train_and_evaluate(X, y)

# if __name__ == "__main__":
#     main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def load_data(file_path):
    """Load the dataset."""
    try:
        print("Loading dataset...")
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        return None

def preprocess_data(data):
    """Preprocess the dataset."""
    print("Preprocessing data...")
    try:
        # Map loan_status to binary values
        data['loan_status'] = data['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
        
        # Drop rows with missing target values
        data.dropna(subset=['loan_status'], inplace=True)
        
        # Clean and encode the 'term' column (remove 'months' and convert to integer)
        if 'term' in data.columns:
            data['term'] = data['term'].str.replace(' months', '').str.strip().astype(int)
        
        # Drop irrelevant or high-cardinality columns
        high_cardinality_columns = ['emp_title', 'title', 'zip_code', 'addr_state', 'desc']
        for col in high_cardinality_columns:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)

        # Select categorical and numeric columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

        # One-hot encode categorical columns with low cardinality
        for col in categorical_columns:
            if data[col].nunique() < 50:  # Only encode columns with fewer than 50 unique values
                data = pd.get_dummies(data, columns=[col], drop_first=True)
            else:
                data.drop(col, axis=1, inplace=True)  # Drop high-cardinality columns
        
        # Fill missing values for numeric columns
        for col in numeric_columns:
            data[col] = data[col].fillna(data[col].median())

        print("Preprocessing completed successfully.")
        return data
    except KeyError as e:
        print(f"Key error during preprocessing: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        return None

def visualize_data(data):
    """Generate visualizations for exploratory data analysis."""
    print("Generating visualizations...")
    try:
        # Loan Amount Distribution
        sns.histplot(data['loan_amnt'], kde=True)
        plt.title('Distribution of Loan Amounts')
        plt.xlabel('Loan Amount')
        plt.ylabel('Frequency')
        plt.show()

        # Loan Grade vs Loan Amount
        if 'grade' in data.columns:
            loan_grade_avg = data.groupby('grade')['loan_amnt'].mean().sort_index()
            plt.bar(loan_grade_avg.index, loan_grade_avg.values)
            plt.title('Average Loan Amount by Grade')
            plt.xlabel('Loan Grade')
            plt.ylabel('Average Loan Amount')
            plt.show()

        # Word Cloud for Loan Purposes
        if 'purpose' in data.columns:
            text = ' '.join(data['purpose'].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Loan Purposes')
            plt.show()
    except Exception as e:
        print(f"An error occurred during visualization: {e}")

def train_and_evaluate(data):
    """Train and evaluate a Random Forest model."""
    print("Training and evaluating the model...")
    try:
        # Splitting data into features and target
        X = data.drop('loan_status', axis=1)
        y = data['loan_status']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation metrics
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Confusion matrix
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        print("Model training and evaluation completed successfully.")
    except Exception as e:
        print(f"An error occurred during model training or evaluation: {e}")

def main():
    file_path = 'lending_club_loan_two.csv'
    
    # Load dataset
    data = load_data(file_path)
    if data is None:
        return
    
    # Visualize data
    visualize_data(data)
    
    # Preprocess data
    data = preprocess_data(data)
    if data is None:
        return
    
    # Train and evaluate the model
    train_and_evaluate(data)

if __name__ == "__main__":
    main()
