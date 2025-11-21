import pandas as pd
import matplotlib.pyplot as plt  # For graphing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree  # This is the specific model
from sklearn.metrics import accuracy_score

# import of the csv file
data = pd.read_csv("Loan_Prediction.csv")

# start by looking at the first few rows
print(data.head())

# Check if there are missing or invalid cells
print(data.isnull().sum())
# Drop rows or columns with missing data
# data = data.dropna()

# verify data types of each column
print(data.dtypes)

# since the data typer are object, we need to convert them to numerical values because Scikit-learn (the library we are using) cannot read text
# --- PRE-PROCESSING (Cleaning) ---

# A. Drop columns that don't help prediction
# Customer_ID is just a name tag, it doesn't predict loan repayment.
data = data.drop("Customer_ID", axis=1)

# B. Convert the Target (Loan_Status) to numbers manually
# We want Y = 1 (Approved) and N = 0 (Rejected)
data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

# C. Convert all other Text columns to Numbers (One-Hot Encoding)
# This turns "Gender" into "Gender_Male" (1 or 0)
# This turns "Property_Area" into "Property_Area_Urban" (1 or 0), etc.
X = pd.get_dummies(data.drop("Loan_Status", axis=1), drop_first=True)
# using drop_first=True to avoid the redundant columns
# first we see Gender, Married, Education, Self_Employed, Property_Area columns
# after convertion, we see now as Gender_Male, Married_Yes, Education_Not Graduate, Self_Employed_Yes, Property_Area_Semiurban, Property_Area_Urban
# and its value is 1 and 0 only

# D. Define Target
y = data["Loan_Status"]

# now we see again at the data types of each column
print(X.dtypes)


# ---- Model Training ----

# first step is splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Next step is creating the model
# we will use two algorithmns, the ID3 and CART algorithmns
# ID3 algorithm uses entropy and information gain to build the decision tree
# 1. Train with ID3 Logic (Entropy)
model_id3 = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model_id3.fit(X_train, y_train)
acc_id3 = accuracy_score(y_test, model_id3.predict(X_test))

# Cart algorithm uses Gini impurity to build the decision tree
# 2. Train with CART Logic (Gini)
model_cart = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model_cart.fit(X_train, y_train)
acc_cart = accuracy_score(y_test, model_cart.predict(X_test))

print(f"ID3 (Entropy) Accuracy: {acc_id3 * 100:.2f}%")
print(f"CART (Gini) Accuracy:   {acc_cart * 100:.2f}%")

# Lists for comparison
id3_scores = []
cart_scores = []

depths = range(1, 21)

for depth in depths:
    # ID3 Model
    model_id3 = DecisionTreeClassifier(
        criterion="entropy", max_depth=depth, random_state=42
    )
    model_id3.fit(X_train, y_train)
    id3_scores.append(accuracy_score(y_test, model_id3.predict(X_test)))

    # CART Model
    model_cart = DecisionTreeClassifier(
        criterion="gini", max_depth=depth, random_state=42
    )
    model_cart.fit(X_train, y_train)
    cart_scores.append(accuracy_score(y_test, model_cart.predict(X_test)))

# Plot Comparison
plt.figure(figsize=(12, 6))
plt.plot(depths, id3_scores, marker="o", label="ID3 (Entropy)", color="blue")
plt.plot(
    depths, cart_scores, marker="x", label="CART (Gini)", color="red", linestyle="--"
)

plt.title("Head-to-Head: ID3 vs CART Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Test Accuracy")
plt.xticks(depths)
plt.legend()
plt.grid(True)
#this plot shows the accuracy of both models at different tree depths
plt.show()

# --- THE FINAL PRUNED MODEL ---

# 1. Train both models (Pruned to depth 3)
model_id3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model_id3.fit(X_train, y_train)

model_cart = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model_cart.fit(X_train, y_train)

# 2. Create a figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 10))

# 3. Plot ID3 (Entropy) on the Left
plot_tree(model_id3, 
          feature_names=X.columns, 
          class_names=["Rejected", "Approved"], 
          filled=True, 
          fontsize=10,
          ax=axes[0]) # Put this on the first plot
axes[0].set_title("ID3 Algorithm (Entropy)", fontsize=16)

# 4. Plot CART (Gini) on the Right
plot_tree(model_cart, 
          feature_names=X.columns, 
          class_names=["Rejected", "Approved"], 
          filled=True, 
          fontsize=10,
          ax=axes[1]) # Put this on the second plot
axes[1].set_title("CART Algorithm (Gini)", fontsize=16)

# 5. Show
plt.tight_layout()
#this plot shows the structure of both decision trees side by side for comparison after pruned to depth 3
plt.show()

# 1. Get the Importance Scores from your Pruned CART model
# (Ensure you run this AFTER 'model_cart.fit')
importance_scores = model_cart.feature_importances_

# 2. Match the scores with the column names
# We create a Pandas Series to make it easy to sort and plot
features = pd.Series(importance_scores, index=X.columns)

# 3. Sort them from Most Important to Least Important
features = features.sort_values(ascending=True)

# 4. Plot the Horizontal Bar Chart
plt.figure(figsize=(12, 8))
features.plot(kind='barh', color='teal')
plt.title("Feature Importance: Which Columns Actually Matter?")
plt.xlabel("Importance Score (0 to 1)")
plt.grid(axis='x', linestyle='--', alpha=0.7)
#this plot shows the features importance scores from the pruned CART model
plt.show()