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

# Lists for comparison of the algorithms at different depths
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
#this plot shows the accuracy of both models at 20 depths
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

#for advancee pruning, we use cost complexity pruning
#where it will let the tree grow fully and then prunee the last nodes based on the complexity parameter
# 1. Let the tree grow to its full depth (Overfitted)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas  # This gives us a list of possible alpha values
ccp_alphas = ccp_alphas[:-1]  # Remove the maximum alpha (which deletes the whole tree)

# 2. Train a separate model for every single Alpha value
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# 3. Record scores for each model
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# 4. Plot the "Alpha Graph"
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label="Training Accuracy", drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label="Testing Accuracy", drawstyle="steps-post")
plt.xlabel("Alpha (Penalty Score)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Alpha for Training and Testing Sets")
plt.legend()
plt.grid()
plt.show()

# Using the optimal Alpha from your graph
final_model_smart = DecisionTreeClassifier(criterion='entropy', 
                    ccp_alpha=0.011, 
                    random_state=42)
final_model_smart.fit(X_train, y_train)

print("Final model optimized with ccp_alpha=0.011")
# --- FINAL COMPARISON ---

# 1. Calculate Accuracy of the Smart Model
smart_acc = final_model_smart.score(X_test, y_test)

# 2. Print a "Scoreboard" to compare all approaches
print("\n" + "="*40)
print("      FINAL MODEL SHOWDOWN      ")
print("="*40)
print(f"1. ID3 (Max Depth 3):      {acc_id3 * 100:.2f}%")
print(f"2. CART (Max Depth 3):     {acc_cart * 100:.2f}%")
print(f"3. Smart Alpha (0.011):    {smart_acc * 100:.2f}%")
print("="*40)

# 3. Visualize the Smart Alpha Tree
plt.figure(figsize=(15, 8))
plot_tree(final_model_smart, 
        feature_names=X.columns, 
        class_names=["Rejected", "Approved"], 
        filled=True, 
        fontsize=10)
plt.title(f"Final Optimized Tree (Alpha = 0.011) | Accuracy: {smart_acc*100:.2f}%")
plt.show()
#In Smart Pruning, we set a 'performance threshold' called Alpha (0.011).
#The model grows fully, then checks every branch starting from the bottom. 
# If a branch doesn't improve the model's accuracy by at least 1.1% (the Alpha score), 
# we consider it 'unnecessary complexity' and cut it off


#For user Testing data

def predict_new_loan():
    print("\n" + "="*40)
    print("REAL-TIME LOAN PREDICTOR")
    print("="*40)
    
    # 1. Collect Input (Raw Text)
    # We ask for the exact same columns as your CSV
    gender = input("Gender (Male/Female): ")
    married = input("Married (Yes/No): ")
    dependents = input("Dependents (0, 1, 2, 3+): ")
    education = input("Education (Graduate/Not Graduate): ")
    self_employed = input("Self Employed (Yes/No): ")
    app_income = float(input("Applicant Income (e.g., 5000): "))
    coapp_income = float(input("Coapplicant Income (e.g., 2000): "))
    loan_amt = float(input("Loan Amount divided by 1000 (e.g., 150): "))
    term = float(input("Loan Amount Term in months (e.g., 360): "))
    credit = float(input("Credit History (1.0 for Good, 0.0 for Bad): "))
    area = input("Property Area (Urban/Rural/Semiurban): ")

    # 2. Create a DataFrame
    # We map these inputs to the original CSV column names
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'Applicant_Income': [app_income],
        'Coapplicant_Income': [coapp_income],
        'Loan_Amount': [loan_amt],
        'Loan_Amount_Term': [term],
        'Credit_History': [credit],
        'Property_Area': [area]
    })

    # 3. Pre-Process (The Magic Step)
    # Convert text to numbers just like we did for training
    input_prepared = pd.get_dummies(input_data, drop_first=True)

    # CRITICAL: Align columns with the Training Data (X)
    # If the user enters "Male", get_dummies creates "Gender_Male".
    # But if they enter "Female", "Gender_Male" won't exist!
    # .reindex() ensures we have ALL the columns the model expects (X.columns), 
    # and fills missing ones with 0.
    input_prepared = input_prepared.reindex(columns=X.columns, fill_value=0)

    # 4. Predict using the Smart Model
    prediction = final_model_smart.predict(input_prepared)
    probability = final_model_smart.predict_proba(input_prepared)

    # 5. Show Result
    print("\n" + "-"*30)
    if prediction[0] == 1:
        print(f"RESULT: APPROVED ✅")
        print(f"Confidence: {probability[0][1] * 100:.2f}%")
    else:
        print(f"RESULT: REJECTED ❌")
        print(f"Reason: Likely due to Credit History or Income.")
        print(f"Confidence: {probability[0][0] * 100:.2f}%")
    print("-"*30)

# Run the function
predict_new_loan()