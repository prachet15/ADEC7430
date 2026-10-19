import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Make the data (copied from the professor's quadratic script)
theta_vector = np.array([-25,0,0,1,1])

np.random.seed(10)
x1_inputs = np.random.uniform(-10, 10, size=10_000)
np.random.seed(100)
x2_inputs = np.random.uniform(-10, 10, size=10_000)

row_list = []
labels = []

for x1,x2 in zip(x1_inputs,x2_inputs):
    x_i_vector = np.array([1,x1,x2,x1**2,x2**2])
    row_list.append(x_i_vector)
    
    hypothesis_prediction = np.dot(theta_vector, x_i_vector)
    if hypothesis_prediction >= 25:
        labels.append(1)
    else:
        labels.append(0)

data = pd.DataFrame(row_list,columns = ['intercept','x1','x2','x1_squared','x2_squared'])
data['label'] = labels

# Split into train and test, same way the professor did
np.random.seed(123)
test_ids = np.random.choice(np.arange(0, data.shape[0]), size=int(.20*data.shape[0]), replace=False)
test_data = data[data.index.isin(test_ids)]
training_data = data[~data.index.isin(test_ids)]

print("Training rows:", len(training_data))
print("Test rows:", len(test_data))
print("Share of class 1:", data['label'].mean())

# Only give the models x1 and x2
X_train = training_data[['x1', 'x2']]
y_train = training_data['label']
X_test = test_data[['x1', 'x2']]
y_test = test_data['label']

# Baseline: just guess 1 every time
results = {}
results['Baseline (always guess 1)'] = (y_test == 1).mean()

# Logistic regression
logit = LogisticRegression()
logit.fit(X_train, y_train)
results['Logistic regression'] = accuracy_score(y_test, logit.predict(X_test))

print("Logistic coefficients (intercept, x1, x2):", logit.intercept_, logit.coef_)
print("Baseline accuracy:", results['Baseline (always guess 1)'])
print("Logistic accuracy:", results['Logistic regression'])

# KNN with a few different k values
knn_models = {}
for k in [1, 15, 1001]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_models[k] = knn
    results[f'KNN, k = {k}'] = accuracy_score(y_test, knn.predict(X_test))
    print(f"KNN, k = {k} accuracy:", results[f'KNN, k = {k}'])

# Plots: shade where each model predicts 0 or 1, then draw the real circle on top

# grid of points covering the whole square
xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
grid = pd.DataFrame({'x1': xx.ravel(), 'x2': yy.ravel()})

# the real boundary is x1^2 + x2^2 = 50, a circle with radius sqrt(50)
angle = np.linspace(0, 2 * np.pi, 300)
circle_x = np.sqrt(50) * np.cos(angle)
circle_y = np.sqrt(50) * np.sin(angle)

plots = [
    (logit, 'Logistic regression', 'plot_logistic.png'),
    (knn_models[1], 'KNN, k = 1', 'plot_knn_k1.png'),
    (knn_models[15], 'KNN, k = 15', 'plot_knn_k15.png'),
    (knn_models[1001], 'KNN, k = 1001', 'plot_knn_k1001.png'),
]

for model, name, filename in plots:
    pred = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], colors=['lightblue', 'navajowhite'])
    plt.scatter(X_test['x1'], X_test['x2'], c=y_test, cmap='coolwarm', s=5)
    plt.plot(circle_x, circle_y, 'k--', linewidth=2)
    plt.title(f"{name} (accuracy {results[name]:.3f})")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(filename)
    plt.close()
    print("Saved", filename)

# logistic regression again, but this time it also gets the squared columns
cols = ['x1', 'x2', 'x1_squared', 'x2_squared']
logit_sq = LogisticRegression()
logit_sq.fit(training_data[cols], y_train)
results['Logistic + squared terms'] = accuracy_score(y_test, logit_sq.predict(test_data[cols]))

# Put all the accuracies in one table and save it
results_table = pd.DataFrame(list(results.items()), columns=['Model', 'Test accuracy'])
print(results_table)
results_table.to_csv('results.csv', index=False)
