from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score


X, Y = datasets.make_regression(n_samples=240000, random_state=123)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=123)


X_train, X_test = X_train.reshape(-1,24,100), X_test.reshape(-1,24,100)
Y_train, Y_test = Y_train.reshape(-1,24), Y_test.reshape(-1,24)


regressor = SGDRegressor()

epochs = 10

for k in range(epochs): ## Number of loops through data
    for i in range(X_train.shape[0]): ## Looping through batches
        X_batch, Y_batch = X_train[i], Y_train[i]
        regressor.partial_fit(X_batch, Y_batch) ## Partially fitting data in batches


Y_test_preds = []
for j in range(X_test.shape[0]): ## Looping through test batches for making predictions
    Y_preds = regressor.predict(X_test[j])
    Y_test_preds.extend(Y_preds.tolist())

print(f'Test MSE      : {mean_squared_error(Y_test.reshape(-1), Y_test_preds)}')
print(f'Test R2 Score : {r2_score(Y_test.reshape(-1), Y_test_preds)}')


Y_train_preds = []
for j in range(X_train.shape[0]): ## Looping through train batches for making predictions
    Y_preds = regressor.predict(X_train[j])
    Y_train_preds.extend(Y_preds.tolist())

print(f'Train MSE      : {mean_squared_error(Y_train.reshape(-1), Y_train_preds)}')
print(f'Train R2 Score : {r2_score(Y_train.reshape(-1), Y_train_preds)}')
