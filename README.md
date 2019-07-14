# Genetic Programming for Searching Prime Number Function.
遺伝的プログラミングで素数関数を探索する

# run
実行方法

```python
import gp
import prime_number

X,y = prime_number.create(N=100,start_index=0)
model = gp.GPRegressor()
model.fit(X,y)
train_MSE = model.score(X,y)
test_MSE = model.score(X,y)
```

# Compare with Machine Learning
機械学習との比較
比較手法

- Full Connection Neural Network(=Multi Layer Perceptron)
- Light GBM
- XGBoost
- Random Forest
- Linear Regression

