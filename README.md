# Genetic Programming for Searching Prime Number Function.
遺伝的プログラミングで素数関数を探索する

# run
実行方法

```python
import gp
import prime_number

X, y = prime_number.create(N=10,start_index=0)
test_X, test_y = prime_number.create(N=20,start_index=0)
model = gp.GPRegressor(pop_size=500,printlog=True,max_generation=200)
model.fit(X, y)
print(model.tree.toTexText())
print(model.score(X, y))
print(model.score(test_X, test_y))
```




# Compare with Machine Learning
機械学習との比較
比較手法

- Full Connection Neural Network(=Multi Layer Perceptron)
- Light GBM
- XGBoost
- Random Forest
- Linear Regression

