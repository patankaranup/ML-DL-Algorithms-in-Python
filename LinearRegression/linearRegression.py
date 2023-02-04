import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('./data.csv')

# print(data)
# plt.scatter(data.TV, data.Sales)
# plt.show()


# Also known as mean squared error 
def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].TV
        y = points.iloc[i].Sales
        # y is actual value and y predicted is (m*x + b) and squaring the error 
        total_error = (y - (m*x + b))**2
    # gettting the mean of the error 
    total_error / float(len(points))


# gradient descent also known for optimzing the learning curve
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].TV
        y = points.iloc[i].Sales

        # this is partial derivative function behind the scene maths 
        m_gradient += -(2/n) * x * (y-(m_now *x + b_now))
        b_gradient += -(2/n) * (y-(m_now *x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m,b

m = 0
b = 0
L = 0.0001 # the low the leraning rate the more accurate the model is but it takes a lot of time
epochs = 500

for i in range(epochs):
    if i % 50 == 0:
        print("Epochs : ", i)
    m,b = gradient_descent(m,b,data,L)

print(m,b)


plt.scatter(data.TV, data.Sales, color= "black")
plt.plot(list(range(20,80)),[m*x+b for x in range(20,80)], color="red")
plt.show()