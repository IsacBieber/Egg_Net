import torch
import torch.nn as nn
import random
import math
# ================model===================
# define layers and forward process
class Egg(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        # output shape : [2, 1] 
        # out[0, 0] is answer
        # out[0, 1] is useless

    def forward(self, x):
        # x shape : [1, 2]
        return self.trans(x)

# ================loss=====================
# define loss_function
def loss_function(predict, target):
    # both shape [2, 1], but we only care about [0, 0]
    return (predict[0, 0] - target[0, 0]) ** 2

# ==============get a model================
egg = Egg()

# ==============prepare datas==============
x = []
y = []
for i in range(2):
    for j in range(2):
        x.append([[i * 1.0, j * 1.0]])
        y.append([[float(i ^ j), 0.0]])

# ==============train loop=================
# back propagation = backward + optimizer
# backward : calcualte the gradient
# optimizer : use gradient and update

opt = torch.optim.Adam(egg.parameters(), lr=0.05)
for i in range(10000):
    # ==========init gradient===============
    opt.zero_grad()

    # ===========predict(forward)===========
    test_idx = random.randint(0, 3)

    predict = egg(torch.tensor(x[test_idx]))
    target = torch.tensor(y[test_idx])

    # ===========backward===================
    loss = loss_function(predict, target)
    loss.backward()

    # ===========optimize===================
    opt.step()

    if i % 500 == 0:
        print(loss.item())

# ================test loop===================
correct = 0
for i in range(100):
    # ==========init gradient===============
    opt.zero_grad()

    # ===========predict(forward)===========
    test_idx = random.randint(0, 3)

    predict = egg(torch.tensor(x[test_idx]))
    target = torch.tensor(y[test_idx])
    
    if math.sqrt(loss_function(predict, target)) < 1e-5:
        correct = correct + 1
        print("Accept!")
    else:
        print("Wrong Answer!")

print(correct / 100)    

    
