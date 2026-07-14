import torch
import torch.nn as nn
import random
# ================model===================
# define layers and forward process
class Egg(nn.Module):
    def __init__(self):
        super().__init__()
        self.normal = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        # x shape : [1, 2]
        return self.normal(x)

# ================init weights=============
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.5, 0.5)
        nn.init.uniform_(m.bias, -0.5, 0.5)

# ================loss=====================
# define loss_function
def loss_function(predict, target):
    # both shape [2, 1], but we only care about [0, 0]
    return (predict[0, 0] - target[0, 0]) ** 2

# ==============get a model================
device = "cpu"
egg = Egg()
egg.to(device)
egg.apply(init_weights)

# ==============prepare datas==============
x = []
y = []
for i in range(2):
    for j in range(2):
        x.append([[i * 1.0, j * 1.0]])
        y.append([[float(i ^ j)]])

# ==============train loop=================
# back propagation = backward + optimizer
# backward : calcualte the gradient
# optimizer : use gradient and update
opt = torch.optim.Adam(egg.parameters(), lr=0.01)
for i in range(10000):
    # ==========init gradient===============
    opt.zero_grad()

    # ===========predict(forward)===========
    test_idx = random.randint(0, 3)

    X = torch.tensor(x[test_idx]).to(device)
    Y = torch.tensor(y[test_idx]).to(device)

    predict = egg(X)
    target = Y

    # ===========backward===================
    loss = loss_function(predict, target)
    loss.backward()

    # ===========optimize===================
    opt.step()

    if i % 500 == 0:
        print(loss.item())

# ================test loop===================
correct = 0
for i in range(4):
# ================predict(forward)============
    predict = egg(torch.tensor(x[i]))
    target = torch.tensor(y[i])
    
    if torch.sqrt(loss_function(predict, target)) < 1e-5:
        correct = correct + 1
        print("Accept!")
    else:
        print("Wrong Answer!")

print(correct / 4)    

    
