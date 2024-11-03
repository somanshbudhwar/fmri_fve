import torch
import torch.nn as nn

mps_device = torch.device("cpu")
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)  # ,mean=0,std=1/math.sqrt(100))
        m.bias.data.fill_(0.01)

def col_normalization(x):
    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True)

    x = (x - mean) / std
    return x


class SimpleLinearNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=100, output_size=1, layers=1, sigmoid_constant=1.0):
        super(SimpleLinearNN, self).__init__()
        self.layers = layers
        self.sigmoid_constant = sigmoid_constant
        if self.layers > 0:
            self.stack1 = nn.Sequential(
                nn.Linear(input_size, hidden_size),
            )

        if self.layers > 1:
            self.stack2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
            )
        if self.layers > 2:
            self.stack3 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), )
        if self.layers > 3:
            self.stack4 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
            )
        if self.layers > 4:
            self.stack5 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
            )
        if self.layers > 5:
            self.stack6 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
            )
        if self.layers > 6:
            self.stack7 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
            )
        self.final = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )
        self.activation = nn.Tanh()
        # self.activation = nn.Hardswish()
    def norm_and_act(self, x):
        x = col_normalization(x)
        x = x / self.sigmoid_constant
        x = self.activation(x)
        # x = col_normalization(x)

        return x

    def forward(self, x):
        if self.layers > 0:
            x = self.stack1(x)
            x = self.norm_and_act(x)
        if self.layers > 1:
            x = self.stack2(x)
            x = self.norm_and_act(x)
        if self.layers > 2:
            x = self.stack3(x)
            x = self.norm_and_act(x)
        if self.layers > 3:
            x = self.stack4(x)
            x = self.norm_and_act(x)
        if self.layers > 4:
            x = self.stack5(x)
            x = self.norm_and_act(x)
        if self.layers > 5:
            x = self.stack6(x)
            x = self.norm_and_act(x)
        if self.layers > 6:
            x = self.stack7(x)
            x = self.norm_and_act(x)

        # plt.hist(x.detach().cpu().numpy().reshape(-1, 1))
        # plt.title("Creation of Xdash")
        # plt.show()
        # plt.close()
        x = self.final(x)


        return x


class SimpleLinearNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1):
        super(SimpleLinearNN2, self).__init__()
        self.layers = layers
        if self.layers > 0:
            self.stack1 = nn.Sequential(
                nn.Linear(input_size, hidden_size), )
        if self.layers > 1:
            self.stack2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), )
        if self.layers > 2:
            self.stack3 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), )
        if self.layers > 3:
            self.stack4 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), )
        if self.layers > 4:
            self.stack5 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), )
        if self.layers > 5:
            self.stack6 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), )
        if self.layers > 6:
            self.stack7 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), )

        self.stack_final = nn.Sequential(
            nn.Linear(hidden_size, output_size))
        # self.activation = nn.Hardswish()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.get_xdash(x)
        x = self.stack_final(x)
        return x

    def norm_and_act(self, x):
        # x = col_normalization(x)
        # x = nn.BatchNorm1d(x.shape[-1]).to(mps_device)(x)
        x = self.activation(x)
        # x = nn.BatchNorm1d(x.shape[-1]).to(mps_device)(x)
        return x

    def get_xdash(self, x):
        if self.layers > 0:
            x = self.stack1(x)
            x = self.norm_and_act(x)
        if self.layers > 1:
            x = self.stack2(x)
            x = self.norm_and_act(x)
        if self.layers > 2:
            x = self.stack3(x)
            x = self.norm_and_act(x)
        if self.layers > 3:
            x = self.stack4(x)
            x = self.norm_and_act(x)
        if self.layers > 4:
            x = self.stack5(x)
            x = self.norm_and_act(x)
        if self.layers > 5:
            x = self.stack6(x)
            x = self.norm_and_act(x)
        if self.layers > 6:
            x = self.stack7(x)
            x = self.norm_and_act(x)

        # x = power_transform(x)
        return x