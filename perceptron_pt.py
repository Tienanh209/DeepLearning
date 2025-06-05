import torch
import torch.nn as nn

class MyDenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad=True))
    def forward(self, inputs):
        z = torch.matmul(inputs, self.W) + self.b
        output = torch.sigmoid(z)
        return output

model = nn.Sequential(
    MyDenseLayer(input_dim=3, output_dim=1)  # output_dim=1 for binary output
)

# Hours studied, hours slept, practice tests
student_data = torch.tensor([[7.0, 7.0, 4.0]], dtype=torch.float32)

output = model(student_data)

prob = output.detach().numpy()[0][0]

print(f"Passing: {round(prob * 100)}%")
print(f"Failing: {round((1 - prob) * 100)}%")
