import torch
import torch.nn as nn


class TransformNet(nn.Module):
    """
    Learn a transformation:T(x) = x + delta(x)
    """

    def __init__(self, dim, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)



def sliced_wasserstein(x,y, num_projections=128, p=2): ##  Estimate Sliced Wasserstein Distance between two empirical distributions.

    n = min(x.shape[0], y.shape[0])

    x_idx = torch.randperm(x.shape[0],device=x.device)[:n]

    y_idx = torch.randperm(y.shape[0],device=y.device)[:n]

    x = x[x_idx]
    y = y[y_idx]

    dim = x.shape[1]

    theta = torch.randn(num_projections,dim,device=x.device)

    theta = theta / (theta.norm(dim=1,keepdim=True) + 1e-8)

    x_proj = x @ theta.T
    y_proj = y @ theta.T

    x_sorted, _ = torch.sort(x_proj,dim=0)

    y_sorted, _ = torch.sort(
        y_proj,
        dim=0
    )


    distance = torch.mean(torch.abs(x_sorted - y_sorted) ** p,dim=0)

    return distance.mean() ** (1.0 / p)



def geometry_loss(original,transformed,num_pairs=3000):
    """
    Penalizes changes in pairwise Euclidean distances.

    We want:
        ||T(x_i) - T(x_j)|| ≈ ||x_i - x_j||

    Therefore:
        L = mean(
            (d_transformed - d_original)^2
        )
    """

    n = original.shape[0]


    i = torch.randint(0,n,(num_pairs,),device=original.device)

    j = torch.randint(0,n,(num_pairs,),device=original.device)

    original_dist = torch.norm(original[i] - original[j],dim=1)

    transformed_dist = torch.norm(transformed[i] - transformed[j],dim=1)

    return torch.mean((transformed_dist-original_dist) ** 2)



def train_transform(A, B, use_geometry=True, epochs=2000, lr=1e-3, lambda_structure=0.1, num_projections=128, num_pairs=3000, verbose=True):
    device = A.device
    dim = A.shape[1]

    model = TransformNet(dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        A_transformed = model(A)
        B_transformed = model(B)

        swd = sliced_wasserstein(A_transformed, B_transformed, num_projections=num_projections)

        if use_geometry:
            loss_A = geometry_loss(A, A_transformed, num_pairs=num_pairs)
            loss_B = geometry_loss(B, B_transformed, num_pairs=num_pairs)
            structure = (loss_A + loss_B) / 2
        else:
            structure = torch.tensor(0.0, device=device)

        loss = -swd + lambda_structure * structure

        loss.backward()
        optimizer.step()

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | SWD: {swd.item():.5f} | Structure: {structure.item():.6f} | Loss: {loss.item():.5f}")

    return model


def train_transform(A, B, use_geometry=True, epochs=2000, lr=1e-3, lambda_structure=0.1, num_projections=128, num_pairs=3000, verbose=True):
    """
Minimizes: -SWD(T(A), T(B))+lambda * geometry_loss hence geometric constrain saves from data being modified
    """

    device = A.device
    dim = A.shape[1]

    model = TransformNet(dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        A_transformed = model(A)
        B_transformed = model(B)

        swd = sliced_wasserstein(A_transformed, B_transformed, num_projections=num_projections)

        if use_geometry:
            loss_A = geometry_loss(A, A_transformed, num_pairs=num_pairs)
            loss_B = geometry_loss(B, B_transformed, num_pairs=num_pairs)
            structure = (loss_A + loss_B) / 2
        else:
            structure = torch.tensor(0.0, device=device)

        loss = -swd + lambda_structure * structure

        loss.backward()
        optimizer.step()

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | SWD: {swd.item():.5f} | Structure: {structure.item():.6f} | Loss: {loss.item():.5f}")

    return model