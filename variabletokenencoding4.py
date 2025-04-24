import torch
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# Simulate a single Term in the additive series
class Term(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.randn(1))
        self.weight = nn.Parameter(torch.randn(1))
        self.route = nn.Parameter(torch.randn(3))  # forward, backward, sideways

    def forward(self, target, neighbor=None):
        route_weights = F.softmax(self.route, dim=0)  # ensure they sum to 1
        fwd = route_weights[0] * self.weight * self.value
        bwd = -route_weights[1] * self.weight * self.value
        side = route_weights[2] * neighbor.weight * neighbor.value if neighbor else 0.0
        return fwd + bwd + side

class SymbolicTerm(nn.Module):
    def __init__(self, func_type="sin"):
        super().__init__()
        self.func_type = func_type
        self.a = nn.Parameter(torch.randn(1))  # scale
        self.b = nn.Parameter(torch.randn(1))  # freq/multiplier
        self.c = nn.Parameter(torch.randn(1))  # phase/offset
        self.weight = nn.Parameter(torch.randn(1))
        self.route = nn.Parameter(torch.randn(3))  # fwd/bwd/side

    def safe_input(self, x):
        return torch.clamp(x, 1e-6, 1e6)  # for log, tan, etc.

    def compute(self, x):
        x = self.safe_input(x)
        t = self.func_type
        if t == "sin":
            return self.a * torch.sin(self.b * x + self.c)
        elif t == "cos":
            return self.a * torch.cos(self.b * x + self.c)
        elif t == "tan":
            return self.a * torch.tan(self.b * x + self.c)
        elif t == "csc":
            return self.a / torch.sin(self.b * x + self.c)
        elif t == "sec":
            return self.a / torch.cos(self.b * x + self.c)
        elif t == "cot":
            return self.a / torch.tan(self.b * x + self.c)
        elif t == "arcsin":
            return self.a * torch.arcsin(torch.clamp(self.b * x + self.c, -1 + 1e-3, 1 - 1e-3))
        elif t == "arccos":
            return self.a * torch.arccos(torch.clamp(self.b * x + self.c, -1 + 1e-3, 1 - 1e-3))
        elif t == "arctan":
            return self.a * torch.arctan(self.b * x + self.c)
        elif t == "exp":
            return self.a * torch.exp(torch.clamp(self.b * x + self.c, max=10))
        elif t == "log":
            return self.a * torch.log(torch.clamp(self.b * x + self.c, min=1e-3))
        elif t == "poly1":
            return self.a * x + self.b
        elif t == "poly2":
            return self.a * x**2 + self.b * x + self.c
        elif t == "poly3":
            return self.a * x**3 + self.b * x**2 + self.c
        elif t == "cheby2":
            return self.a * (2 * x**2 - 1)
        elif t == "cheby3":
            return self.a * (4 * x**3 - 3 * x)
        elif t == "leg2":
            return self.a * (0.5 * (3 * x**2 - 1))
        elif t == "leg3":
            return self.a * (0.5 * (5 * x**3 - 3 * x))
        else:
            return self.a * x  # fallback

    def forward(self, x, neighbor=None):
        value = self.compute(x)
        route_weights = F.softmax(self.route, dim=0)
        fwd = route_weights[0] * self.weight * value
        bwd = -route_weights[1] * self.weight * value
        side = route_weights[2] * neighbor.compute(x) if neighbor else 0.0
        return fwd + bwd + side


# Controller that generates evolving target values
class GoalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def update_goal(self, step):
        self.phase += 0.1
        return torch.tensor([[torch.sin(torch.tensor(self.phase))]])  # shape: [1, 1]

# Series that aggregates active terms
class ScalarLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Linear(1, 1)

    def forward(self, x):
        return self.scalar(x)

def analyze_model(model, step):
    func_counts = {}
    coef_stats = {"a": [], "b": [], "c": []}
    routing_entropy = []

    for term in model.terms:
        func = term.func_type
        func_counts[func] = func_counts.get(func, 0) + 1

        coef_stats["a"].append(term.a.item())
        coef_stats["b"].append(term.b.item())
        coef_stats["c"].append(term.c.item())

        route_probs = F.softmax(term.route, dim=0)
        entropy = -(route_probs * torch.log(route_probs + 1e-8)).sum().item()
        routing_entropy.append(entropy)

    print(f"\n[Diagnostics at step {step}]")
    print("Function distribution:", func_counts)
    print("Mean a/b/c:", {k: round(sum(v) / len(v), 4) for k, v in coef_stats.items()})
    print("Avg routing entropy:", round(sum(routing_entropy) / len(routing_entropy), 4))

class SeriesSimulator(nn.Module):
    def __init__(self, num_terms):
        super().__init__()
        self.scalar_layer = ScalarLayer()
        funcs = [
            "sin", "cos", "tan", "csc", "sec", "cot",
            "arcsin", "arccos", "arctan",
            "exp", "log",
            "poly1", "poly2", "poly3",
            "cheby2", "cheby3", "leg2", "leg3"
        ]

        self.terms = nn.ModuleList([SymbolicTerm(func_type=random.choice(funcs)) for _ in range(num_terms)])
        self.goal_gen = GoalGenerator()

    def forward(self, step):
        x = torch.tensor([[step / 100.0]])
        scalar_out = self.scalar_layer(x)

        # Combine base scalar with symbolic input
        x_symbolic = x + scalar_out

        target = self.goal_gen.update_goal(step)
        total = 0.0
        for i, term in enumerate(self.terms):
            neighbor = self.terms[i + 1] if i + 1 < len(self.terms) else None
            total += term(x_symbolic, neighbor)

        loss = F.smooth_l1_loss(total, target)
        return total, target, loss


# Training loop
def train_series(num_terms=50, steps=1001, lr=0.1):
    print(f"num_terms: {num_terms} steps: {steps} lr={lr}")
    model = SeriesSimulator(num_terms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    for step in range(steps):
        optimizer.zero_grad()
        output, target, loss = model(step)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.6f} | Output: {output.item():.4f} | Target: {target.item():.4f}")
        if step % 200 == 0:
            analyze_model(model, step)


    return model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TokenSelector(nn.Module):
    def __init__(self, hidden_dim, token_pool_size):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, token_pool_size)  # Not logits: each unit outputs a token ID
        )

    def forward(self, hidden):
        return torch.argmax(self.reduce(hidden), dim=-1)  # Direct choice of token
class ZipfTokenManager:
    def __init__(self, base_pool, max_active=128):
        self.full_pool = base_pool
        self.freq = {tok: 0 for tok in base_pool}
        self.max_active = max_active
        self.active_tokens = set()

    def update(self, used_tokens):
        for t in used_tokens:
            self.freq[t] += 1
        # Rebuild active token set based on frequency
        sorted_by_freq = sorted(self.freq.items(), key=lambda kv: -kv[1])
        self.active_tokens = set(t for t, _ in sorted_by_freq[:self.max_active])

    def encode_token(self, token):
        if token in self.active_tokens:
            return self.active_tokens.index(token)  # position in active list
        return -1  # flag as fallback

    def decode_token(self, idx):
        return list(self.active_tokens)[idx]

# === XY Encoding/Decoding ===
def encode_term(cx, ex, cy, ey):
    return (cx << 12) | (ex << 8) | (cy << 4) | ey

def decode_term(encoded):
    return ((encoded >> 12) & 0xF,
            (encoded >> 8) & 0xF,
            (encoded >> 4) & 0xF,
            encoded & 0xF)

def batch_decode_xy(terms):
    terms = torch.tensor(terms, dtype=torch.int16, device=device)
    cx = (terms >> 12) & 0xF
    ex = (terms >> 8)  & 0xF
    cy = (terms >> 4)  & 0xF
    ey = terms & 0xF
    return cx, ex, cy, ey

def evaluate_terms_batch(terms, x_vals, y_vals):
    cx, ex, cy, ey = batch_decode_xy(terms)
    x = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, -1)
    y = torch.tensor(y_vals, dtype=torch.float32, device=device).view(1, -1)
    x_part = cx.unsqueeze(1) * (x ** ex.unsqueeze(1))
    y_part = cy.unsqueeze(1) * (y ** ey.unsqueeze(1))
    return x_part + y_part

def differentiate_term(encoded):
    cx, ex, cy, ey = decode_term(encoded)
    dx = encode_term(cx * ex, max(ex - 1, 0), 0, 0)
    dy = encode_term(0, 0, cy * ey, max(ey - 1, 0))
    return dx, dy

def evaluate_term(encoded, x_vals, y_vals):
    cx, ex, cy, ey = decode_term(encoded)
    x = torch.tensor(x_vals, dtype=torch.float32, device=device)
    y = torch.tensor(y_vals, dtype=torch.float32, device=device)
    return (cx * (x ** ex)) + (cy * (y ** ey))

# === Polar Encoding ===
def encode_polar_term(term_type, coeff, exp):
    return (term_type << 12) | (coeff << 8) | exp

def decode_polar_term(encoded):
    return ((encoded >> 12) & 0b11,
            (encoded >> 8) & 0xF,
            encoded & 0xF)

def batch_decode_polar(terms):
    terms = torch.tensor(terms, dtype=torch.int16, device=device)
    term_type = (terms >> 12) & 0b11
    coeff     = (terms >> 8)  & 0xF
    exp       = terms & 0xF
    return term_type, coeff, exp

def evaluate_polar_term_batch(terms, x_vals, y_vals):
    term_type, coeff, exp = batch_decode_polar(terms)
    x = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, -1)
    y = torch.tensor(y_vals, dtype=torch.float32, device=device).view(1, -1)
    r = torch.sqrt(x ** 2 + y ** 2)
    result = torch.zeros((len(terms), len(x_vals)), device=device)

    for i in range(len(terms)):
        if term_type[i] == 1:
            result[i] = coeff[i] * (r[0] ** exp[i])
        elif term_type[i] == 2:
            result[i] = coeff[i] * torch.sin(r[0] ** exp[i])
        elif term_type[i] == 3:
            result[i] = coeff[i] * torch.cos(r[0] ** exp[i])
        else:
            result[i] = 0
    return result

# === Target Function ===
def target_function(x, y):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return 4 * x**2 + 2 * y**2


def generate_term_pool(max_coeff=4, max_exp=4):
    pool = []
    for cx in range(max_coeff + 1):
        for ex in range(max_exp + 1):
            for cy in range(max_coeff + 1):
                for ey in range(max_exp + 1):
                    pool.append(encode_term(cx, ex, cy, ey))
    return pool

def generate_polar_term_pool(max_coeff=4, max_exp=4):
    pool = []
    for typ in [1, 2, 3]:
        for c in range(max_coeff + 1):
            for e in range(max_exp + 1):
                pool.append(encode_polar_term(typ, c, e))
    return pool

def generate_hybrid_term_pool():
    return [('xy', t) for t in generate_term_pool()] + [('polar', t) for t in generate_polar_term_pool()]

# === Evaluation Logic ===
def evaluate_hybrid_terms(terms, x_vals, y_vals):
    xy_terms = [t[1] for t in terms if t[0] == 'xy']
    polar_terms = [t[1] for t in terms if t[0] == 'polar']
    preds = torch.zeros((len(x_vals),), device=device)
    if xy_terms:
        preds += evaluate_terms_batch(xy_terms, x_vals, y_vals).sum(dim=0)
    if polar_terms:
        preds += evaluate_polar_term_batch(polar_terms, x_vals, y_vals).sum(dim=0)
    return preds

def evaluate_hybrid_derivatives(terms, x_vals, y_vals):
    dx = torch.zeros(len(x_vals), device=device)
    dy = torch.zeros(len(x_vals), device=device)
    for kind, term in terms:
        if kind == 'xy':
            dx_t, dy_t = differentiate_term(term)
            dx += evaluate_term(dx_t, x_vals, y_vals)
            dy += evaluate_term(dy_t, x_vals, y_vals)
        # Polar derivs can be added symbolically or estimated numerically
    return dx, dy

def hybrid_loss(terms, x_vals, y_vals, target_fn, grad_weight=1.0):
    pred = evaluate_hybrid_terms(terms, x_vals, y_vals)
    target = target_fn(torch.tensor(x_vals), torch.tensor(y_vals)).to(device)
    loss = ((pred - target) ** 2).sum()

    dx_pred, dy_pred = evaluate_hybrid_derivatives(terms, x_vals, y_vals)
    dx_target = torch.tensor([8 * x for x in x_vals], device=device)
    dy_target = torch.tensor([4 * y for y in y_vals], device=device)
    grad_loss = ((dx_pred - dx_target)**2 + (dy_pred - dy_target)**2).sum()

    return loss + grad_weight * grad_loss

# === Plotting ===
def plot_surface(terms, fn_label='Symbolic Output'):
    x_grid = torch.linspace(0, 4, 50)
    y_grid = torch.linspace(0, 4, 50)
    Z = np.zeros((50, 50))
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            val = evaluate_hybrid_terms(terms, [x.item()], [y.item()])
            Z[i, j] = val.item()
    plt.imshow(Z, extent=[0, 4, 0, 4], origin='lower', cmap='magma')
    plt.colorbar()
    plt.title(fn_label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# === Main ===
def main():
    x_vals = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    y_vals = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)

    target_fn = lambda x, y: 4 * x**2 + 2 * y**2

    pool = generate_hybrid_term_pool()
    best_loss = float('inf')
    best_combo = None

    print("Searching combinations...")
    model = None#SymbolicSelectorModel(...)  # includes embedding, selector, etc.
    manager = ZipfTokenManager(pool, max_active=128)

    # Simulate initial token usage
    used_tokens = [random.choice(pool) for _ in range(50)]
    manager.update(used_tokens)

    selected_idxs = model(...)  # output token indices
    selected_tokens = [manager.decode_token(idx) for idx in selected_idxs]

    loss = hybrid_loss(selected_tokens, x_vals, y_vals, target_fn)


    print("\nBest combo (loss = {:.4f}):".format(best_loss))
    for kind, term in best_combo:
        if kind == 'xy':
            print(" XY term:", decode_term(term))
        elif kind == 'polar':
            print("Polar term:", decode_polar_term(term))

    plot_surface(best_combo, "Best Symbolic Surface")

if __name__ == "__main__":
    main()
