import torch
import numpy as np
import torch.nn as nn
from typing import Optional, List, Union
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

def _get_activation(act_name: str) -> nn.Module:
    """Map activation name to a simple nn.Module. Default: ReLU."""
    name = act_name.lower()
    if name == "relu":
        return nn.ReLU()
    if name in ("leakyrelu", "leaky_relu", "leaky"):
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "elu":
        return nn.ELU()
    # fallback
    return nn.ReLU()

def irt2pl(theta, a, b, *, F=np):
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))


class MLP(nn.Module):
    """
    Simple MLP builder.
    Parameters must be passed explicitly by caller (we do NOT read args here).
    - input_dim: input feature dim
    - out_dim: output dim
    - hidden_sizes: list or int (single hidden layer)
    - activation: activation name (string)
    - dropout: float
    - use_batchnorm: bool
    - use_mf: bool (apply low-rank bottleneck reconstruction on input)
    - emb_dim: int (required if use_mf True)
    - init_std: float (normal init std)
    """
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_sizes: Optional[Union[int, List[int]]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        use_mf: bool = False,
        emb_dim: Optional[int] = None,
        init_std: float = 0.02,
    ):
        super().__init__()

        # normalize hidden_sizes to list
        if hidden_sizes is None:
            hidden_sizes = []
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.hidden_sizes = list(hidden_sizes)

        self.use_mf = bool(use_mf)
        if self.use_mf and emb_dim is None:
            raise ValueError("emb_dim must be provided when use_mf is True")

        # optional MF projection (encoder->decoder)
        if self.use_mf:
            self.mf_encoder = nn.Linear(input_dim, emb_dim)
            self.mf_decoder = nn.Linear(emb_dim, input_dim)
        else:
            self.mf_encoder = None
            self.mf_decoder = None

        act = _get_activation(activation)

        layers = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout and float(dropout) > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h

        self.mlp_body = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_layer = nn.Linear(prev, out_dim)

        self._init_weights(float(init_std))

    def _init_weights(self, std: float):
        """Initialize Linear weights with normal(mean=0,std=std) and biases to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: optional MF reconstruct -> MLP -> output layer"""
        if self.use_mf:
            z = self.mf_encoder(x)
            x = self.mf_decoder(z)
        h = self.mlp_body(x)
        return self.output_layer(h)

class MIRTNet(nn.Module):
    def __init__(self, input_dim, latent_dim, a_range = None, theta_range = None,irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Linear(input_dim, latent_dim, bias=False)
        self.a = nn.Linear(input_dim, latent_dim, bias=False)
        self.b = nn.Linear(input_dim, 1, bias=False)
        self.a_range = a_range
        self.theta_range = theta_range

    def forward(self, llm, item):
        theta = torch.squeeze(self.theta(llm), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        if self.theta_range is not None:
            theta = self.theta_range * torch.sigmoid(theta)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        b = torch.squeeze(self.b(item), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        
        pred = self.irf(theta, a, b, **self.irf_kwargs)
        return pred, theta, a, b

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
    
class NIRTNet(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        self.knowledge_dim = latent_dim
        self.item_dim = input_dim
        self.llm_dim = input_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 512

        super(NIRTNet, self).__init__()

        self.model_emb = nn.Linear(self.llm_dim, self.knowledge_dim)
        self.k_difficulty = nn.Linear(self.item_dim, self.knowledge_dim)
        self.k = nn.Linear(self.knowledge_dim, self.knowledge_dim)
        self.e_difficulty = nn.Linear(self.item_dim, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len2)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        self.softmax = nn.Softmax(dim=1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, llm, input_query, input_knowledge_point):

        llm_emb = self.model_emb(llm)
        stat_emb = torch.sigmoid(llm_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_query))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_query)) * 9

        if len(input_knowledge_point.shape) == 1:
            input_knowledge_point = input_knowledge_point.unsqueeze(0)  
        input_knowledge_point = self.softmax(input_knowledge_point)

        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.squeeze(), stat_emb, e_difficulty, k_difficulty, input_knowledge_point


class KMeansWrapper(nn.Module):
    """
    Wrap sklearn KMeans. Normally use .fit() outside torch, then load centers here.
    If fit=True, we run sklearn KMeans inside this class (CPU only).
    """
    def __init__(self, n_clusters=64, n_init="auto", max_iter=300, algorithm="lloyd", fit=True,random_state = 42):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.algorithm = algorithm
        self.fit = fit
        self.random_state = random_state
        self.centers = None  # will be torch.Tensor

    def fit_kmeans(self, X: torch.Tensor):
        """Fit KMeans using sklearn (CPU) and store cluster centers."""
        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            algorithm=self.algorithm,
            random_state=self.random_state,
        )
        km.fit(X)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        self.centers = nn.Parameter(centers, requires_grad=False)

    def forward(self, x: torch.Tensor):
        """
        Return nearest cluster index for each sample.
        """
        if self.centers is None:
            raise RuntimeError("KMeans centers not initialized. Call fit_kmeans() first.")
        centers = self.centers
        if centers.device != x.device:
            centers = centers.to(x.device, non_blocking=True)
        dist = torch.cdist(x, centers)
        return dist.argmin(dim=1)

class EdgeGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        src_emb = x[edge_index[0]]  # (num_edges, hidden_dim)
        dst_emb = x[edge_index[1]]  # (num_edges, hidden_dim)
        edge_in = torch.cat([src_emb, dst_emb], dim=1)  # (num_edges, hidden_dim*2)

        preds = self.edge_mlp(edge_in)  # (num_edges, out_dim)
        return preds


def init_model(args, input_dim: int = 0, out_dim: int = 0) -> nn.Module:
    if args["model"]["name"] == "mlp":
        hidden_sizes = args["model"]["hidden_sizes"]
        activation = args["model"]["activation"]
        dropout = args["model"]["dropout"]
        use_batchnorm = args["model"]["use_batchnorm"]
        use_mf = args["model"]["mf"]["use_mf"]
        emb_dim = input_dim//4
        init_std = args["model"]["mf"]["init_std"]

        model = MLP(
            input_dim=input_dim,
            out_dim=out_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            use_mf=use_mf,
            emb_dim=emb_dim,
            init_std=init_std,
        )
    elif args["model"]["name"] == "kmeans":
        n_clusters = args["model"]["clusters"]
        n_init = args["model"]["n_init"]
        max_iter = args["model"]["max_iter"]
        algorithm = args["model"]["algorithm"] 
        random_state = args["seed"]

        model = KMeansWrapper(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            algorithm=algorithm,
            fit=True,
            random_state=random_state
        )
    elif args["model"]["name"] == "mirt":
        latent_dim = args["model"]["latent_dim"]

        model = MIRTNet(
            input_dim=input_dim,
            latent_dim=latent_dim
        )
    elif args["model"]["name"] == "nirt":
        latent_dim = args["model"]["latent_dim"]

        model = NIRTNet(
            input_dim=input_dim,
            latent_dim=latent_dim
        )
    elif args["model"]["name"] == "gnn":
        hidden_dim = args["model"]["hidden_dim"]
        num_layers = args["model"]["num_layers"]

        model = EdgeGNN(
            input_dim=input_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    return model
