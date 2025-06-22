def generate_pytorch_code(config: dict, dataset_name: str, data_type: str) -> str:
    """
    Generate executable PyTorch code string based on the given configuration.
    """
    code_lines = []
    code_lines.append(f"# Auto-generated PyTorch Model for {dataset_name} ({data_type})")
    code_lines.append("import torch")
    code_lines.append("import torch.nn as nn")
    code_lines.append("import torch.optim as optim")
    code_lines.append("from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset")
    code_lines.append("")
    # Dataset stub
    code_lines.append("class CustomDataset(Dataset):")
    code_lines.append("    def __init__(self, X, y):")
    code_lines.append("        self.X = X")
    code_lines.append("        self.y = y")
    code_lines.append("")
    code_lines.append("    def __len__(self):")
    code_lines.append("        return len(self.X)")
    code_lines.append("")
    code_lines.append("    def __getitem__(self, idx):")
    code_lines.append("        return self.X[idx], self.y[idx]")
    code_lines.append("")
    # Model definition
    code_lines.append("class DLModel(nn.Module):")
    code_lines.append("    def __init__(self, input_size, num_classes):")
    code_lines.append("        super(DLModel, self).__init__()")
    arch = config.get('architecture')
    if arch == 'CNN':
        layers = []
        filters = config['filters']
        for i, f in enumerate(filters):
            in_ch = 3 if i == 0 else filters[i-1]
            layers.append(f"self.conv{i+1} = nn.Conv2d({in_ch}, {f}, kernel_size=3, padding=1)")
            layers.append(f"self.bn{i+1} = nn.BatchNorm2d({f})")
            layers.append(f"self.relu{i+1} = nn.ReLU()")
            layers.append(f"self.pool{i+1} = nn.MaxPool2d(2,2)")
        code_lines.extend(["        " + ln for ln in layers])
        code_lines.append("        self.flatten = nn.Flatten()")
        code_lines.append(f"        self.fc1 = nn.Linear({filters[-1]} * (input_size//{2**len(filters)})**2, 128)")
        code_lines.append("        self.fc2 = nn.Linear(128, num_classes)")
    elif arch == 'LSTM':
        code_lines.append(f"        self.embedding = nn.Embedding(config.get('vocab_size', 10000), config.get('embedding_dim', 128))")
        code_lines.append(f"        self.lstm = nn.LSTM(config.get('embedding_dim',128), {config['hidden_size']}, batch_first=True)")
        code_lines.append(f"        self.fc = nn.Linear({config['hidden_size']}, num_classes)")
    else:  # MLP
        hidden = config['hidden_layers']
        code_lines.append(f"        self.fc1 = nn.Linear(input_size, {hidden[0]})")
        for idx in range(1, len(hidden)):
            code_lines.append(f"        self.fc{idx+1} = nn.Linear({hidden[idx-1]}, {hidden[idx]})")
        code_lines.append(f"        self.fc_out = nn.Linear({hidden[-1]}, num_classes)")
    code_lines.append("")
    # forward
    code_lines.append("    def forward(self, x):")
    if arch == 'CNN':
        for i in range(len(config['filters'])):
            code_lines.append(f"        x = self.conv{i+1}(x)")
            code_lines.append(f"        x = self.bn{i+1}(x)")
            code_lines.append(f"        x = self.relu{i+1}(x)")
            code_lines.append(f"        x = self.pool{i+1}(x)")
        code_lines.append("        x = self.flatten(x)")
        code_lines.append("        x = torch.relu(self.fc1(x))")
        code_lines.append("        x = self.fc2(x)")
    elif arch == 'LSTM':
        code_lines.append("        x = self.embedding(x)")
        code_lines.append("        out, (hn, cn) = self.lstm(x)")
        code_lines.append("        x = hn[-1]")
        code_lines.append("        x = self.fc(x)")
    else:
        for idx in range(len(config['hidden_layers'])):
            code_lines.append(f"        x = torch.relu(self.fc{idx+1}(x))")
        code_lines.append("        x = self.fc_out(x)")
    code_lines.append("        return x")
    code_lines.append("")
    # Initialization and training stub
    code_lines.append("# Initialize model")
    code_lines.append("model = DLModel(input_size=config.get('input_size'), num_classes=config.get('num_classes'))")
    code_lines.append("")
    code_lines.append("# Weight initialization")
    code_lines.append("def init_weights(m):")
    code_lines.append("    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):")
    wi = config['weight_init']
    code_lines.append(f"        nn.init.{ 'kaiming_normal_' if wi=='he_normal' else 'xavier_uniform_' }(m.weight, nonlinearity='relu')")
    code_lines.append("        if m.bias is not None: nn.init.constant_(m.bias, 0)")
    code_lines.append("model.apply(init_weights)")
    code_lines.append("")
    code_lines.append("# Optimizer")
    code_lines.append(f"optimizer = optim.Adam(model.parameters(), lr={config['learning_rate']})")
    code_lines.append("")
    code_lines.append("# You can now prepare DataLoader and run training loop")

    return " ".join(code_lines)
