import torch
import torch.nn as nn


class LSTMForecasterSubhead(nn.Module):
    def __init__(
        self,
        input_features: int = 6,
        output_tmestampes: int = 36,
        hidden_size: int = 256,
        num_layers: int = 2, 
        dropout: float = 0.2
    ) -> None:
        super(LSTMForecasterSubhead, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        self.head = nn.Linear(
            hidden_size,
            output_tmestampes
        )
        
    def forward(self, x) -> torch.Tensor:
        
        out, _ = self.lstm(x)
        
        out = out[:, -1, :]  # take last time step
        out = self.head(out)
        
        return out


class LSTMForecaster(nn.Module):
    def __init__(self, output_timestamps: int, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.series_1_head = LSTMForecasterSubhead(output_tmestampes=output_timestamps, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.series_2_head = LSTMForecasterSubhead(output_tmestampes=output_timestamps, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.series_3_head = LSTMForecasterSubhead(output_tmestampes=output_timestamps, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.series_4_head = LSTMForecasterSubhead(output_tmestampes=output_timestamps, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.series_5_head = LSTMForecasterSubhead(output_tmestampes=output_timestamps, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.series_6_head = LSTMForecasterSubhead(output_tmestampes=output_timestamps, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        predicted_series_1: torch.Tensor = self.series_1_head(x)
        predicted_series_2: torch.Tensor = self.series_2_head(x)
        predicted_series_3: torch.Tensor = self.series_3_head(x)
        predicted_series_4: torch.Tensor = self.series_4_head(x)
        predicted_series_5: torch.Tensor = self.series_5_head(x)
        predicted_series_6: torch.Tensor = self.series_6_head(x)
        
        y = torch.stack(
            [
                predicted_series_1,
                predicted_series_2,
                predicted_series_3,
                predicted_series_4,
                predicted_series_5,
                predicted_series_6
            ],
            1
        )
        
        return y.permute(0, 2, 1)
    
    
class LSTMForecaster(nn.Module):
    def __init__(self, input_features: int, output_timestamps: int, hidden_size=256, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Separate heads for each series
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, output_timestamps) for _ in range(6)
        ])
        
        self.attention_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        
        attention_weights = torch.softmax(self.attention_layer(out), dim=1)  # (batch_size, seq_len, 1)
        context_vector = (out * attention_weights).sum(dim=1)
        
        outputs = [head(context_vector) for head in self.heads]  # list of tensors (batch_size, output_timestamps)
        
        y = torch.stack(outputs, dim=1)  # (batch_size, 6, output_timestamps)
        
        return y.permute(0, 2, 1)
