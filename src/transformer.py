import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Shape: [max_len, 1]
        # Term for calculating frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices in pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in pe
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension: [1, max_len, d_model] -> becomes [max_len, 1, d_model] after unsqueeze(0) below
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]

        # Register 'pe' as a buffer that should not be considered a model parameter.
        # 'pe' will be moved to the correct device along with the module.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [sequence_length, batch_size, d_model]
               OR [batch_size, sequence_length, d_model] if batch_first=True elsewhere
        """
        # If batch_first=True was used for the input x to this module:
        # x shape: [batch_size, sequence_length, d_model]
        # self.pe shape: [max_len, 1, d_model]
        # We need to add positional encodings to x.
        # Select positional encodings up to the sequence length of x: self.pe[:x.size(1), :]
        # Transpose pe slice to match x's batch-first format if needed, or adjust x.
        # Let's assume x comes in as [batch_size, sequence_length, d_model]
        # self.pe[:x.size(1), :].transpose(0,1) gives shape [1, sequence_length, d_model]
        # This will broadcast correctly during addition.

        # If x is [sequence_length, batch_size, d_model] (PyTorch default)
        # x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)

        # If x is [batch_size, sequence_length, d_model] (using batch_first=True)
        # Need pe slice shape [1, sequence_length, d_model]
        pe_slice = self.pe[:x.size(1), :].transpose(0, 1) # Shape [1, sequence_length, d_model]
        x = x + pe_slice # Broadcasting adds positional encoding to each batch element
        return self.dropout(x)


class TransformerTimeSeriesPredictor(nn.Module):
    """
    Transformer-based model for time series prediction.

    Args:
        input_dim (int): The number of features in the input time series (e.g., 6).
        d_model (int): The dimension of the transformer embeddings and hidden layers.
                       Must be divisible by nhead.
        nhead (int): The number of attention heads in the multi-head attention mechanism.
        num_encoder_layers (int): The number of stacked transformer encoder layers.
        dim_feedforward (int): The dimension of the feedforward network model in encoder layers.
        output_dim (int): The number of time steps to predict into the future.
        dropout (float): Dropout probability.
        max_len (int): Maximum sequence length for positional encoding.
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0.1, max_len=5000):
        super(TransformerTimeSeriesPredictor, self).__init__()
        
        self.output_timestamps = output_dim
        
        self.input_dim = input_dim

        self.d_model = d_model

        # --- Input Embedding ---
        # Linear layer to project input features to d_model dimension
        self.input_embedding = nn.Linear(input_dim, d_model)

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # --- Transformer Encoder ---
        # Define a single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu', # or 'gelu'
            batch_first=True # IMPORTANT: Input/Output shape (batch, seq, feature)
        )
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model) # Optional final normalization
        )

        # --- Output Layer ---
        # Linear layer to map the final transformer output to the desired prediction dimension
        self.output_layer = nn.Linear(d_model, output_dim)

        # Initialize weights (optional but often recommended)
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def forward(self, src):
        """
        Defines the forward pass of the model.

        Args:
            src (torch.Tensor): The input time series data with shape
                               (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: The predicted values with shape (batch_size, output_dim).
        """
        # 1. Input Embedding
        # src shape: [batch_size, seq_len, input_dim]
        # embedded shape: [batch_size, seq_len, d_model]
        embedded = self.input_embedding(src) * math.sqrt(self.d_model) # Scale embedding

        # 2. Positional Encoding
        # pos_encoded shape: [batch_size, seq_len, d_model]
        pos_encoded = self.pos_encoder(embedded)

        # 3. Transformer Encoder
        # encoder_output shape: [batch_size, seq_len, d_model]
        # Note: No mask is applied here, assuming we only use the final output
        # for forecasting. If using for other tasks or intermediate outputs,
        # a mask might be needed (e.g., nn.Transformer.generate_square_subsequent_mask).
        encoder_output = self.transformer_encoder(pos_encoded)

        # 4. Output Layer
        # We typically use the output corresponding to the *last* time step
        # of the input sequence for forecasting.
        # last_step_output shape: [batch_size, d_model]
        last_step_output = encoder_output[:, -1, :]

        # prediction shape: [batch_size, output_dim]
        prediction = self.output_layer(last_step_output)

        return prediction.reshape(src.shape[0], self.output_timestamps // 6, self.input_dim)