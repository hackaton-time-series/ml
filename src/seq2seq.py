import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc_out(outputs)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_seq_len, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_seq_len = output_seq_len
        self.device = device
    
    def forward(self, src):
        batch_size = src.size(0)
        output_dim = src.size(2)  # assuming (batch, seq, feature)
        
        # Encode
        hidden, cell = self.encoder(src)

        # Prepare decoder input: start with zeros (or could use last encoder input)
        decoder_input = torch.zeros((batch_size, 1, output_dim)).to(self.device)

        outputs = []

        # Predict output_seq_len steps
        for _ in range(self.output_seq_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(decoder_output)
            decoder_input = decoder_output  # auto-regressive (no teacher forcing here)

        outputs = torch.cat(outputs, dim=1)  # (batch, output_seq_len, feature)
        return outputs
