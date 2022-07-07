import torch
import torch.nn as nn


class BaseModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(BaseModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim

        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, output_size)

        self.init_weights(input_size)
        
    def init_weights(self, input_size):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        embedding = self.embedding(x)

        outputs, hidden = self.rnn(embedding, None)  # outputs.shape -> (sequence length, batch size, hidden size)

        outputs = outputs[:, -1, :] if self.batch_first else outputs[-1, :, :]
        output = self.fc(outputs)
        
        return output, hidden