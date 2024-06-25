import torch.nn as nn
import torch.nn.functional as F  
from text_preprocess import vocab

class TextEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output 


# embedding_dim = 128 
# hidden_dim = 256 
# vocab_size = len(vocab)


# text_embedding_model = TextEmbeddingModel(vocab_size, embedding_dim, hidden_dim)
# print(text_embedding_model)
