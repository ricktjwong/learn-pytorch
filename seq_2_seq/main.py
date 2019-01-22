import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab
import torch.utils as u

PAD_token = 0
SOS_token = 1
EOS_token = 2
VOCAB_SZ = 6000 
EMBEDDING_DIM = 100
MAX_LENGTH_DESC = 25 # 50
MAX_LENGTH_HEAD = 25 # 50
BATCH_SZ = 32
HIDDEN_SZ = 500
nb_unknown_words = 100

torch.cuda.manual_seed(1)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

import itertools
import random

glove = vocab.GloVe(name='6B', dim=100)

"""
Data loading
"""

import os
import re
myPath = 'data/bbc/'


def getDataFilePaths(dataset='tech'):
  # Read the file and split into lines
    f = []
    result = []
    for (dirpath, dirnames, filenames) in os.walk(myPath+dataset):
        f.extend(filenames)
        break
    for file in f:
      # some files are duplicates, these will have whitespaces in their names
      if not any([x.isspace() for x in file]):
        result.append(os.path.join(myPath+dataset,file))
    return result


def normalizeString(s):
  # Lowercase, trim, and remove non-letter characters unless . ! ?
  s = re.sub(r"([.!?])", r" \1", s) 
  s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) 
  return s


def isCorrectSize(p, max_len_d=MAX_LENGTH_DESC, max_len_h=MAX_LENGTH_HEAD):
  # Keep only words that start with selected english phrases and max_length (For quick training)
  return len(p[0].split(' ')) < max_len_d and \
          len(p[1].split(' ')) < max_len_h 


def filterPairByLen(pairs, max_len_d=MAX_LENGTH_DESC, max_len_h=MAX_LENGTH_HEAD):
    return [pair for pair in pairs if isCorrectSize(pair, max_len_d, max_len_h)]
  

def readDataFiles(dataset): 
  print("Reading lines...") 
  data_tuples = [] # first index is heading, then beginning of article (description) 
  data_files = getDataFilePaths(dataset) 
  for file in data_files: 
    lines = open(file, encoding='utf-8', errors='ignore').read().strip().split('\n')         
    lines = list(filter(lambda line: line, lines)) # filter out empty lines
    # make sure label and instance in right order
    data_tuples.append((normalizeString(lines[1]), normalizeString(lines[0]))) 
  return data_tuples


def prepareData(dataset, max_len_d=MAX_LENGTH_DESC, max_len_h=MAX_LENGTH_HEAD):
  data_tuples = readDataFiles(dataset)
  res = filterPairByLen(data_tuples)
  return res

training_data_tech = prepareData('tech')
training_data_sport = prepareData('sport')
training_data_pol = prepareData('politics')
training_data_ent = prepareData('entertainment')
training_data_biz = prepareData('business')
training_data = training_data_sport + training_data_pol + training_data_ent + training_data_biz + training_data_tech

"""
Data preparing
"""

from collections import Counter
from itertools import chain

data = filterPairByLen(training_data)
desc = [pair[0] for pair in data]
heads = [pair[1] for pair in data]
vocabcount = Counter(w for txt in heads+desc for w in txt.split())
vocab = list(map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1])))

# now we have to index the words 
start_idx = 3 # first real word​
word2idx = dict((word, idx+start_idx) for idx, word in enumerate(vocab))
print(len(word2idx))
word2idx['<SOS>'] = SOS_token
word2idx['<EOS>'] = EOS_token    
word2idx['<PAD>'] = PAD_token
idx2word = dict((idx,word) for word,idx in word2idx.items())    

print(vocab[-10:])

"""
Build Embeddings Matrix
"""
import numpy as np

# Init word embeddings from GloVe
def get_new_word_embeddings(embedding_dim, vocab_sz=VOCAB_SZ):  
  emb_layer = nn.Embedding(vocab_sz, embedding_dim)  
  
  # generate random embedding with same scale as glove
  np.random.seed(42)
  shape = (vocab_sz, embedding_dim) 
  print(f'shape: {shape}')
  scale = 0.04 *np.sqrt(12)/2 # uniform and not normal
  embeddings = np.random.uniform(low=-scale, high=scale, size=shape)
  print('random-embedding/glove scale', scale, 'std', embeddings.std())

  # copy from glove weights of words that appear in our short vocabulary (idx2word)
  c = 0
  for i in range(vocab_sz):
      w = idx2word[i]
      g_idx = None
      if w in glove.stoi:
        g_idx = glove.stoi[w]        
      elif w.lower() in glove.stoi:
        g_idx = glove.stoi[w.lower()]                
      if g_idx is not None:
          embeddings[i,:] = glove.vectors[g_idx]
          c+=1
  print('number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(vocab_sz))
  emb_layer.load_state_dict({'weight': torch.tensor(embeddings)})
  return emb_layer, torch.tensor(embeddings)


nb_unknown_words = 100 # this many idxs will be used for unkown words in the embedding matrix
vocab_size = VOCAB_SZ
embedding_dim = EMBEDDING_DIM

_, embedding = get_new_word_embeddings(embedding_dim,vocab_size)

# we want uni vectors for each word/dimension
normed_embedding = embedding/torch.tensor([torch.sqrt(torch.dot(gweight,gweight)) for gweight in embedding]).view(-1,1)

glove_thr = 0.7

glove_match = []
for w,idx in word2idx.items():
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in glove.stoi:
        gidx = glove.stoi[w]
        gweight = glove.vectors[gidx]
        # find row in embedding that has the highest cos score with gweight
        gweight /= torch.sqrt(torch.dot(gweight,gweight))
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
        # keep trying until we find a similar word that isnt same and is in vocab size
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_thr:
                break
            if embedding_idx < vocab_size-nb_unknown_words:
                glove_match.append((w, embedding_idx, s)) 
                break
            score[embedding_idx] = -1
glove_match.sort(key = lambda x: -x[2])
print(f'# of glove substitutes found {len(glove_match)}')

word2similarIdx = dict((word2idx[w],embedding_idx) for w, embedding_idx, _ in glove_match)

"""
Further filter training data
"""

new = []
for pair in training_data:              
  desc = ' '.join(list(filter(lambda word: word in word2idx or word in word2similarIdx , pair[0].split(" "))))
  head = ' '.join(list(filter(lambda word: word in word2idx or word in word2similarIdx , pair[1].split(" "))))
  new.append((desc,head))  

training_data = new

def indexesFromSentence(sentence,vocab_sz=VOCAB_SZ):
    idxs = []
    for word in sentence.split(" "):
      if word:
        idx = word2idx[word]
        if idx >= vocab_sz-nb_unknown_words:
          out_of_bound_word = idx2word[idx]
          if out_of_bound_word in word2similarIdx:
            similar_word_idx = word2similarIdx[out_of_bound_word]
            idx = similar_word_idx
          else:
            continue
        idxs.append(idx)        
    return idxs + [EOS_token]   


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))
  
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m  
  
# Returns padded input sequence tensor and lengths
def inputVar(l):
    indexes_batch = [indexesFromSentence(sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths
  
  
# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l):
    indexes_batch = [indexesFromSentence(sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len  

  
def coutRealWords(sen):
    cnt = 0
    for word in sen:
      if word and word in word2similarIdx:
        cnt += 1
    return cnt    


# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch):
#     pair_batch.sort(key=lambda x: coutRealWords(x[0].split(" ")), reverse=True)
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch)  
#     print(f'inptss:{inp}')
    print(f'lengths:{lengths}')    
    output, mask, max_target_len = outputVar(output_batch)
    return inp, lengths, output, mask, max_target_len
    
# Example for validation
small_batch_size = 5
batches = batch2TrainData([random.choice(training_data) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

# how fucked up is this thing
for orig, sub, score in glove_match[-10:]:
    print(score, orig,'=>', idx2word[sub])


"""
Model
"""
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size=EMBEDDING_DIM, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embeddings, _ = get_new_word_embeddings(embedding_size)
        
        # this time we use a bi-directional gru
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
      
        # Convert word indexes to embeddings
        # input is like (max_len, batch_sz)
        
#         print(f'this is input seq: {input_lengths}')
        input_embedded = self.embeddings(input_seq)
#         print(f'this is it embedded: {input_seq}')
        print("Input lengths: ")
        print(len(input_lengths))
        print("Input embedded: ")
        print(len(input_embedded))
        input_lengths, sorted_indices = torch.sort(input_lengths, descending=True)

        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_embedded, input_lengths)
        
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        
        # Sum bidirectional GRU outputs. becuase we have a bidrectional gru, first half
        # is forward Gru, second half is backward gru.
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        
        # Return output and final hidden state
        return outputs, hidden


class Attn(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        
        # using the general model from luong et al.
        self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
    
    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        
        # outputs.shape = (max_len,batch_sz,hidden_size)
        # hidden.shape = (1,batch_sz,hidden_size)    
        # we sum over 3rd dimesion to end up with (max_len, B, 1) i.e the energies
        return torch.sum(hidden * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
      
        # Calculate the attention weights (energies) based on the given method                
        attn_energies = self.general_score(hidden, encoder_outputs)
        
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# still using the Luong model
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embeddings, _ = get_new_word_embeddings(embedding_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
      
        # Note: we run this one step (word) at a time
        # input_step.shape = (1,batch_sz)
        
        # Get embedding of current input word
        input_embedded = self.embeddings(input_step)
        input_embedded = self.embedding_dropout(input_embedded)
        
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(input_embedded, last_hidden)
        
        # Calculate attention weights from the current GRU output (i.e last Hidden)
        # with shape = (batch_sz,1, max_len)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # we filp the encoder_outputs so that the face of the cube is outputs from one batch
        # then we weight with attention weights. Nb matmul not dot.
        # gives new cotext of shape = (batch_sz,1, hidden_sz)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0) # now (batch_sz, hidden)
        context = context.squeeze(1) # now (batch_sz, hidden) as well
        concat_input = torch.cat((rnn_output, context), 1) # shape = (batch_sz, hidden + hidden) as well 
        concat_output = torch.tanh(self.concat(concat_input)) # shape = (batch_sz, hidden) 
        
        # Predict next word using Luong eq. 6
        output = self.out(concat_output) # shape = (batch_sz, output_sz) 
        output = F.softmax(output, dim=1) # shape = (batch_sz, output_sz) with probabilities 
        
        # Return output and final hidden state
        return output, hidden


# Configure models
model_name = 'text_summariser_2_model'
attn_model = 'general'
save_dir = 'model_data'
corpus_name = 'bbc_dataset_small'
hidden_size = HIDDEN_SZ
embedding_size = EMBEDDING_DIM
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = BATCH_SZ

# check that we have a saved model directory etc, if not make one.
directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = 4000

loadFilename = os.path.join(save_dir, model_name, corpus_name,
                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                           '{}_checkpoint.tar'.format(checkpoint_iter)) if os.path.exists(directory) else None


# Load model if a loadFilename is provided
if loadFilename:
  
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    
    # get save_dicts and what not
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    word2similarIdx = checkpoint['word2similarIdx']
    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']

print('Building encoder and decoder ...')

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding_size, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(embedding_size, hidden_size, len(word2idx), decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

for param in encoder.parameters():
  print(param)

"""
Training helper functions
"""

def maskNLLLoss(inp, target, mask):
    # inp,shape = (batch_sz,vocab_sz) and is probs of next word
    # target.shape = ([batch_sz]) and is all the correct next words 
    # mask.shape = (max_len, batch_sz) and is byteTensor (i.e binary)
    
    nTotal = mask.sum()    
    
    # we get the output probability of the correct word as shape (1,batch_sz)
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    
    # we mask out probs of next words following paddings and calculate loss
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    
    # returning the mean loss
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding_size,
        encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH_HEAD):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            r_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            
            # transform most probable words into shape (1,batch_sz) for decoder input
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding_size, 
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration,
               batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData([random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        
#         print(f'training on: var: {input_variable}, lens: {lengths}')
        
        # Run a training iteration with batch        
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding_size, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)                
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'word2similarIdx': word2similarIdx,
                'word2idx': word2idx,
                'idx2word': idx2word
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
# n_iteration = 4000
n_iteration = 100
print_every = 1
save_every = 100

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(model_name, training_data, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding_size, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)
