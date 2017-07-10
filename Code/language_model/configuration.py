data_type = "cmu_dict"
data_src_cmu = "./data/cmudict-0.7b" 
data_src = "./data/imdb.txt"

max_sequence_length_imdb = 80
max_sequence_length_cmu_dict = 60
MAX_SEQUENCE_LENGTH=None
if data_type=="cmu_dict":
	MAX_SEQUENCE_LENGTH = max_sequence_length_cmu_dict
else:
	MAX_SEQUENCE_LENGTH = 80
print "MAX_SEQUENCE_LENGTH= ",MAX_SEQUENCE_LENGTH

VALIDATION_SPLIT = 0.1
TEST_SPLIT=0.1
MAX_VOCAB_SIZE=1500

inp_length = MAX_SEQUENCE_LENGTH
vocab_size = MAX_VOCAB_SIZE
print "MAX_VOCAB_SIZE = ", MAX_VOCAB_SIZE
embeddings_dim = 50
print "embeddings_dim = ", embeddings_dim
num_epochs = 200
batch_size = 25
dropout_val = 0.2

character_model = "CHARACTER_MODEL"
word_model = "WORD_MODEL"
char_or_word = character_model
lstm_cell_size=100
