
import os
from my_py_toolkit.log.logger import get_logger
home = os.path.expanduser(".")
train_file = os.path.join(home, "data", "squad", "train-v1.1.json")
dev_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
test_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
glove_word_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

device = "cuda"

bert_path = "./data/model/RoBERTa-wwm-ext-large"

target_dir = "data"
event_dir = "log"
save_dir = "model"
answer_dir = "log"
train_record_file = os.path.join(target_dir, "train.npz")
dev_record_file = os.path.join(target_dir, "dev.npz")
test_record_file = os.path.join(target_dir, "test.npz")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval_file = os.path.join(target_dir, "train/train_eval.json")
dev_eval_file = os.path.join(target_dir, "train/dev_eval.json")
test_eval_file = os.path.join(target_dir, "test_eval.json")
dev_meta_file = os.path.join(target_dir, "dev_meta.json")
test_meta_file = os.path.join(target_dir, "test_meta.json")
word2idx_file = os.path.join(target_dir, "word2idx.json")
char2idx_file = os.path.join(target_dir, "char2idx.json")
answer_file = os.path.join(answer_dir, "answer.json")

dataset_name = "cmcr2018"
dataset_dir = {
  "cmcr2018": "./data/train/cmrc2018"
}

glove_char_size = 94 #Corpus size for Glove
glove_word_size = int(2.2e6) #Corpus size for Glove
glove_dim = 300 #Embedding dimension for Glove
char_dim = 64 #Embedding dimension for char
bert_dim = 1024 #Embedding dimension for char

para_limit = 512 #Limit length for paragraph
ques_limit = 20 #Limit length for question
ans_limit = 30 #Limit length for answers
char_limit = 16 #Limit length for character
word_count_limit = -1 #Min count for word
char_count_limit = -1 #Min count for char

is_continue = False
model_dir = "./model"
continue_checkpoint = 0
capacity = 15000 #Batch size of dataset shuffle
num_threads = 4 #Number of threads in input pipeline
is_bucket = False #build bucket batch iterator or not
bucket_range = [40, 401, 40] #the range of bucket

batch_size = 11 #Batch size
num_steps = 1 #Number of steps
checkpoint = 50 # 200 #checkpoint to save and evaluate the model
period = 100 #period to save batch loss
val_num_batches = 10 #Number of batches to evaluate the model
val_num_steps = 1
test_num_batches = 10 #Number of batches to evaluate the model
test_num_steps = 100
dropout = 0.1 #Dropout prob across the layers
dropout_char = 0.05 #Dropout prob across the layers
grad_clip = 5.0 #Global Norm gradient clipping rate
learning_rate = 0.001 #Learning rate
lr_warm_up_num = 1000 #Number of warm-up steps of learning rate
ema_decay = 0.9999 #Exponential moving average decay
beta1 = 0.8 #Beta 1
beta2 = 0.999 #Beta 2
early_stop = 10 #Checkpoints for early stop
d_model = 96 #Dimension of connectors of each layer
num_heads = 8 #Number of heads in multi-head attention
epochs = 20 # The epoch of train.
steps_num = 5000 # The num of train step in one epoch.
interval_save = 50 # The interval of save model.
min_loss = 0 # The scope of loss.可能出现 loss 非常大的情况
max_loss = 1000

# Extensions (Uncomment corresponding line in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
pretrained_char = False #Whether to use pretrained char embedding

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
fasttext = False #Whether to use fasttext

is_only_save_params = True

log_path = "./log/log.txt"
losses_path = "./log/losses_log.txt"
valid_result_dir = "./log/valid_result"
logger = get_logger(log_path)