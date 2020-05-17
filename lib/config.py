
import os
import sys
from my_py_toolkit.log.logger import get_logger
home = os.path.expanduser(".")
train_file = os.path.join(home, "data", "squad", "train-v1.1.json")
dev_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
test_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
glove_word_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

device = "cuda" # cpu、 cuda

mode = "train" # train, classify, debug

bert_path = "./data/model/bert"
vocab_file = "./data/model/bert/vocab.txt"
bert_config = "./data/model/bert/bert_config.json"
use_segment_embedding = True
use_pretrained_bert = True
# bert_path = "./data/model/RoBERTa-wwm-ext-large"
# vocab_file = "./data/model/RoBERTa-wwm-ext-large/vocab.txt"

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
is_save_features = False
path_save_feature = "./data/cmrc2018_features"
is_test_with_test_dev_dataset = False

do_lower_case = True

dataset_name = "cmcr2018"
dataset_dir = {
  "cmcr2018": "./data/cmrc2018_squal_style"
}

model_class_name = "ModelBaseLine"
model_package = "lib"
model_name = "model_baseline"

glove_char_size = 94 #Corpus size for Glove
glove_word_size = int(2.2e6) #Corpus size for Glove
glove_dim = 300 #Embedding dimension for Glove
char_dim = 64 #Embedding dimension for char
bert_dim = 768 #Embedding dimension for char
embedding_trainable_dim = 512 # 可训练词向量维度
embedding_trainable_model = "./model/embedding_trainable_model.bin" # 可训练词向量维度

context_length_limit = 512 #Limit length for paragraph
ques_length_limit = 64 #Limit length for question
ans_length_limit = 30 #Limit length for answers
char_length_limit = 16 #Limit length for character
word_count_limit = -1 #Min count for word
char_count_limit = -1 #Min count for char
context_kernel_size = 5
question_kernel_size = 5
doc_stride = 128

is_train = True
is_continue = False
model_dir = "./model"
continue_checkpoint = 2700 # 500 的时候 loss 到达最低处 2.0+
capacity = 15000 #Batch size of dataset shuffle
num_threads = 4 #Number of threads in input pipeline
is_bucket = False #build bucket batch iterator or not
bucket_range = [40, 401, 40] #the range of bucket

data_size = 0.01 # 使用数据集的大小
batch_size = 6 #Batch size
num_steps = 10000 #Number of steps
checkpoint = 50 # 200 #checkpoint to save and evaluate the model
period = 100 #period to save batch loss
val_num_batches = 2 #Number of batches to evaluate the model
val_num_steps = sys.maxsize # 100
test_num_batches = 2 #Number of batches to evaluate the model
test_num_steps = 100
dropout = 0.1 #Dropout prob across the layers
dropout_char = 0.05 #Dropout prob across the layers
grad_clip = 5.0 #Global Norm gradient clipping rate
learning_rate = 0.01# 3e-7 #Learning rate
lr_warm_up_num = 1000 #Number of warm-up steps of learning rate
T_MAX = 1000
min_lr = 1e-5
warmup_proportion = 0.1
ema_decay = 0.9999 #Exponential moving average decay
beta1 = 0.9 #Beta 1
beta2 = 0.999 #Beta 2
early_stop = 10 #Checkpoints for early stop
d_model = 96 #Dimension of connectors of each layer
num_heads = 12 #Number of heads in multi-head attention
epochs = 1000 # The epoch of train.
start_epoch = 0
steps_num = 5000 # The num of train step in one epoch.
interval_save = 50 # The interval of save model.
min_loss = 0 # The scope of loss.可能出现 loss 非常大的情况
max_loss = None
max_postion = 512
attention_probs_dropout_prob = 0.1
attention_use_bias = False
encoder_hidden_layers = 11
encoder_intermediate_dim = 3072
encoder_dropout_prob = 0.1
hidden_act = "gelu"
use_ema = False

# Extensions (Uncomment corresponding line in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
pretrained_char = False #Whether to use pretrained char embedding

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
fasttext = False #Whether to use fasttext

is_only_save_params = True

record_interval_steps = 500
log_path = "../log/log.txt"
losses_path = "../log/losses_log.txt"
valid_result_dir = "../log/valid_result"
less_loss_path = "../data/less_loss.json"
high_loss_path = "../data/high_loss.json"
logger = get_logger(log_path)

# Conv cf
use_conv = True
chan_in = 768
chan_out=768
kernel = 7

# data visualization
font_file = "./data/font/fangsong.ttf"
visual_data_dir = "./runs"
visual_loss = True
visual_gradient = True
visual_parameter = True
visual_optimizer = True
visual_valid_result = True
visual_gradient_dir = "../log/gradients"
visual_parameter_dir = "../log/parameters"
visual_loss_dir = "../log/losses"
visual_optimizer_dir = "../log/optimizer"
visual_valid_result_dir = "../log/valid"
