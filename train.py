import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualTranslationDataset, causal_mask
from model import build_tf
from config import get_config, get_weights_file

from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

def get_sentences(ds, lang):
    """
    ds: dataset
    lang: Language
    """
    for item in ds:
        yield item['translation'][lang]



def get_or_build_tokenizer(config, ds, lang):
    """
    config: config file, contains tokenizer file info
    ds: Dataset
    lang: flag for checking which language the tpokenizer is being built for 
    """

    path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens= ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(path))
    else:
        tokenizer = Tokenizer.from_file(str(path))

    return tokenizer  

def get_ds(config):
    """
    config: config file of the model
    """
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # tokenizers
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tgt_tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split dataset
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size  = len(ds_raw) - train_ds_size
    train, val = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualTranslationDataset(train, src_tokenizer, tgt_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualTranslationDataset(val, src_tokenizer, tgt_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # calc max seq_len
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt = tgt_tokenizer.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src))
        max_len_tgt = max(max_len_tgt, len(tgt))

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer  

def get_model(config, src_vocab_len, tgt_vocab_len):
    model = build_tf(src_vocab_len, tgt_vocab_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Dataloaders
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)

    # model
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # preload model if exists
    epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file(config, str(epoch))
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    # label smoothing (prevents overfitting): takes probability of highest and decreases it by the specified value, dsitributing this value over the rest of the classes

    # training loop 
    for ep in range(epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Epoch: {ep}')
        for batch in batch_iterator:
            enc_input = batch['enc_input'].to(device) # (B x S)
            dec_input = batch['dec_input'].to(device) # (B x S)
            src_mask = batch['enc_mask'].to(device)
            tgt_mask = batch['dec_mask'].to(device)

            enc_out = model.encode(enc_input, src_mask) # (b x s x d)
            dec_out = model.decode(dec_input, enc_out, src_mask, tgt_mask) # (b x s x d)
            out = model.project(dec_out) # (b x s x tgt_vocab_size)

            label = batch['label'].to(device)

            # cross-entropy works on targets and input logits
            # target-> (b * s,), input logits -> ([b * s] x d) 
            loss = loss_fn(out.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # logging
            writer.add_scalar('Train loss', loss.item(), global_step)
            writer.flush() # flushes event file to disk       
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1 # for tensorboard

        # Save model for this epoch
        model_filename = get_weights_file(config, str(ep))
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optmizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)

            

