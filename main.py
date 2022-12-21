import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from data.data_proc import DefaultDataset, enlang, zhlang, SOS_token, EOS_token, UNK_token, PAD_token
from config import DefaultConfig
from utils import sequence_mask, DefaultTBWriter
from models.transformer import Transformer
import logging
from torchtext.data import bleu_score



logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'
                    )

logger = logging.getLogger('NMT')
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler = logging.FileHandler('NMT.log')
handler.setFormatter(formatter)
logger.addHandler(handler)


def collate_fn(batch):
    N = len(batch)
    L = config.max_len
    srcs = torch.ones((L, N), dtype=torch.long) * PAD_token
    trgs = torch.ones((L, N), dtype=torch.long) * PAD_token
    src_valid_lens = []
    trg_valid_lens = []
    for idx, (src, trg) in enumerate(batch):
        if len(src) > L:
            src = src[: L]
        if len(trg) > L:
            trg = trg[: L]
        src_valid_lens.append(len(src))
        trg_valid_lens.append(len(trg))
        srcs[: src_valid_lens[-1], idx] = torch.LongTensor(src)
        trgs[: trg_valid_lens[-1], idx] = torch.LongTensor(trg)
    src_mask = sequence_mask(torch.LongTensor(src_valid_lens), max_len=L)
    trg_mask = sequence_mask(torch.LongTensor(trg_valid_lens), max_len=L)
    return srcs, trgs, src_mask, trg_mask


tb_writer = DefaultTBWriter()
prefix_path = 'data/en-zh/{}'
train_dataset = DefaultDataset(prefix_path.format('train.en'), prefix_path.format('train.zh'))
valid_dataset = DefaultDataset(prefix_path.format('valid.en'), prefix_path.format('valid.zh'))
test_dataset = DefaultDataset(prefix_path.format('test.en'), prefix_path.format('test.zh'))

config = DefaultConfig(vocab_size=len(enlang.idx2word), output_dim=len(zhlang.idx2word))
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn, drop_last=True)


steps_per_log = 100

model = Transformer(config.vocab_size,
                    config.output_dim,
                    config.embed_dim,
                    config.num_heads,
                    config.ffn_input_dim,
                    config.ffn_hidden_dim,
                    config.normalized_shape,
                    config.dropout,
                    config.num_layers,
                    config.max_len).to(config.device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
optimizer = Adam(model.parameters(), lr=config.lr)
logger.info('开始训练......')
for epoch in range(config.epochs):
    model.train()
    score_sum = 0
    loss_sum = 0
    item_cnt = 0
    for src, trg, src_key_padding_mask, trg_key_padding_mask in train_dataloader:
        optimizer.zero_grad()
        src = src.to(config.device)
        trg = trg.to(config.device)
        src_key_padding_mask = src_key_padding_mask.to(config.device)
        trg_key_padding_mask = trg_key_padding_mask.to(config.device)
        input_trg = torch.cat((torch.LongTensor([SOS_token]).unsqueeze(0).expand(1, trg.shape[1]), trg[:-1, :].cpu()), dim=0).to(config.device)
        output = model(src, trg, src_key_padding_mask, trg_key_padding_mask)
        
        
        
        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        loss_sum += loss.cpu().detach().item()
        item_cnt += 1
        if(item_cnt == steps_per_log):
            loss_sum /= item_cnt
            score_sum /= item_cnt
            step = tb_writer.add_scalar('loss/train', loss_sum)
            logger.info(f'|train|epoch{epoch}|step: {step * steps_per_log}|loss: {loss_sum}')
            score_sum = 0
            loss_sum = 0
            item_cnt = 0


    model.eval()
    score_sum = 0
    loss_sum = 0
    item_cnt = 0
    for src, trg, src_key_padding_mask, trg_key_padding_mask in valid_dataloader:
        src = src.to(config.device)
        trg = trg.to(config.device)
        src_key_padding_mask = src_key_padding_mask.to(config.device)
        trg_key_padding_mask = trg_key_padding_mask.to(config.device)
        input_trg = torch.cat((torch.LongTensor([SOS_token]).unsqueeze(0).expand(1, trg.shape[1]), trg[:-1, :].cpu()), dim=0).to(config.device)
        output = model(src, trg, src_key_padding_mask, trg_key_padding_mask)
        
        reference =  trg.transpose(0, 1).unsqueeze(1).tolist()
        for i in range(len(reference)):
            for j in range(len(reference[i])):
                for k in range(len(reference[i][j])):
                    reference[i][j][k] = zhlang.idx2word[reference[i][j][k]]
        candidate =  torch.argmax(output, dim=-1).transpose(0, 1).tolist()
        for i in range(len(candidate)):
            for j in range(len(candidate[i])):
                candidate[i][j] = zhlang.idx2word[candidate[i][j]]
        score_sum == bleu_score(candidate, reference)
        
        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)
        loss = criterion(output, trg)
        loss_sum += loss.cpu().detach().item()
        item_cnt += 1
        
        if(item_cnt == steps_per_log):
            loss_sum /= item_cnt
            score_sum /= item_cnt
            step = tb_writer.add_scalar('loss/valid', loss_sum)
            tb_writer.add_scalar('BLEU/valid', score_sum)
            logger.info(f'|valid|epoch{epoch}|step: {step * steps_per_log}|loss: {loss_sum}|BLEU: {score_sum}')
            score_sum = 0
            loss_sum = 0
            item_cnt = 0
    
    
    score_sum = 0
    loss_sum = 0
    item_cnt = 0
    for src, trg, src_key_padding_mask, trg_key_padding_mask in test_dataloader:
        src = src.to(config.device)
        trg = trg.to(config.device)
        src_key_padding_mask = src_key_padding_mask.to(config.device)
        trg_key_padding_mask = trg_key_padding_mask.to(config.device)
        input_trg = torch.cat((torch.LongTensor([SOS_token]).unsqueeze(0).expand(1, trg.shape[1]), trg[:-1, :].cpu()), dim=0).to(config.device)
        output = model(src, trg, src_key_padding_mask, trg_key_padding_mask)
        
        reference =  trg.transpose(0, 1).unsqueeze(1).tolist()
        for i in range(len(reference)):
            for j in range(len(reference[i])):
                for k in range(len(reference[i][j])):
                    reference[i][j][k] = zhlang.idx2word[reference[i][j][k]]
        candidate =  torch.argmax(output, dim=-1).transpose(0, 1).tolist()
        for i in range(len(candidate)):
            for j in range(len(candidate[i])):
                candidate[i][j] = zhlang.idx2word[candidate[i][j]]
        score_sum == bleu_score(candidate, reference)
        
        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)
        loss = criterion(output, trg)
        loss_sum += loss.cpu().detach().item()
        item_cnt += 1
        
        if(item_cnt == steps_per_log):
            loss_sum /= item_cnt
            score_sum /= item_cnt
            step = tb_writer.add_scalar('loss/test', loss_sum)
            tb_writer.add_scalar('BLEU/test', score_sum)
            logger.info(f'|test|epoch{epoch}|step: {step * steps_per_log}|loss: {loss_sum}|BLEU: {score_sum}')
            score_sum = 0
            loss_sum = 0
            item_cnt = 0
    
    checkpoint = {"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch}
    path_checkpoint = "./checkpoints/checkpoint_{}_epoch.pkl".format(epoch)
    torch.save(checkpoint, path_checkpoint)

logger.info('结束训练...')