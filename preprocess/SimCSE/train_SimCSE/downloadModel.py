from transformers import BertTokenizer, BertModel

pretrained_model_name = 'bert-base-uncased'
cache_dir = './PretrainedLM/bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, cache_dir=cache_dir)
model = BertModel.from_pretrained(pretrained_model_name, cache_dir=cache_dir)

tokenizer.save_pretrained(cache_dir)
model.save_pretrained(cache_dir)

print(f"模型和分词器已下载并保存到：{cache_dir}")