from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_dataset

if __name__ == "__main__":
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    dataset_name = 'paws'
    dataset_variant = 'labeled_final'  
    
    model = SentenceTransformer(model_name)
    dataset = load_dataset(dataset_name, dataset_variant)
    dataset = dataset.remove_columns(['id'])
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    loss = CosineSimilarityLoss(model)
    
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["label"],
        name="paws-validation"
    )
    
    args = SentenceTransformerTrainingArguments(
        output_dir='fine-tuned-bi-encoder',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps", 
        save_steps=1000,
        logging_steps=100,
        load_best_model_at_end=True
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,   
        loss=loss,
        evaluator=evaluator,
    )
    
    trainer.train()

