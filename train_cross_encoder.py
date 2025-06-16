from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from datasets import load_dataset


if __name__ == "__main__":
    model_name = 'microsoft/deberta-v3-small'
    dataset_name = 'paws'
    dataset_variant = 'labeled_final'

    # model_kwargs = {'force_download': True, 'resume_download': False}
    model = CrossEncoder(model_name, num_labels=2)
    dataset = load_dataset(dataset_name, dataset_variant)
    dataset = dataset.remove_columns(['id'])
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    eval_sentence_pairs = [[item['sentence1'], item['sentence2']] for item in eval_dataset]
    eval_labels = [int(item['label']) for item in eval_dataset]
    
    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=eval_sentence_pairs,
        labels=eval_labels,
        name="paws-validation"
    )
    
    args = CrossEncoderTrainingArguments(
        output_dir='fine-tuned-cross-encoder',
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

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,   
        evaluator=evaluator,
    )
    
    trainer.train()