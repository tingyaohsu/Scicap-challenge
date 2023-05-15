from transformers import AutoConfig, AutoTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments, set_seed
import torch, nltk
import numpy as np
import os, argparse
import logging, evaluate
from datasets import load_dataset

logger = logging.getLogger(__name__)

def main(args):

	dataset = load_dataset('json', data_files={'train':'split_json_data_dir/train.json',
												'val':'split_json_data_dir/val.json',
												'test':'split_json_data_dir/test-wg.json'},
								field='annotations')
	# print(dataset.keys())
	# print(dataset['val'][0])
	column_names = dataset['train'].column_names

	set_seed(args.seed)

	config = AutoConfig.from_pretrained(args.model_name_or_path)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=args.use_fast_tokenizer)
	model = PegasusForConditionalGeneration.from_pretrained(args.model_name_or_path,config=config)
	print("model config:",model.config)
	padding = "max_length" if args.pad_to_max_length else False
        
	def preprocess_data(examples):
		"""
		Prepare input data for model fine-tuning
		"""
		inputs, targets = [], []
		for i in range(len(examples["caption_no_index"])):
			if examples["caption_no_index"][i] and examples["paragraph"][i][0]:
				inputs.append(examples["paragraph"][i][0])
				targets.append(examples["caption_no_index"][i])

		encoding_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
		labels = tokenizer(targets, max_length=args.max_target_length, padding=padding, truncation=True)

		if padding == "max_length" and args.ignore_pad_token_for_loss:
			labels["input_ids"] = [
				[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
			]

		encoding_inputs["labels"] = labels["input_ids"]
		return encoding_inputs

	print("overwriting cache:",args.overwrite_cache)

	training_args = TrainingArguments(
		output_dir=args.output_dir,          # output directory
		overwrite_output_dir=True,
		learning_rate=5e-5,               # learning rate
		load_best_model_at_end=True,
		metric_for_best_model=args.metric_name,
		num_train_epochs=args.epochs,              # total # of training epochs
		per_device_train_batch_size=args.train_batch_size ,  # batch size per device during training
		per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		eval_accumulation_steps=args.eval_accumulation_steps,
		do_train=True,
		do_eval=True,
		warmup_steps=500,                # number of warmup steps for learning rate scheduler
		weight_decay=0.01,               # strength of weight decay
		evaluation_strategy="steps",
		eval_steps=200,
		save_total_limit=10,
		save_steps=200,
		fp16=True,
	)

	if args.do_train:
		# with training_args.main_process_first(desc="train dataset map pre-processing"):
		train_dataset = dataset['train']
		# train_dataset = train_dataset.select(range(1000))
		train_dataset = train_dataset.map(
			preprocess_data,
			batched=True,
			num_proc=args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not args.overwrite_cache,
			desc="Running tokenizer on train dataset",
		)
	if args.do_eval:
		# with training_args.main_process_first(desc="validation dataset map pre-processing"):
		val_dataset = dataset['val']
		# val_dataset = val_dataset.select(range(100))
		val_dataset = val_dataset.map(
			preprocess_data,
			batched=True,
			num_proc=args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not args.overwrite_cache,
			desc="Running tokenizer on val dataset",
		)
	if args.do_predict:
		# with training_args.main_process_first(desc="prediction dataset map pre-processing"):
		test_dataset = dataset['test']
		# test_dataset = test_dataset.select(range(100))
		test_dataset = test_dataset.map(
			preprocess_data,
			batched=True,
			num_proc=args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not args.overwrite_cache,
			desc="Running tokenizer on test dataset",
		)
	# print("train_dataset:",train_dataset[0])
	# print("val_dataset:",val_dataset[0])
	# print("test_dataset:",test_dataset[0])

	# Metric
	metric = evaluate.load(args.metric_name)

	def postprocess_text(preds, labels):
		preds = [pred.strip() for pred in preds]
		labels = [label.strip() for label in labels]

		# rougeLSum expects newline after each sentence
		preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
		labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

		return preds, labels
	
	def compute_metrics(eval_preds):
		preds, labels = eval_preds
		if isinstance(preds, tuple):
			preds = preds[0]
		
		preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
		preds = np.argmax(preds, axis=-1)
		# print("preds:",preds.shape)
		decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
		labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
		decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

		# Some simple post-processing
		decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

		result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
		result = {k: round(v * 100, 4) for k, v in result.items()}
		prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
		result["gen_len"] = np.mean(prediction_lens)
		return result

	

	trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, #if training_args.do_train else None,
        eval_dataset=val_dataset, #if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics #if training_args.predict_with_generate else None,
    )

	# Training
	if args.do_train:
		checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
		train_result = trainer.train()
		trainer.save_model()  # Saves the tokenizer too for easy upload

		metrics = train_result.metrics
		# print("metrics:",metrics)
        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
		metrics["train_samples"] = len(train_dataset)
		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()

    # Evaluation
	results = {}
	if args.do_eval:
		logger.info("*** Evaluate ***")
		metrics = trainer.evaluate(metric_key_prefix="eval")
		metrics["eval_samples"] = len(val_dataset)

		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", metrics)

	if args.do_predict:
		logger.info("*** Predict ***")
		predict_results = trainer.predict(test_dataset, metric_key_prefix="predict")
		# print("predict_results:",predict_results)
		metrics = predict_results.metrics
		# print("metrics:",metrics)

		trainer.log_metrics("predict", metrics)
		trainer.save_metrics("predict", metrics)

		predictions = predict_results.predictions	
		if isinstance(predictions, tuple):
			predictions = predictions[0]	
		# print("predictions:",predictions.shape)
		predictions = np.argmax(predictions, axis=-1)
		predictions = tokenizer.batch_decode(
			predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
		)
		predictions = [pred.strip() for pred in predictions]
		output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
		with open(output_prediction_file, "w") as writer:
			writer.write("\n".join(predictions))

if __name__=='__main__':
    
	'''create argument parser'''
	parser = argparse.ArgumentParser(description='Process some integers.')
	# for data
	parser.add_argument('--train_file', type=str, default=None, help='train file')
	parser.add_argument('--val_file', type=str, default=None, help='val file')
	parser.add_argument('--test_file', type=str, default=None, help='test file')
	parser.add_argument('--max_source_length', type=int, default=512, help='max input length')
	parser.add_argument('--max_target_length', type=int, default=100, help='max output length')
	# parser.add_argument('--num_beams', type=int, default=None, help='beam search width')
	parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True, help='ignore pad token for loss')
	# parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
	parser.add_argument('--pad_to_max_length', type=bool, default=True, help='pad to max length')
	parser.add_argument('--preprocessing_num_workers', type=int, default=16, help='preprocessing num workers')
	parser.add_argument('--overwrite_cache', type=bool, default=False, help='overwrite cache')
	# for model
	parser.add_argument('--model_name_or_path', type=str, default='google/pegasus-arxiv', help='model name or path')
	parser.add_argument('--use_fast_tokenizer', type=bool, default=True, help='use fast tokenizer')
	# for training
	parser.add_argument('--output_dir', type=str, default='./output', help='output directory')
	parser.add_argument('--seed', type=int, default=42, help='random seed')
	parser.add_argument('--do_train', type=bool, default=True, help='do train and eval')
	parser.add_argument('--do_eval', type=bool, default=True, help='do eval')
	parser.add_argument('--do_predict', type=bool, default=True, help='do predict')
	parser.add_argument('--metric_name', type=str, default="rouge", help='metric name')
	parser.add_argument('--ckpt_path', type=str, default=None, help='checkpoint path')
	# parser.add_argument('--random_iter', type=int, default=1000, help='random iteration times')
	parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
	# parser.add_argument('--batch_size', type=int, default=16, help='batch size')
	parser.add_argument('--train_batch_size', type=int, default=16, help='train batch size')
	parser.add_argument('--eval_batch_size', type=int, default=2, help='eval batch size')
	parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='gradient accumulation steps')
	parser.add_argument('--eval_accumulation_steps', type=int, default=16, help='eval accumulation steps')
	parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
	parser.add_argument('--warmup_steps', type=int, default=500,help='warmup steps')
	parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')

	args = parser.parse_args()
			
	main(args)
