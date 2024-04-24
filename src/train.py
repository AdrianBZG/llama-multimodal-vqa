"""
Main entrypoint for training the language-vision model.
"""

import argparse
import os
import pathlib
import sys
import logging
from transformers import TrainingArguments

from dataset.data_handling import create_dataset
from model.model_utils import build_model
from utils.utils import get_available_device, set_seed, make_save_folder
from trainer_llama import MultimodalLlamaTrainer


logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model_id',
                        default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Huggingface model path to the language model. Examples:'
                             '- meta-llama/Llama-2-13b-chat-hf'
                             '- lmsys/vicuna-7b-v1.5'
                             '- meta-llama/Meta-Llama-3-8B-Instruct')

    parser.add_argument('--vision_model_id',
                        default='openai/clip-vit-large-patch14',
                        help='Huggingface model path to the vision model. Examples:'
                             '- openai/clip-vit-large-patch14')

    parser.add_argument('--data_path',
                        default='data/llava_instruct_80k.json',
                        help='Path to the visual-instruction dataset for finetuning')

    parser.add_argument('--checkpoint_save_path',
                        default='./model_checkpoints/',
                        help='Path to the folder where to save the model checkpoints during training')

    parser.add_argument('--load_in_4bit', action='store_true')
    parser.set_defaults(load_in_4bit=False)

    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="The batch size to use for training. Default: 16")

    parser.add_argument('--report_to',
                        default='none',
                        choices=['wandb', 'none'],
                        help='Which reporting tool to use. Options: wandb, none. Default: none')

    args = parser.parse_args(sys.argv[1:])
    logging.info(f"Parameters received: {args}")

    # Reproducibility
    set_seed()

    logging.info("Building model stack...")
    model_stack = build_model(text_model_id=args.text_model_id,
                              vision_model_id=args.vision_model_id,
                              freeze_vision_model=True,
                              freeze_language_model=False,
                              freeze_multimodal_projector=False,
                              device=get_available_device(),
                              use_bfloat16=True,
                              load_in_4bit=args.load_in_4bit)

    logging.info("Building data module...")
    data_module = create_dataset(tokenizer=model_stack['tokenizer'],
                                 image_processor=model_stack['processor'].image_processor,
                                 is_multimodal=True,
                                 data_path=args.data_path,
                                 image_folder="data/images",
                                 image_aspect_ratio="pad",
                                 config=model_stack['config'])

    output_dir = make_save_folder(args.checkpoint_save_path)
    logging.info(f"Output dir is: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=1,
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        save_total_limit=1,
        learning_rate=2e-4,
        weight_decay=0.,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=500,
        logging_steps=1,
        evaluation_strategy="no",
        do_eval=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to
    )

    if training_args.gradient_checkpointing:
        model_stack['model'].gradient_checkpointing_enable()

    logging.info("Building trainer...")
    trainer = MultimodalLlamaTrainer(model=model_stack['model'],
                                     tokenizer=model_stack['tokenizer'],
                                     group_by_modality_length=True,
                                     args=training_args,
                                     **data_module)

    logging.info("Starting training...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    logging.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
