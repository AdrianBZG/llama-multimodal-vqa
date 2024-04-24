from dataset.data_classes import SupervisedDataset, DataCollatorForSupervisedDataset
from dataset.data_utils import get_preprocess_func, preprocess, preprocess_llama_2, preprocess_llama_3


def create_dataset(tokenizer, image_processor, data_path, image_folder, image_aspect_ratio, is_multimodal, config):
    """Make dataset and collator for supervised fine-tuning."""

    preprocess_func = get_preprocess_func(config.text_model_id)
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      image_processor=image_processor,
                                      data_path=data_path,
                                      image_folder=image_folder,
                                      image_aspect_ratio=image_aspect_ratio,
                                      is_multimodal=is_multimodal,
                                      preprocess_func=preprocess_func)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
