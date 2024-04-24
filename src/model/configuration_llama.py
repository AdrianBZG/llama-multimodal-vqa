from transformers import PretrainedConfig, AutoConfig

from constants import LORA_CONFIG


class MultimodalLlamaConfig(PretrainedConfig):
    model_type = "multimodal_llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_model_id=None,
        text_model_id=None,
        ignore_index=-100,
        load_in_4bit=False,
        projector_hidden_act="gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        freeze_multimodal_projector=False,
        freeze_language_model=False,
        freeze_vision_model=False,
        tokenizer_len=None,
        lora_config=LORA_CONFIG,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.load_in_4bit = load_in_4bit

        self.vision_model_id = vision_model_id
        self.text_model_id = text_model_id

        self.freeze_multimodal_projector = freeze_multimodal_projector
        self.freeze_language_model = freeze_language_model
        self.freeze_vision_model = freeze_vision_model
        self.tokenizer_len = tokenizer_len
        self.lora_config = lora_config

        # Vision feature selection
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        # Instantiate the pretraining configs for text and vision models
        if text_model_id is not None:
            text_config = AutoConfig.from_pretrained(text_model_id)
            self.text_config = text_config
        else:
            self.text_config = None

        if vision_model_id is not None:
            vision_config = AutoConfig.from_pretrained(vision_model_id)
            if vision_config.model_type == "clip":
                vision_config = vision_config.vision_config
            self.vision_config = vision_config
        else:
            self.vision_config = None
