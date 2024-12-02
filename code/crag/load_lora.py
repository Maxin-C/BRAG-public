from peft import PeftModel
import os

class LoadLoraModel():
    def __init__(self, model, adapter_path) -> None:
        self.model = model
        if os.path.exists(adapter_path):
            self.adapter_path = adapter_path
        else:
            print("Lora loading failed, adapter path not exits")
            self.adapter_path = None
    
    def load_lora_adapter(self):
        if self.adapter_path is not None:
            return PeftModel.from_pretrained(self.model, self.adapter_path)
        else:
            return self.model
        