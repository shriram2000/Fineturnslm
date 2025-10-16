"""
Small Language Model Training Script
For Fashion Product Recommendations
Uses LoRA fine-tuning for efficient training
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import pandas as pd
from tqdm import tqdm

class FashionRecommenderSLM:
    """Train and deploy SLM for personalized product recommendations"""
    
    def __init__(self, model_name="microsoft/phi-2", use_lora=True):
        """
        Initialize the training pipeline
        
        Args:
            model_name: Hugging Face model identifier
                - "microsoft/phi-2" (2.7B params, efficient)
                - "meta-llama/Llama-2-7b-chat-hf" (7B params, powerful)
                - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B params, very fast)
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Initializing {model_name}")
        print(f"📱 Device: {self.device}")
        
    def load_training_data(self, jsonl_path):
        """Load training data from JSONL file"""
        print(f"\n📂 Loading training data from {jsonl_path}...")
        
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict({
            'instruction': [item['instruction'] for item in data],
            'response': [item['response'] for item in data]
        })
        
        print(f"✓ Loaded {len(dataset)} training samples")
        return dataset
    
    def prepare_model_and_tokenizer(self):
        """Load and prepare model and tokenizer"""
        print(f"\n🔧 Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Apply LoRA if enabled
        if self.use_lora:
            print("🎯 Applying LoRA configuration...")
            lora_config = LoraConfig(
                r=16,  # Rank
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],  # Attention layers
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        print("✓ Model and tokenizer ready")
        
    def preprocess_function(self, examples):
        """Preprocess data for training"""
        # Format: <instruction>\n\n<response>
        texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{self.tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding='max_length',
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def train(self, dataset, output_dir="fashion_recommender_model", 
              num_epochs=3, batch_size=4, learning_rate=2e-4):
        """Train the model"""
        print(f"\n🎓 Starting training...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Learning Rate: {learning_rate}")
        
        # Prepare dataset
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing data"
        )
        
        # Split into train/validation (90/10)
        split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=self.device == "cuda",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            warmup_steps=100,
            weight_decay=0.01,
            report_to="none",  # Disable wandb/tensorboard
            push_to_hub=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # Train
        print("\n⏳ Training in progress...")
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ Training completed! Model saved to {output_dir}")
        
        return trainer
    
    def generate_recommendations(self, customer_profile, max_length=500):
        """Generate product recommendations for a customer"""
        
        # Format prompt
        prompt = f"### Instruction:\n{customer_profile}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()
        
        return response
    
    def evaluate_recommendations(self, test_data_path, num_samples=50):
        """Evaluate model on test data"""
        print(f"\n📊 Evaluating model on {num_samples} samples...")
        
        # Load test data
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                test_data.append(json.loads(line))
        
        results = []
        for item in tqdm(test_data, desc="Generating predictions"):
            prediction = self.generate_recommendations(item['instruction'])
            results.append({
                'customer_id': item['customer_id'],
                'instruction': item['instruction'],
                'ground_truth': item['response'],
                'prediction': prediction
            })
        
        # Save results
        pd.DataFrame(results).to_csv('evaluation_results.csv', index=False)
        print("✓ Evaluation complete. Results saved to evaluation_results.csv")
        
        return results


# Complete Training Pipeline
def run_complete_pipeline():
    """Execute the complete training pipeline"""
    print("="*80)
    print("FASHION PRODUCT RECOMMENDATION SLM - TRAINING PIPELINE")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'model_name': "microsoft/phi-2",  # Change to your preferred model
        'training_data': "fashion_recommendations_train.jsonl",
        'output_dir': "fashion_recommender_model",
        'num_epochs': 3,
        'batch_size': 4,
        'learning_rate': 2e-4,
        'use_lora': True
    }
    
    print("\n📋 Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Step 1: Initialize
    slm = FashionRecommenderSLM(
        model_name=CONFIG['model_name'],
        use_lora=CONFIG['use_lora']
    )
    
    # Step 2: Load data
    dataset = slm.load_training_data(CONFIG['training_data'])
    
    # Step 3: Prepare model
    slm.prepare_model_and_tokenizer()
    
    # Step 4: Train
    trainer = slm.train(
        dataset=dataset,
        output_dir=CONFIG['output_dir'],
        num_epochs=CONFIG['num_epochs'],
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate']
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return slm


# Inference Example
def inference_example():
    """Example of using the trained model for inference"""
    
    # Example customer profile
    customer_profile = """Customer Profile:
- Gender: F
- Age: 32 years (26-35)
- Location: New York, United States
- Total Orders: 15
- Total Spending: $2,450.00
- Average Order Value: $163.33
- Favorite Category: Feminine
- Preferred Subcategories: Dresses, T-shirts and Tops, Skirts
- Favorite Colors: Black, White, Navy
- Preferred Sizes: M, S

Recent Purchase History:
- Feminine > Dresses: elegant silk dress with floral pattern (Black, Size: M) - $89.99
- Feminine > T-shirts and Tops: casual cotton top (White, Size: M) - $34.99
- Feminine > Skirts: midi pencil skirt (Navy, Size: M) - $54.99
- Feminine > Dresses: summer sundress (White, Size: M) - $44.99
- Feminine > T-shirts and Tops: striped casual shirt (Black, Size: S) - $29.99

Based on this customer's profile and shopping history, suggest 5 personalized product recommendations:"""
    
    print("\n🎯 INFERENCE EXAMPLE")
    print("="*80)
    print("\nCustomer Profile:")
    print(customer_profile)
    
    # Load trained model
    print("\n📥 Loading trained model...")
    slm = FashionRecommenderSLM(model_name="fashion_recommender_model")
    slm.prepare_model_and_tokenizer()
    
    # Generate recommendations
    print("\n🤖 Generating recommendations...")
    recommendations = slm.generate_recommendations(customer_profile)
    
    print("\n📝 Generated Recommendations:")
    print("-"*80)
    print(recommendations)
    print("-"*80)


if __name__ == "__main__":
    print("""
    Usage:
    
    1. Train the model:
       python train_slm.py --mode train
       
    2. Run inference:
       python train_slm.py --mode inference
       
    3. Evaluate:
       python train_slm.py --mode evaluate
    """)
    
    # Uncomment to run:
    # slm = run_complete_pipeline()
    # inference_example()
