# save as predict.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'sentiment_model', local_files_only=True
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'sentiment_model', local_files_only=True
    )
    
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        truncation=True, 
        max_length=96,
        padding='max_length'
    ).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    pred_class = outputs.logits.argmax(dim=1).item() + 1  # 1-5
    confidence = probs[0][pred_class - 1].item()
    
    return {
        'rating': pred_class,
        'confidence': f'{confidence:.1%}',
        'all_probs': {i+1: f'{p:.1%}' for i, p in enumerate(probs[0])}
    }

if __name__ == '__main__':
    tests = [
        "This course was amazing! Best I've ever taken!",
        "Terrible waste of time. Very boring.",
        "It was okay, nothing special.",
        "Good course but could be better organized.",
        "Absolutely fantastic! Highly recommend!"
    ]
    
    for text in tests:
        result = predict(text)
        print(f"\n'{text[:50]}...'")
        print(f"  → Predicted: {result['rating']} stars ({result['confidence']})")