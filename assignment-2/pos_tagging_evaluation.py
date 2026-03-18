import nltk
from nltk.corpus import brown
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

print("Downloading required NLTK data...")
nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

print("\n" + "="*80)
print("POS TAGGING EVALUATION ON BROWN CORPUS")
print("="*80)

print("\n1. Loading Brown Corpus...")
tagged_sents = brown.tagged_sents(tagset='universal')
print(f"   Total sentences: {len(tagged_sents)}")
print(f"   Sample sentence: {tagged_sents[0][:5]}...")

train_size = int(len(tagged_sents) * 0.8)
train_sents = tagged_sents[:train_size]
test_sents = tagged_sents[train_size:]

print(f"   Training sentences: {len(train_sents)}")
print(f"   Test sentences: {len(test_sents)}")

print("\n2. Training POS Taggers...")

print("\n   a) Tagger 1: Unigram Tagger with Default Tagger Backoff")
default_tagger = DefaultTagger('NOUN')
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)
print("      Training completed!")

print("\n   b) Tagger 2: Trigram Tagger with Bigram and Unigram Backoff")
t0 = DefaultTagger('NOUN')
t1 = UnigramTagger(train_sents, backoff=t0)
t2 = BigramTagger(train_sents, backoff=t1)
trigram_tagger = TrigramTagger(train_sents, backoff=t2)
print("      Training completed!")

def evaluate_tagger(tagger, test_sentences, tagger_name):
    print(f"\n3. Evaluating {tagger_name}...")
    
    y_true = []
    y_pred = []
    
    for sent in test_sentences:
        words = [word for word, tag in sent]
        true_tags = [tag for word, tag in sent]
        
        predicted = tagger.tag(words)
        pred_tags = [tag for word, tag in predicted]
        
        y_true.extend(true_tags)
        y_pred.extend(pred_tags)
    
    unique_tags = sorted(set(y_true + y_pred))
    print(f"   Unique POS tags: {unique_tags}")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        labels=unique_tags,
        average=None,
        zero_division=0
    )
    
    print(f"\n   Per-class metrics:")
    print(f"   {'Tag':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("   " + "-"*60)
    
    for i, tag in enumerate(unique_tags):
        print(f"   {tag:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}")
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print("\n   " + "="*60)
    print(f"   {'MACRO AVG':<10} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    print("   " + "="*60)
    
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    print(f"\n   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': {
            'tags': unique_tags,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    }

print("\n" + "="*80)
print("TAGGER 1: UNIGRAM TAGGER WITH DEFAULT BACKOFF")
print("="*80)
results_tagger1 = evaluate_tagger(unigram_tagger, test_sents, "Unigram Tagger")

print("\n" + "="*80)
print("TAGGER 2: TRIGRAM TAGGER WITH BACKOFF CHAIN")
print("="*80)
results_tagger2 = evaluate_tagger(trigram_tagger, test_sents, "Trigram Tagger")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"\n{'Metric':<25} {'Tagger 1 (Unigram)':<20} {'Tagger 2 (Trigram)':<20}")
print("-"*65)
print(f"{'Accuracy':<25} {results_tagger1['accuracy']:<20.4f} {results_tagger2['accuracy']:<20.4f}")
print(f"{'Macro Precision':<25} {results_tagger1['macro_precision']:<20.4f} {results_tagger2['macro_precision']:<20.4f}")
print(f"{'Macro Recall':<25} {results_tagger1['macro_recall']:<20.4f} {results_tagger2['macro_recall']:<20.4f}")
print(f"{'Macro F1-Score':<25} {results_tagger1['macro_f1']:<20.4f} {results_tagger2['macro_f1']:<20.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if results_tagger2['macro_f1'] > results_tagger1['macro_f1']:
    print(f"Tagger 2 (Trigram) performs better with {results_tagger2['macro_f1']:.4f} macro-F1")
    print(f"vs Tagger 1 (Unigram) with {results_tagger1['macro_f1']:.4f} macro-F1")
else:
    print(f"Tagger 1 (Unigram) performs better with {results_tagger1['macro_f1']:.4f} macro-F1")
    print(f"vs Tagger 2 (Trigram) with {results_tagger2['macro_f1']:.4f} macro-F1")
print("="*80)
