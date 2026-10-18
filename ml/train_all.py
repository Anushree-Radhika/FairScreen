"""
Master Training Orchestrator for FairScreen:
Trains and evaluates exclusively on authentic empirical datasets:
1. 10,174 real candidates from Kaggle AI Recruitment Pipeline Dataset
2. 962 authentic multi-domain resumes for zero-utility-loss verification
"""
from ml.train_recruitment_dataset import train_recruitment_models
from ml.real_benchmark import run_all_real_benchmarks

def run_training_pipeline():
    print("=" * 70)
    print("FAIRSCREEN: TRAINING ON 100% AUTHENTIC EMPIRICAL DATASETS")
    print("=" * 70)
    
    print("\n[Stage 1/2] Training on 10,174 real resumes (Kaggle Recruitment Pipeline)...")
    rec_results = train_recruitment_models()
    
    print("\n[Stage 2/2] Evaluating on 962 authentic multi-domain resumes...")
    real_results = run_all_real_benchmarks()
    
    print("\n" + "=" * 70)
    print("SUCCESS: All models trained on real data and artifacts saved to models/!")
    print("=" * 70)
    return {
        "recruitment": rec_results,
        "real_resumes": real_results
    }

if __name__ == "__main__":
    run_training_pipeline()

