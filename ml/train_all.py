"""
Master Training Orchestrator for FairScreen:
Trains and evaluates exclusively on authentic empirical datasets:
1. LabHC/bias_in_bios (138k authentic biographies) & sumin-yu/PopResume (2,500 population resumes)
2. 962 authentic multi-domain resumes for zero-utility-loss verification
"""
from ml.bias_audit import run_consolidated_bias_audit
from ml.real_benchmark import run_all_real_benchmarks

def run_training_pipeline():
    print("=" * 70)
    print("FAIRSCREEN: TRAINING ON 100% AUTHENTIC EMPIRICAL DATASETS")
    print("=" * 70)
    
    print("\n[Stage 1/2] Auditing and training on Bias in Bios & PopResume...")
    bias_results = run_consolidated_bias_audit()
    
    print("\n[Stage 2/2] Evaluating on 962 authentic multi-domain resumes...")
    real_results = run_all_real_benchmarks()
    
    print("\n" + "=" * 70)
    print("SUCCESS: All models trained on real data and artifacts saved to models/!")
    print("=" * 70)
    return {
        "bias_audit": bias_results,
        "real_resumes": real_results
    }

if __name__ == "__main__":
    run_training_pipeline()

