"""Training Blueprint for TinyCodeGPT (<4GB RAM Budget)."""

from agent_zed.slm.model import TinyCodeGPT

def train_slm_blueprint(
    dataset_name: str = "nampdn-ai/tiny-codes",
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    lr: float = 5e-4
):
    print("=== INITIALIZING SLM TRAINING BLUEPRINT (Target: <4GB RAM Budget) ===")
    print(f"Dataset: {dataset_name} | Batch size: {batch_size} | Grad Accumulation: {grad_accum_steps}")

    model = TinyCodeGPT(vocab_size=16000, max_len=256, d_model=256, n_heads=4, n_layers=6)
    print(f"TinyCodeGPT Model Parameter Count: {model.param_count / 1e6:.2f}M parameters")
    print("SLM Model and Training Pipeline Ready for dataset streaming!")
    return model

if __name__ == "__main__":
    train_slm_blueprint()
