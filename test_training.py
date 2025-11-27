import os, sys, subprocess

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)  # ensure 'core' is importable

    cmd = [
        sys.executable,
        "scripts/02_training_and_evaluation/01_train_sol_edge_predictor.py",
        "data_path=data/samples",
        "model=gcnn",
        "model.num_conv_layers=10",
        "model.num_dense_layers=2",
        "model.hidden_layer_dim=20",
        "out_dir=trained_models",
        "seed=0",
        "cross_validate=true",
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
