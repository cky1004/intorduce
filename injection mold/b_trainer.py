from pathlib import Path
import json

class Trainer:
    def __init__(self, setting: dict = None) -> None:
        self.model = None
        self.artifacts = dict()
        self._results = dict()
        self.artifact_path = setting.get("artifact_path", ".")
        self.dir_model = Path(self.artifact_path)
        self.dir_model.mkdir(parents=True, exist_ok=True)

    def fit(self, fit_args, fit_kwargs) -> None:
        self.model.fit(*fit_args, **fit_kwargs)

    def update_results(self, name, val):
        self._results[name] = val

    def log_model(self, name, type):
        model = getattr(self, name)
        model_path = None

        if type.name == "SCIKIT_LEARN":
            from joblib import dump
            model_path = f"{name}.joblib"
            model_path = self.dir_model / model_path
            dump(model, str(model_path))

        elif type.name == "XGBOOST":
            model_path = f"{name}.json"
            model_path = self.dir_model / model_path
            model.save_model(str(model_path))

        elif type.name == "TENSORFLOW":
            model_path = f"{name}.h5"
            model_path = self.dir_model / model_path
            model.save(str(model_path))

        elif type.name == "PYTORCH":
            from torch import save
            model_path = f"{name}.pt"
            model_path = self.dir_model / model_path
            save(model, str(model_path))

        return model_path

    def log_metric(self, metric):
        self.update_results("metrics", metric)

    def save_train_results(self):
        with open(self.dir_model / 'artifacts.json', "w") as artifacts_file:
            json.dump(self.artifacts, artifacts_file)
            
