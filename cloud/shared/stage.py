"""Copy exactly the code a cloud training job needs into one folder.

Author: Yves Schillings.

Run:  python cloud/shared/stage.py

Azure Machine Learning uploads a job's "code" folder, and SageMaker AI uploads its
"source_dir". Both should receive only the training code, never local results,
models or credentials. This script builds that folder, cloud/build/src:

  machine_learning.py   copy of examples/01_machine_learning.py (a module name cannot start with a digit)
  deep_learning.py      copy of examples/02_deep_learning.py
  fictional_data.py, page_images.py, mlflow_tracking.py   the helpers they import
  train_triage.py, train_page_cnn.py                     the cloud entry scripts
  requirements.txt      the packages the entry scripts need inside the job

The copies are the same code as the local examples, so the cloud run trains the
same model with the same settings.
"""
import shutil
from pathlib import Path

CLOUD = Path(__file__).resolve().parents[1]
EXAMPLES = CLOUD.parent / "examples"
BUILD = CLOUD / "build" / "src"

FILES = {EXAMPLES / "01_machine_learning.py": "machine_learning.py",
         EXAMPLES / "02_deep_learning.py": "deep_learning.py",
         EXAMPLES / "fictional_data.py": "fictional_data.py",
         EXAMPLES / "page_images.py": "page_images.py",
         EXAMPLES / "mlflow_tracking.py": "mlflow_tracking.py",
         CLOUD / "shared" / "train_triage.py": "train_triage.py",
         CLOUD / "shared" / "train_page_cnn.py": "train_page_cnn.py",
         CLOUD / "shared" / "job_requirements.txt": "requirements.txt"}


def stage(build=BUILD):
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    for src, name in FILES.items():
        shutil.copyfile(src, build / name)
    return sorted(p.name for p in build.iterdir())


if __name__ == "__main__":
    print("Staged in", BUILD, ":", ", ".join(stage()))
