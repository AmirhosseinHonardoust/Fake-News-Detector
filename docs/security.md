# Security

## Loading model artifacts is code execution

The trained model is stored as a joblib/pickle file (`outputs/pipeline.joblib`).
Deserializing a pickle runs arbitrary Python code, so **loading a `.joblib` file
is equivalent to executing whatever code it contains**.

Only load artifacts that you trained yourself or obtained from a fully trusted
source. Never load a pipeline downloaded from an untrusted or unverified location.

## Integrity checks (checksum sidecars)

Training writes a SHA-256 sidecar next to the pipeline:

```text
outputs/pipeline.joblib
outputs/pipeline.joblib.sha256
```

`model_compat.load_pipeline()` verifies this sidecar by default. If the artifact
no longer matches its sidecar, loading fails with a clear error instead of
silently using a changed file:

```python
from model_compat import load_pipeline

pipeline = load_pipeline("outputs/pipeline.joblib")          # verifies if a sidecar exists
pipeline = load_pipeline("outputs/pipeline.joblib", verify=False)  # bypass the check
```

You can (re)generate or check a sidecar manually:

```python
from model_compat import write_checksum, verify_checksum

write_checksum("outputs/pipeline.joblib")    # create the sidecar
verify_checksum("outputs/pipeline.joblib")   # True / False / None (no sidecar)
```

### Threat model and limitations

The checksum protects against **accidental corruption, truncation, or a casual
file swap**. It does **not** protect against a determined attacker, who could
replace both the artifact and its sidecar. It is therefore a convenience and
integrity aid, not a trust boundary. The rule above — only load artifacts you
trust — still applies.

The bundled `outputs/pipeline.joblib` in version control may predate this feature
and may not ship with a sidecar; in that case loading proceeds without an
integrity check. Retrain locally (`python src/train_model.py`) to regenerate the
artifact together with its sidecar.
