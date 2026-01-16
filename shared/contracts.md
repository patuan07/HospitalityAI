# API Contract (Do Not Break)

This file describes the **response contract** that the product UI expects.

## Endpoint

`POST /analyze?run_stage5={true|false}`

Request: `multipart/form-data` with file field:

- `image`: `.jpg/.png`

## Response (JSON)

Top-level keys:

- `image_id` : string
- `stages_run` : list of strings (e.g. `["stage1","stage2","stage3","stage4"]`)
- `stage1` : object
  - `prob_made` : float (0..1)
  - `pred_made` : bool
  - `debug` : object (freeform)
- `defects` : object mapping `label -> probability` (Stage 2)
- `localizations` : list (Stage 3)
  - each item:
    - `label` : string
    - `confidence` : float
    - `method` : string
    - `heatmap_path` : string URL (optional)
    - `overlay_path` : string URL (optional)
    - `regions` : list of boxes (optional)
- `alignment_score` : float (0..1) (Stage 4)
- `alignment_pass` : bool
- `alignment_debug` : object
- `stage5` : optional object (only when `run_stage5=true`)
  - `robustness_score` : float (0..1)
  - `details` : object
- `artifacts` : object mapping `artifact_name -> URL`

## Rule for teams

Teams can change internal ML logic, but should keep this response shape stable so:

- judges can evaluate product demos consistently,
- students can swap models without rewriting the UI.
