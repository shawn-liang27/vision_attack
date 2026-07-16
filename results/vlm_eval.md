# VLM transfer test: does the VLM still see the dog?

| image | llava-1.5-7b-hf |
|---|---|
| V_match_eps16.png | 🐕 seen |
| V_match_eps4.png | 🐕 seen |
| V_match_eps8.png | 🐕 seen |
| V_match_suppress_eps16.png | 🐕 seen |
| V_match_suppress_eps4.png | 🐕 seen |
| V_match_suppress_eps8.png | 🐕 seen |
| X_match_eps16.png | 🐕 seen |
| X_match_eps4.png | 🐕 seen |
| X_match_eps8.png | 🐕 seen |
| baseline_eps0.png | 🐕 seen |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 10/10 images still show the dog

## Full answers

### llava-hf/llava-1.5-7b-hf

**V_match_eps16.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with its mouth open.
  - *animals*: There is a dog in this image.

**V_match_eps4.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a smile on its face.
  - *animals*: There is a dog in this image.

**V_match_eps8.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with its mouth open.
  - *animals*: There is a dog in this image.

**V_match_suppress_eps16.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with its mouth open.
  - *animals*: There is a dog in the image.

**V_match_suppress_eps4.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a smile on its face.
  - *animals*: There is a dog in this image.

**V_match_suppress_eps8.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a smile on its face.
  - *animals*: There is a dog in this image.

**X_match_eps16.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a white collar.
  - *animals*: There is a dog in this image.

**X_match_eps4.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**X_match_eps8.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**baseline_eps0.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

