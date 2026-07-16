# VLM transfer test: does the VLM still see the cat?

| image | llava-1.5-7b-hf |
|---|---|
| baseline_eps0.png | — missed |
| dog2cat_eps128.png | 🐕 seen |
| dog2cat_eps16.png | 🐕 seen |
| dog2cat_eps32.png | 🐕 seen |
| dog2cat_eps64.png | 🐕 seen |
| dog2cat_eps8.png | 🐕 seen |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 5/6 images still show the cat

## Full answers

### llava-hf/llava-1.5-7b-hf

**baseline_eps0.png** — detected=False
  - *binary*: No
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**dog2cat_eps128.png** — detected=True
  - *binary*: Yes
  - *describe*: A cat is sitting on a couch with a picture of a cat on it.
  - *animals*: There is a cat in this image.

**dog2cat_eps16.png** — detected=True
  - *binary*: Yes
  - *describe*: A black and white cat is sitting on a couch.
  - *animals*: There is a cat in this image.

**dog2cat_eps32.png** — detected=True
  - *binary*: Yes
  - *describe*: A cat is sitting on a couch with its mouth open.
  - *animals*: There is a cat in this image.

**dog2cat_eps64.png** — detected=True
  - *binary*: Yes
  - *describe*: A cat is sitting on a couch and looking at the camera.
  - *animals*: There is a cat in this image.

**dog2cat_eps8.png** — detected=True
  - *binary*: Yes
  - *describe*: A cat is sitting on a couch and yawning.
  - *animals*: There is a cat in this image.

