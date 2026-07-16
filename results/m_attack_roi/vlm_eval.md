# VLM transfer test: does the VLM still see the dog?

| image | llava-1.5-7b-hf |
|---|---|
| baseline.png | 🐕 seen |
| dose_response.png | — missed |
| pad0.05.png | 🐕 seen |
| pad0.15.png | 🐕 seen |
| pad0.35.png | 🐕 seen |
| pad0.png | 🐕 seen |
| pad1.png | — missed |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 5/7 images still show the dog

## Full answers

### llava-hf/llava-1.5-7b-hf

**baseline.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**dose_response.png** — detected=False
  - *binary*: No
  - *describe*: A graph shows the difference between a single surrogate and a multiple surrogate.
  - *animals*: There are no animals in this image. The image features a graph or chart with two lines, one red and one green, and a single surrogate line labeled "LAVA 1.5."

**pad0.05.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a smile on its face.
  - *animals*: There is a dog in this image.

**pad0.15.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch and looking at the camera.
  - *animals*: There is a dog in this image.

**pad0.35.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog statue is sitting on a couch next to a blanket.
  - *animals*: There is a dog in the image.

**pad0.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch and smiling.
  - *animals*: There is a dog in this image.

**pad1.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a clock on it and a bookshelf in the background.
  - *animals*: There are no animals in this image. The image features a couch with a clock on it, a chair, and a clock on the wall.

