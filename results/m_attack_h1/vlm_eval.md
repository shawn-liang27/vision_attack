# VLM transfer test: does the VLM still see the dog?

| image | llava-1.5-7b-hf |
|---|---|
| baseline.png | 🐕 seen |
| h1_dose_response.png | 🐕 seen |
| pad0.05.png | 🐕 seen |
| pad0.15.png | 🐕 seen |
| pad0.35.png | 🐕 seen |
| pad0.png | 🐕 seen |
| pad1.png | 🐕 seen |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 7/7 images still show the dog

## Full answers

### llava-hf/llava-1.5-7b-hf

**baseline.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**h1_dose_response.png** — detected=True
  - *binary*: Yes
  - *describe*: A graph shows the percentage of image-based attacks and the percentage of attacks that use a masked update.
  - *animals*: There are no animals in this image. The image is focused on a graph showing the percentage of attacks and the percentage of successful attacks.

**pad0.05.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0.15.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a striped blanket.
  - *animals*: There is a dog in this image.

**pad0.35.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a book in front of it.
  - *animals*: There is a dog in this image.

**pad0.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a smile on its face.
  - *animals*: There is a dog in this image.

**pad1.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch in front of a window.
  - *animals*: There is a dog in this image.

