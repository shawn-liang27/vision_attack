# VLM transfer test: does the VLM still see the dog?

| image | llava-1.5-7b-hf |
|---|---|
| baseline_eps0.png | 🐕 seen |
| dog2couch_eps128.png | 🐕 seen |
| dog2couch_eps16.png | 🐕 seen |
| dog2couch_eps32.png | 🐕 seen |
| dog2couch_eps64.png | 🐕 seen |
| dog2couch_eps8.png | 🐕 seen |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 6/6 images still show the dog

## Full answers

### llava-hf/llava-1.5-7b-hf

**baseline_eps0.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**dog2couch_eps128.png** — detected=True
  - *binary*: Yes
  - *describe*: A couch with a dog on it that is wearing a shirt.
  - *animals*: There are no animals in this image. The image features a couch with a pillow shaped like a dog.

**dog2couch_eps16.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch and appears to be watching TV.
  - *animals*: There is a dog in the image.

**dog2couch_eps32.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch and watching TV.
  - *animals*: There is a dog in the image, sitting on a couch.

**dog2couch_eps64.png** — detected=True
  - *binary*: Yes
  - *describe*: A stuffed dog is sitting on a couch with a pair of eyes and a mouth.
  - *animals*: There are no animals in this image. The image features a stuffed dog sitting on a couch.

**dog2couch_eps8.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

