# VLM transfer test: does the VLM still see the dog?

| image | llava-1.5-7b-hf |
|---|---|
| baseline.png | 🐕 seen |
| pad0.05_seed0.png | 🐕 seen |
| pad0.05_seed1.png | 🐕 seen |
| pad0.05_seed2.png | 🐕 seen |
| pad0.05_seed3.png | 🐕 seen |
| pad0.05_seed4.png | 🐕 seen |
| pad0.15_seed0.png | 🐕 seen |
| pad0.15_seed1.png | 🐕 seen |
| pad0.15_seed2.png | 🐕 seen |
| pad0.15_seed3.png | 🐕 seen |
| pad0.15_seed4.png | 🐕 seen |
| pad0.35_seed0.png | 🐕 seen |
| pad0.35_seed1.png | — missed |
| pad0.35_seed2.png | — missed |
| pad0.35_seed3.png | — missed |
| pad0.35_seed4.png | — missed |
| pad0_seed0.png | 🐕 seen |
| pad0_seed1.png | 🐕 seen |
| pad0_seed2.png | 🐕 seen |
| pad0_seed3.png | 🐕 seen |
| pad0_seed4.png | 🐕 seen |
| pad1_seed0.png | — missed |
| pad1_seed1.png | — missed |
| pad1_seed2.png | 🐕 seen |
| pad1_seed3.png | — missed |
| pad1_seed4.png | 🐕 seen |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 19/26 images still show the dog

## Full answers

### llava-hf/llava-1.5-7b-hf

**baseline.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0.05_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch and smiling.
  - *animals*: There is a dog in this image.

**pad0.05_seed1.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a collar on.
  - *animals*: There is a dog in this image.

**pad0.05_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A corgi dog is sitting on a couch and smiling.
  - *animals*: There is a dog in this image.

**pad0.05_seed3.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a collar on.
  - *animals*: There is a dog in this image.

**pad0.05_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0.15_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A stuffed dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0.15_seed1.png** — detected=True
  - *binary*: Yes
  - *describe*: A stuffed dog is sitting on a couch.
  - *animals*: There is a stuffed dog in the image.

**pad0.15_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A stuffed dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0.15_seed3.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog statue is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0.15_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A stuffed dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0.35_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A couch with a dog statue on it.
  - *animals*: There is a dog in the image, sitting on a couch.

**pad0.35_seed1.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a white throw pillow on it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a chair with a stuffed animal on it.

**pad0.35_seed2.png** — detected=False
  - *binary*: No
  - *describe*: A green couch with a pillow on it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on the wall.

**pad0.35_seed3.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a clock on it and a vase in front of it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on the wall.

**pad0.35_seed4.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a blanket on it and a clock on the wall.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on the wall.

**pad0_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0_seed1.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a smile on its face.
  - *animals*: There is a dog in this image.

**pad0_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a smile on its face.
  - *animals*: There is a dog in this image.

**pad0_seed3.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a smile on its face.
  - *animals*: There is a dog in this image.

**pad0_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch and smiling.
  - *animals*: There is a dog in this image.

**pad1_seed0.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a blanket on it and a clock on the wall.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on a table.

**pad1_seed1.png** — detected=False
  - *binary*: No
  - *describe*: A stuffed animal is sitting on a couch next to a bookshelf.
  - *animals*: There are no animals in this image. The image features a couch with a stuffed animal on it, a mirror, and a bookshelf.

**pad1_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog statue is placed on a couch, with a bookshelf in the background.
  - *animals*: There is a dog in the image, sitting on a couch.

**pad1_seed3.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a white vase on a table in front of it.
  - *animals*: There are no animals in this image. The image features a couch with a mirror on a table in front of it.

**pad1_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A couch with a white blanket on it and a lamp next to it.
  - *animals*: There are no animals in this image. The image features a couch with a blanket on it, a lamp, and a teddy bear.

