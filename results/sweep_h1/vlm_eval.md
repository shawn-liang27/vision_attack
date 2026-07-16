# VLM transfer test: does the VLM still see the dog?

| image | llava-1.5-7b-hf |
|---|---|
| baseline.png | 🐕 seen |
| pad0.05_seed0.png | 🐕 seen |
| pad0.05_seed1.png | — missed |
| pad0.05_seed2.png | 🐕 seen |
| pad0.05_seed3.png | 🐕 seen |
| pad0.05_seed4.png | 🐕 seen |
| pad0.15_seed0.png | — missed |
| pad0.15_seed1.png | — missed |
| pad0.15_seed2.png | — missed |
| pad0.15_seed3.png | — missed |
| pad0.15_seed4.png | — missed |
| pad0.35_seed0.png | — missed |
| pad0.35_seed1.png | — missed |
| pad0.35_seed2.png | 🐕 seen |
| pad0.35_seed3.png | — missed |
| pad0.35_seed4.png | — missed |
| pad0_seed0.png | 🐕 seen |
| pad0_seed1.png | 🐕 seen |
| pad0_seed2.png | 🐕 seen |
| pad0_seed3.png | 🐕 seen |
| pad0_seed4.png | 🐕 seen |
| pad1_seed0.png | — missed |
| pad1_seed1.png | — missed |
| pad1_seed2.png | — missed |
| pad1_seed3.png | — missed |
| pad1_seed4.png | — missed |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 11/26 images still show the dog

## Full answers

### llava-hf/llava-1.5-7b-hf

**baseline.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**pad0.05_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A stuffed dog is sitting on a couch next to a clock.
  - *animals*: There is a dog in the image.

**pad0.05_seed1.png** — detected=False
  - *binary*: No
  - *describe*: A green couch with a mirror on top of it.
  - *animals*: There are no animals in this image. The image features a couch with pillows, a clock, and a mirror.

**pad0.05_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A stuffed dog is sitting on a couch next to a table.
  - *animals*: There is a dog in the image.

**pad0.05_seed3.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog statue is placed on a table in front of a couch.
  - *animals*: In this image, there is a dog.

**pad0.05_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A couch with a clock on the wall behind it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on the wall.

**pad0.15_seed0.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a blanket on it and a clock on the wall.
  - *animals*: There are no animals in this image. The image features a couch with a clock on it, a chair, and a cat.

**pad0.15_seed1.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a stack of books on it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a stack of books on a table.

**pad0.15_seed2.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a pillow and a vase on it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a vase on a table.

**pad0.15_seed3.png** — detected=False
  - *binary*: No
  - *describe*: A green couch with a pillow on it.
  - *animals*: There are no animals in this image. The image features a couch with pillows, a clock, and a vase.

**pad0.15_seed4.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a stuffed animal on it and a stack of books on the arm.
  - *animals*: There are no animals in this image. The image features a couch with a stuffed animal on it, and a bookshelf with books and a teddy bear.

**pad0.35_seed0.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a clock on it and a white table in front of it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on a table.

**pad0.35_seed1.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a white pillow and a clock on the wall.
  - *animals*: There are no animals in this image. The image features a couch with a stuffed animal on it, a clock, and a bookshelf.

**pad0.35_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A stuffed animal is sitting on a couch next to a clock.
  - *animals*: There is a stuffed animal, specifically a stuffed dog, in the image.

**pad0.35_seed3.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a clock on it and a wooden clock on the floor.
  - *animals*: There are no animals in this image. The image features a couch with a clock on it, a wooden clock, and a wooden clock on the wall.

**pad0.35_seed4.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a blanket on it and a bookshelf in the background.
  - *animals*: There are no animals in this image. The image features a couch with pillows and blankets, a bookshelf, and a wall with a picture.

**pad0_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a blanket on it.
  - *animals*: There is a dog in this image.

**pad0_seed1.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a blanket on it.
  - *animals*: There is a dog in this image.

**pad0_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a blanket on it.
  - *animals*: There is a dog in this image.

**pad0_seed3.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a blanket on it.
  - *animals*: There is a dog in this image.

**pad0_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch in front of a window.
  - *animals*: There is a dog in this image.

**pad1_seed0.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a blanket on it and a clock on the wall.
  - *animals*: There are no animals in this image. The image features a couch with a blanket on it, a clock, and a picture on the wall.

**pad1_seed1.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a mirror on the table in front of it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a mirror on a table.

**pad1_seed2.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a blanket on it and a clock on the wall.
  - *animals*: There are no animals in this image. The image features a couch with a blanket on it, a clock, and a vase.

**pad1_seed3.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a pillow and a clock on a table in the background.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on a table.

**pad1_seed4.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a white pillow and a black pillow on it.
  - *animals*: There are no animals in this image. The image features a couch with a pillow and a blanket on it, and a clock on the wall.

