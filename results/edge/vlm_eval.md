# VLM transfer test: does the VLM still see the dog?

| image | llava-1.5-7b-hf |
|---|---|
| baseline.png | 🐕 seen |
| edge_v1_seed0.png | 🐕 seen |
| edge_v1_seed1.png | 🐕 seen |
| edge_v1_seed2.png | — missed |
| edge_v1_seed3.png | 🐕 seen |
| edge_v1_seed4.png | — missed |
| edge_v2_seed0.png | 🐕 seen |
| edge_v2_seed1.png | 🐕 seen |
| edge_v2_seed2.png | 🐕 seen |
| edge_v2_seed3.png | 🐕 seen |
| edge_v2_seed4.png | 🐕 seen |
| pad0.15_seed0.png | — missed |
| pad0.15_seed1.png | — missed |
| pad0.15_seed2.png | 🐕 seen |
| pad0.15_seed3.png | — missed |
| pad0.15_seed4.png | 🐕 seen |
| pad0_seed0.png | 🐕 seen |
| pad0_seed1.png | 🐕 seen |
| pad0_seed2.png | 🐕 seen |
| pad0_seed3.png | 🐕 seen |
| pad0_seed4.png | 🐕 seen |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 16/21 images still show the dog

## Full answers

### llava-hf/llava-1.5-7b-hf

**baseline.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**edge_v1_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A couch with a stuffed animal on it.
  - *animals*: There is a dog in the image, sitting on a chair in the living room.

**edge_v1_seed1.png** — detected=True
  - *binary*: Yes
  - *describe*: A couch with a stuffed dog on it.
  - *animals*: In this image, there is a dog statue or a stuffed dog in the room.

**edge_v1_seed2.png** — detected=False
  - *binary*: No
  - *describe*: A green couch with a pile of blankets on it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a vase on a table.

**edge_v1_seed3.png** — detected=True
  - *binary*: Yes
  - *describe*: A green couch with a statue of a dog on it.
  - *animals*: There is a dog in the image, sitting on a couch next to a statue.

**edge_v1_seed4.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a pillow and a statue on it.
  - *animals*: There are no animals in this image. The image features a couch with pillows, a vase, and a statue of a horse.

**edge_v2_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a picture of a man in the background.
  - *animals*: There is a dog in this image.

**edge_v2_seed1.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a white belly.
  - *animals*: There is a dog in this image.

**edge_v2_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a brown and white coat.
  - *animals*: There is a dog in this image.

**edge_v2_seed3.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a person's hand in the background.
  - *animals*: There is a dog in this image.

**edge_v2_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a book in front of it.
  - *animals*: There is a dog in this image.

**pad0.15_seed0.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a clock on it and a stack of towels on the arm.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on a table.

**pad0.15_seed1.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a pillow on it is in front of a wall with a clock and a bowl on it.
  - *animals*: There are no animals in this image. The focus is on the couch, the clock, and the decorative items in the room.

**pad0.15_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A couch with a blanket on it and a picture of a dog on the wall.
  - *animals*: There are no animals in this image. The image features a couch with a blanket on it, a picture frame, and a clock.

**pad0.15_seed3.png** — detected=False
  - *binary*: No
  - *describe*: A couch with a blanket on it and a picture of a car on the wall.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a picture on the wall.

**pad0.15_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch next to a clock.
  - *animals*: There is a dog in the image, sitting on a couch.

**pad0_seed0.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a blanket on it.
  - *animals*: There is a dog in this image.

**pad0_seed1.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a brown and white color.
  - *animals*: There is a dog in this image.

**pad0_seed2.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch in front of a window.
  - *animals*: There is a dog in this image.

**pad0_seed3.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a brown and white color.
  - *animals*: There is a dog in this image.

**pad0_seed4.png** — detected=True
  - *binary*: Yes
  - *describe*: A dog is sitting on a couch with a blanket on it.
  - *animals*: There is a dog in this image.

