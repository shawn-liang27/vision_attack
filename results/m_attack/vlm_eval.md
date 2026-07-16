# VLM transfer test: does the VLM still see the dog?

| image | llava-1.5-7b-hf |
|---|---|
| baseline.png | 🐕 seen |
| m_attack.png | 🐕 seen |
| m_attack_eps16.png | — missed |

## Detection rate per model

- **llava-hf/llava-1.5-7b-hf**: 2/3 images still show the dog

## Full answers

### llava-hf/llava-1.5-7b-hf

**baseline.png** — detected=True
  - *binary*: Yes
  - *describe*: A brown and white dog is sitting on a couch.
  - *animals*: There is a dog in this image.

**m_attack.png** — detected=True
  - *binary*: Yes
  - *describe*: A graph shows the difference between a dog and a couch.
  - *animals*: There are no animals in this image. The image is focused on a graph or chart with a red line and a blue line, which are likely representing different data points or trends.

**m_attack_eps16.png** — detected=False
  - *binary*: No
  - *describe*: A green couch with a clock on top of it.
  - *animals*: There are no animals in this image. The image features a couch with pillows and a clock on a table.

