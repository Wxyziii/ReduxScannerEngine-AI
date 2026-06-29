# Redux Maker v1 image-model selection

The recommended Lumina-Image-2.0 checkpoint remains an optional high-tier model. Its declared 12 GB VRAM requirement exceeds the release test PC's 8 GB RTX 3070, so it cannot be the tested default on that hardware.

Redux Maker v1 uses `segmind/SSD-1B` revision `60987f37e94cd59c36b1cba832b9f97b57395a10` as the balanced default. Its upstream model card identifies the model as Apache-2.0 and Diffusers-compatible. Model Manager downloads only the pinned fp16 Diffusers components and records a local tree checksum. Model files remain upstream downloads and are not committed or bundled.

This substitution does not weaken validation: generated PNG dimensions and formats remain engine-enforced, and the vision worker separately checks prompt/reference consistency.
