import tempfile
from pathlib import Path
import unittest
from redux_ai.image_worker import diffusion_dimensions, generate, generate_blank, png_dimensions


class ImageWorkerTests(unittest.TestCase):
    def test_diffusion_work_size_preserves_vertical_tracer_aspect(self):
        width,height=diffusion_dimensions(64,256);self.assertEqual(width*4,height);self.assertGreaterEqual(width*height,500*500)
    def test_exact_tracer_and_hit_effect_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracer = generate("purple tracer", 32, 128, root / "tracer.png")
            hit = generate("red Hit Effect blood spray", 64, 64, root / "hit.png")
            self.assertEqual(png_dimensions((root / "tracer.png").read_bytes()), (32, 128))
            self.assertEqual(png_dimensions((root / "hit.png").read_bytes()), (64, 64))
            self.assertEqual((tracer["width"], hit["height"]), (32, 64))

    def test_reference_changes_output_and_blank_preserves_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference = root / "reference.bin"; reference.write_bytes(b"user-reference")
            result = generate("custom tracer", 8, 16, root / "generated.png", reference)
            blank = generate_blank(8, 16, root / "blank.png")
            self.assertTrue(result["referenceUsed"]); self.assertTrue(blank["entryPreservationRequired"])
            self.assertEqual(png_dimensions((root / "blank.png").read_bytes()), (8, 16))


if __name__ == "__main__": unittest.main()
