import tempfile
from pathlib import Path
import unittest
from redux_ai.image_worker import generate
from redux_ai.vision import validate


class VisionTests(unittest.TestCase):
    def test_prompt_and_dimension_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            image=Path(directory)/"purple.png"; generate("purple tracer", 16, 64, image)
            accepted=validate(image,"purple tracer",16,64); self.assertEqual(accepted["status"],"passed"); self.assertTrue(accepted["dimensionOk"])
            rejected=validate(image,"purple tracer",32,64); self.assertEqual(rejected["status"],"rejected")

    def test_reference_similarity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); image=root/"a.png"; generate("red Hit Effect",8,8,image)
            result=validate(image,"red Hit Effect",8,8,image); self.assertEqual(result["referenceScore"],1.0)


if __name__ == "__main__": unittest.main()
