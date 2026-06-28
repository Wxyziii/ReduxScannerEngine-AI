import json
import tempfile
from pathlib import Path
import unittest
from redux_ai.runtime import Hardware, ModelManager, ResourceBudget, Scheduler, recommend_pack


class RuntimeTests(unittest.TestCase):
    def hardware(self): return Hardware("test",8,4,16.0,"gpu",8.0,100.0,True,True,True,True,False)
    def test_budget_queues_instead_of_overloading(self):
        scheduler=Scheduler(self.hardware(),ResourceBudget(max_ram_gb=4,max_vram_gb=4,max_parallel_models=1))
        self.assertEqual(scheduler.submit({"id":"planner","ramGb":2,"vramGb":0}),"active")
        self.assertEqual(scheduler.submit({"id":"image","ramGb":2,"vramGb":3}),"queued")
    def test_parallel_when_budget_allows(self):
        scheduler=Scheduler(self.hardware(),ResourceBudget(max_ram_gb=12,max_vram_gb=8,max_parallel_models=2))
        self.assertEqual(scheduler.submit({"id":"planner","ramGb":1,"vramGb":0}),"active")
        self.assertEqual(scheduler.submit({"id":"vision","ramGb":1,"vramGb":1}),"active")
    def test_download_requires_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/"manifest.json"; manifest.write_text(json.dumps({"models":[]}),encoding="utf-8")
            with self.assertRaises(PermissionError): ModelManager(manifest,root/"models").install("missing",False)
    def test_private_release_token_is_not_persisted_in_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/"manifest.json"; manifest.write_text(json.dumps({"models":[]}),encoding="utf-8")
            manager=ModelManager(manifest,root/"models",github_token="secret")
            self.assertEqual(manager.status(),[])
            self.assertNotIn("secret",json.dumps(manager.status()))
    def test_pack_recommendation(self): self.assertEqual(recommend_pack(self.hardware()),"full")


if __name__ == "__main__": unittest.main()
