import unittest
import numpy as np
import trimesh
from wg_baseline.measurements import measure, calibrated, errors, summarize

class Measurements(unittest.TestCase):
    def test_known_cylinder_and_height_calibration(self):
        mesh=trimesh.creation.cylinder(radius=.2,height=1.8,sections=256)
        mesh.vertices=mesh.vertices[:,[0,2,1]]
        result=measure(mesh)
        self.assertAlmostEqual(result['height_mm'],1800,places=5)
        self.assertAlmostEqual(result['waist_section_mm'],2*np.pi*.2*1000,delta=.1)
        scaled=mesh.copy(); scaled.vertices*=1.2
        self.assertAlmostEqual(measure(calibrated(scaled,1.8))['chest_section_mm'],result['chest_section_mm'],places=5)
        self.assertAlmostEqual(errors(measure(scaled),result)['height_mm'],360,places=5)

    def test_translation_invariance(self):
        mesh=trimesh.creation.box(extents=[.4,1.8,.2])
        before=measure(mesh)
        mesh.apply_translation([4,5,6])
        for key,value in measure(mesh).items(): self.assertAlmostEqual(value,before[key])

    def test_missing_predictions_count_as_missing(self):
        summary=summarize([{'metric':{'height_mm':-10}}, {'sam_failure':'bad'}],'metric')
        self.assertEqual(summary['height_mm']['valid'],1)
        self.assertEqual(summary['height_mm']['total'],2)
        self.assertEqual(summary['height_mm']['mae_mm'],10)
        self.assertIsNone(summary['waist_section_mm']['mae_mm'])

if __name__=='__main__': unittest.main()
