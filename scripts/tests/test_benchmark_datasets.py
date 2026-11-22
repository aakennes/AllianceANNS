"""Dataset-neutral planning and IP replay; no ANN executable or real data."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib.datasets import build_experiment
from benchlib.spec import digest_file, fingerprint_inputs, validate_spec
from benchlib.validation import ValidationError
from test_benchmark_validation import Fixture


class DatasetTests(unittest.TestCase):
    def test_profile_pins_driver_assets_and_allows_ef_below_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = json.loads((Path(__file__).resolve().parents[1] / 'configs/datasets/laion10m.json').read_text())
            p = profile['defaults']
            p.update(queries=2, quantization_source='meta', navigation=False,
                     aggregation={'policy':'mean','repeats':1})
            p['shape'].update(vectors=4,dimensions=2,max_layer=0)
            p['cache']['static_bytes'] = 0
            config = root/'profile.json'
            config.write_text(json.dumps(profile))
            assets = {}
            names = ('binary','base','attributes','query','ranges','gt','graph','meta','ml_graph','ml_meta')
            for name in names:
                path = root/name
                path.write_text(name)
                assets[name] = dict(path=str(path),sha256=digest_file(path))
            bindings = dict(schema_version=1, profile_sha256=digest_file(config), assets=assets,
                            lock_paths=[str(root/'exclusive.lock')],curves=[dict(workload='mixed',cache_policy='off',
                            efs=[1,7,10],quantization_fingerprint='12345',assets={n:n for n in names})])
            runtime = root/'runtime.json'
            runtime.write_text(json.dumps(bindings))
            spec = build_experiment(config,runtime)
            validate_spec(spec)
            curve = spec['curves'][0]
            self.assertEqual(curve['efs'],[1,7,10])
            self.assertEqual(curve['bits'],3)
            self.assertEqual(curve['env']['BENCH_EXPECTED_D'],'2')
            self.assertEqual(curve['env']['BENCH_QUANTIZATION_SOURCE'],'meta')
            self.assertNotIn('--quant_checkpoint',curve['argv'])
            self.assertNotIn('--inline_rerank',curve['argv'])
            self.assertNotIn('--rerank_interval_max',curve['argv'])
            self.assertEqual(curve['argv'][curve['argv'].index('--efs_list')+1],'1,7,10')
            del p['navigation']  # Omitted native setting now requires navigation.
            config.write_text(json.dumps(profile))
            bindings['profile_sha256']=digest_file(config)
            runtime.write_text(json.dumps(bindings))
            with self.assertRaisesRegex(ValueError,'pinned navigation asset'):
                build_experiment(config,runtime)
            navigation_file = root/'navigation'
            navigation_file.write_bytes(b'identity-only navigation fixture')
            bindings['assets']['navigation'] = dict(path=str(navigation_file),sha256=digest_file(navigation_file))
            bindings['curves'][0]['assets']['navigation'] = 'navigation'
            runtime.write_text(json.dumps(bindings))
            navigation_spec = build_experiment(config,runtime)
            validate_spec(navigation_spec)
            navigation_curve = navigation_spec['curves'][0]
            navigation_argv = navigation_curve['argv']
            for option,value in dict(navigation_file=str(navigation_file),navigation_ef='64',
                                     navigation_budget='256',navigation_count='16',entry_samples='0',entry_count='0').items():
                self.assertEqual(navigation_argv[navigation_argv.index('--'+option)+1],value)
            self.assertEqual(navigation_curve['expected']['measured']['entry_mode'],'ram_navigation')
            self.assertEqual(navigation_curve['expected']['measured']['guided_entry_bytes'],0)
            pinned = next(a for a in navigation_spec['inputs'] if a['path'] == str(navigation_file))
            self.assertEqual(pinned['sha256'],digest_file(navigation_file))
            fingerprint_inputs(navigation_spec)
            navigation_file.unlink()
            with self.assertRaisesRegex(ValueError,'required asset is missing'):
                fingerprint_inputs(navigation_spec)
            navigation_file.write_bytes(b'identity-only navigation fixture')
            del bindings['assets']['navigation']['sha256']
            runtime.write_text(json.dumps(bindings))
            with self.assertRaisesRegex(ValueError,'pinned SHA256'):
                build_experiment(config,runtime)
            bindings['assets']['navigation']['sha256'] = digest_file(navigation_file)
            p['navigation'] = dict(ef=128,budget=64,count=16)
            config.write_text(json.dumps(profile))
            bindings['profile_sha256']=digest_file(config)
            runtime.write_text(json.dumps(bindings))
            explicit = build_experiment(config,runtime)
            self.assertEqual(explicit['curves'][0]['expected']['measured']['navigation_budget'],64)
            self.assertEqual(explicit['curves'][0]['expected']['measured']['navigation_ef'],128)
            for invalid in (dict(ef=65537,budget=256,count=16), dict(ef=64,budget=65537,count=16)):
                p['navigation'] = invalid
                config.write_text(json.dumps(profile))
                bindings['profile_sha256']=digest_file(config)
                runtime.write_text(json.dumps(bindings))
                with self.subTest(navigation=invalid), self.assertRaisesRegex(ValueError,'navigation requires'):
                    build_experiment(config,runtime)
            p['navigation']=False
            config.write_text(json.dumps(profile))
            bindings['profile_sha256']=digest_file(config)
            runtime.write_text(json.dumps(bindings))
            with self.assertRaisesRegex(ValueError,'navigation is disabled'):
                build_experiment(config,runtime)
            del bindings['curves'][0]['assets']['navigation']
            del bindings['assets']['navigation']
            broken=copy.deepcopy(spec)
            broken['curves'][0]['bits']=8
            with self.assertRaisesRegex(ValueError,'query protocol'):
                validate_spec(broken)
            p['ml_only']=True
            config.write_text(json.dumps(profile))
            bindings['profile_sha256']=digest_file(config)
            del bindings['curves'][0]['assets']['graph']
            runtime.write_text(json.dumps(bindings))
            ml_spec=build_experiment(config,runtime)
            validate_spec(ml_spec)
            self.assertTrue(ml_spec['curves'][0]['ml_only'])
            self.assertIn('--ml_only',ml_spec['curves'][0]['argv'])
            self.assertNotIn('--graph_file',ml_spec['curves'][0]['argv'])
            p['ml_only']=False
            config.write_text(json.dumps(profile))
            bindings['profile_sha256']=digest_file(config)
            runtime.write_text(json.dumps(bindings))
            with self.assertRaisesRegex(ValueError,'missing native'):
                build_experiment(config,runtime)
            bindings['profile_sha256']='0'*64
            runtime.write_text(json.dumps(bindings))
            with self.assertRaisesRegex(ValueError,'pin this dataset profile'):
                build_experiment(config,runtime)

    def test_ip_archive_replay_uses_geometry_and_attribute_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            f.curve.update(native_protocol='alliance-formal-v1',metric='ip',bits=3,
                           input_formats={'base':'fbin','attributes':'ibin'},
                           shape={'vectors':4,'dimensions':2,'edge_bytes':4096,'vector_bytes':4096})
            f.curve['expected']['measured'].update(protocol='alliance-formal-v1',
                    dynamic_cache_clear_per_query=False,dynamic_cache_auto_size=False)
            f.measured.update(protocol='alliance-formal-v1',bits=3,vector_block_bytes=4096,io_bytes=16384)
            for obs in f.observations:
                obs['bytes']=8192
            f.results=[[(0,1.0),(1,1.0)],[(2,-3.0),(1,-1.0)]]
            attr=f.root/'attributes.bin'
            attr.write_bytes(struct.pack('<II',4,1)+attr.read_bytes())
            f.save()
            raw=f.directory/'results.bin'
            raw.write_bytes(struct.pack('<Q',0x414c4c49414e4331)+raw.read_bytes()[8:])
            proof=f.verify()
            row=proof['rows'][0]
            self.assertEqual(row['max_relative_ip_error'],0)
            self.assertEqual(row['io_4k_equivalent_per_query'],2)
            self.assertEqual(row['io_8k_equivalent_per_query'],1)
            self.assertNotIn('max_relative_l2_error',row)
            f.curve['shape']['vector_bytes']=8192
            with self.assertRaisesRegex(ValidationError,'vector block size'):
                f.verify()

    def test_profiles_keep_arxiv_and_laion_protocols_distinct(self):
        directory=Path(__file__).resolve().parents[1]/'configs/datasets'
        arxiv=json.loads((directory/'arxiv2m_observer.json').read_text())['defaults']
        laion=json.loads((directory/'laion10m.json').read_text())['defaults']
        self.assertEqual(arxiv['shape']['centroids'],256)
        self.assertEqual(laion['shape']['centroids'],128)
        self.assertTrue(arxiv['cache']['clear_per_query'])
        self.assertFalse(laion['cache']['clear_per_query'])
        self.assertEqual(arxiv['batch_order'],'ef_then_repeat')
        self.assertEqual(laion['batch_order'],'repeat_then_ef')
        self.assertEqual(arxiv['memory']['mode'],'whole_process')
        self.assertEqual(laion['memory']['mode'],'phase')
        self.assertTrue(arxiv['navigation'])
        self.assertEqual(arxiv['entry'],dict(sampling='range',samples=0,count=0))
        self.assertFalse(laion['navigation'])
        self.assertEqual(laion['entry'],dict(sampling='guided',samples=256,count=4))
        archived=json.loads((directory.parent/'reference/arxiv2m_navigation_observer.json').read_text())['defaults']
        self.assertTrue(archived['navigation'])


if __name__=='__main__':
    unittest.main()
