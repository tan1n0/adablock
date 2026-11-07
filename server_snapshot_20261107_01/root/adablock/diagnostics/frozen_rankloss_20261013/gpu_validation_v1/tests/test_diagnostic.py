import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from recovered.block_ranker import load_ranker
from rankloss_diagnostic.core import (Means, attention_output_error, audit_baseline_scores,
                                     coverage_curve, required_prefix, stable_order,
                                     validate_attention, validate_scene)
from rankloss_diagnostic.data import LocalAuditDataset, check_independent_test, sha256
from rankloss_diagnostic.prepare import resolve_config
from rankloss_diagnostic.run import evaluate, json_write, repository_baselines
from rankloss_diagnostic.smoke_fixture import create_fixture


class CoreTests(unittest.TestCase):
    def test_stable_tie_break_uses_block_id_not_input_order(self):
        self.assertEqual(stable_order(torch.tensor([2.,2.,1.]),torch.tensor([8,2,5])).tolist(),[1,0,2])
        with self.assertRaises(ValueError): stable_order(torch.tensor([float('nan')]),torch.tensor([2]))
        with self.assertRaises(ValueError): stable_order(torch.ones(2),torch.tensor([2,2]))

    def test_raw_mass_and_exact_prefix_with_fixed(self):
        curve = coverage_curve(torch.tensor([.05,.1,.15]),.7,torch.tensor([0,1,2]))
        self.assertAlmostEqual(float(curve[1]),.75)
        self.assertEqual(required_prefix(curve,.7,4)['required_k'],0)
        self.assertEqual(required_prefix(curve,.89,2)['required_k'],3)
        self.assertTrue(required_prefix(curve,.89,2)['over_budget_cap'])
        self.assertIsNone(required_prefix(curve,.89,2)['within_cap_k'])
        unreachable = required_prefix(torch.tensor([.5,.6],dtype=torch.float64),.95,64)
        self.assertIsNone(unreachable['required_k'])
        self.assertTrue(unreachable['unreachable'])

    def test_oracle_is_mass_upper_bound_and_min_k(self):
        mass, ids = torch.tensor([.05,.2,.1]), torch.tensor([2,3,4])
        oracle = coverage_curve(mass,.65,stable_order(mass,ids))
        other = coverage_curve(mass,.65,torch.tensor([0,2,1]))
        for k in (4,8,16,32,64):
            take = min(k,len(ids))
            self.assertGreaterEqual(float(oracle[take]),float(other[take])-1e-7)
            self.assertAlmostEqual(float(other[take]),1.,places=6)

    def test_scene_and_document_equal_means_and_conditional_denominators(self):
        means = Means()
        means.add(('m',4),0,{'coverage':.2,'required_k':None,'unreachable':True})
        for _ in range(3): means.add(('m',4),1,{'coverage':.8,'required_k':2,'unreachable':False})
        scene, document = list(means.rows(('method','budget')))
        self.assertAlmostEqual(scene['coverage'],.65)
        self.assertAlmostEqual(document['coverage'],.5)
        self.assertEqual(scene['required_k_available_scenes'],3)
        self.assertEqual(document['required_k_available_documents'],1)
        self.assertAlmostEqual(scene['unreachable'],.25)
        self.assertAlmostEqual(document['unreachable'],.5)

    def test_real_per_head_AV_and_renormalization(self):
        a = torch.zeros(7,65); a[:,0] = .2; a[:,32] = .5; a[:,64] = .3
        # Distinct rows catch implementations that average probabilities first.
        a[0,0] = .1; a[0,32] = .7; a[0,64] = .2
        values = torch.zeros(65,128); values[32] = 2.; values[64] = 1.
        mass = a[:,32:48].sum(-1).mean().reshape(1)
        sample = {'token_position':64,'block_size':16,'block_ids':torch.tensor([2]),
                  'teacher_mass':mass,'fixed_mass':1-mass.sum()}
        sidecar = {'format':'adablock_attention_values_v1','attention':a,'values':values}
        validated = validate_attention(sample,sidecar)
        error = attention_output_error(sample,validated,[])
        expected_full = a.double() @ values.double()
        keep = torch.ones(65,dtype=torch.bool); keep[32:48] = False
        sparse = a.double()*keep; sparse /= sparse.sum(-1,keepdim=True)
        expected = torch.linalg.vector_norm(sparse @ values.double()-expected_full)
        self.assertAlmostEqual(error['output_abs_l2'],float(expected))
        self.assertLess(attention_output_error(sample,validated,[2])['output_abs_l2'],2e-6)
        bad = dict(sidecar, attention=a.mean(0))
        with self.assertRaises(ValueError): validate_attention(sample,bad)
        with self.assertRaises(KeyError): validate_attention(sample,{'format':'adablock_attention_values_v1'})
        bad_mass = dict(sample,teacher_mass=torch.tensor([.1]))
        with self.assertRaises(ValueError): validate_attention(bad_mass,sidecar)
        per_head = dict(sidecar, values=values.unsqueeze(0).repeat(7,1,1))
        self.assertAlmostEqual(attention_output_error(sample,validate_attention(sample,per_head),[])['output_abs_l2'],float(expected))
        zero_v = dict(sidecar,values=torch.zeros_like(values))
        self.assertIsNone(attention_output_error(sample,validate_attention(sample,zero_v),[])['output_relative_l2'])

    def test_repository_aliases_are_explicit(self):
        raw = {'cobs_legacy_mean_logits':torch.ones(1),'quest_aabb':torch.ones(1),'oracle':torch.ones(1)}
        self.assertEqual(set(repository_baselines(raw)),{'COBS','Quest'})
        with self.assertRaises(ValueError): repository_baselines({'unverified_cobs':torch.ones(1)})


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.workspace = Path(__file__).resolve().parents[1]
        cls.temp = tempfile.TemporaryDirectory(dir=cls.workspace, prefix='diagnostic-test-')
        cls.root = Path(cls.temp.name)
        cls.config_path = create_fixture(cls.root/'fixture')
        cls.config = resolve_config(cls.config_path)
        cls.dataset = LocalAuditDataset(cls.config['data'])

    @classmethod
    def tearDownClass(cls):
        cls.root.resolve().relative_to(cls.workspace)
        cls.temp.cleanup()

    def test_CPU_load_frozen_FP32_and_forward(self):
        model, payload = load_ranker(self.config['checkpoint'],device='cpu')
        model.float().requires_grad_(False)
        self.assertEqual(sum(p.numel() for p in model.parameters()),61313)
        with torch.inference_mode(): score = model(self.dataset[0])
        self.assertEqual(score.shape,self.dataset[0]['teacher_mass'].shape)
        self.assertTrue(torch.isfinite(score).all())
        self.assertEqual(score.device.type,'cpu')
        self.assertEqual(score.dtype,torch.float32)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_offline_field_validation(self):
        sample = self.dataset[0]
        validate_scene(sample,self.dataset.metadata)
        self.assertEqual(set(audit_baseline_scores(sample)),{'COBS','Quest'})
        for missing in ('minimum','maximum','basis','query'):
            bad = dict(sample); del bad[missing]
            with self.assertRaises(ValueError): validate_scene(bad,self.dataset.metadata)
        bad = dict(sample, fixed_mass=torch.tensor(.9))
        with self.assertRaises(ValueError): validate_scene(bad,self.dataset.metadata)
        bad = dict(sample,block_ids=sample['block_ids']+1)
        with self.assertRaises(ValueError): validate_scene(bad,self.dataset.metadata)

    def test_smoke_outputs_and_exact_K_from_full_rankings(self):
        output = self.root/'result'
        result = evaluate(self.config,output,smoke=True,local_audit=True,max_scenes=12,
                          require_output_error=True,save_rankings=True,plot=True)
        self.assertEqual(result['evaluation_label'],'NONFORMAL_SMOKE')
        self.assertEqual(result['device'],'cpu')
        self.assertEqual(result['scenes'],12)
        self.assertEqual(result['documents'],2)
        self.assertEqual(result['output_error_scene_status_counts'],{'available':12})
        self.assertTrue(result['checkpoint']['frozen'])
        with (output/'scenes.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows),12*4*5)
        for row in rows:
            self.assertEqual(int(row['actual_k']),min(int(row['budget']),int(row['candidate_count'])))
            self.assertLessEqual(float(row['coverage']),1.0002)
        for line in (output/'scenes.jsonl').read_text().splitlines():
            diagnostic = json.loads(line)
            sample = self.dataset[diagnostic['scene_index']]
            id_mass = dict(zip(sample['block_ids'].tolist(),sample['teacher_mass'].tolist()))
            for q in diagnostic['quality']:
                ranking = diagnostic['ranked_block_ids'][q['method']]
                running = float(sample['fixed_mass'])
                expected = 0 if running >= q['target_mass']-1e-7 else None
                if expected is None:
                    for k,b in enumerate(ranking,1):
                        running += id_mass[b]
                        if running >= q['target_mass']-1e-7:
                            expected = k; break
                self.assertEqual(q['required_k'],expected)
        with self.assertRaises(FileExistsError): evaluate(self.config,output,smoke=True,local_audit=True)

    def test_missing_attention_is_unavailable_and_strict_request_fails(self):
        config = dict(self.config); del config['attention_manifest']
        result = evaluate(config,self.root/'no-attention',smoke=True,local_audit=True,max_scenes=2)
        self.assertEqual(result['output_error_scene_status_counts'],{'unavailable_no_attention_and_V':2})
        self.assertTrue(all('output_rmse' not in row for row in result['coverage']))
        with self.assertRaisesRegex(ValueError,'unavailable'):
            evaluate(config,self.root/'requires-attention',smoke=True,local_audit=True,require_output_error=True)

    def test_checkpoint_mismatch_and_formal_synthetic_rejected(self):
        config = dict(self.config)
        bad_receipt = self.root/'bad-receipt.json'
        receipt = json.loads(Path(config['freeze_receipt']).read_text())
        receipt['checkpoint_sha256'] = '0'*64
        json_write(bad_receipt,receipt); config['freeze_receipt'] = str(bad_receipt)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            evaluate(config,self.root/'wrong-hash',smoke=True,local_audit=True)
        with self.assertRaisesRegex(ValueError,'frozen all28'):
            evaluate(self.config,self.root/'formal-synthetic',smoke=False,local_audit=True)

    def test_three_layer_test_cannot_pass_independent_check(self):
        with self.assertRaisesRegex(ValueError,'all 28'):
            check_independent_test({},self.dataset.manifest,{})

    def test_no_middle_scenes_are_excluded(self):
        data = self.root/'no-middle-data'; data.mkdir()
        manifest = json.loads(json.dumps(self.dataset.manifest))
        for entry in manifest['shards']:
            shutil.copyfile(Path(self.config['data'])/entry['file'], data/entry['file'])
        first = manifest['shards'][0]
        path = data/first['file']
        shard = torch.load(path,map_location='cpu',weights_only=True)
        empty = dict(shard['records'][0],token_position=16,block_ids=torch.empty(0,dtype=torch.long),
                     teacher_mass=torch.empty(4,0),fixed_mass=torch.ones(4))
        shard['records'].insert(0,empty); torch.save(shard,path)
        first['records'] += 1; first['scenes'] += 4
        json_write(data/'manifest.json',manifest)
        config = dict(self.config,data=str(data)); del config['attention_manifest']
        result = evaluate(config,self.root/'no-middle-result',smoke=True,local_audit=True,max_scenes=16)
        self.assertEqual(result['skipped_no_middle'],4)
        self.assertEqual(result['scenes'],12)
        self.assertEqual(sum(result['document_scene_counts'].values()),12)

    def test_independent_test_checks_train_val_reserved_and_receipts(self):
        folder = self.root/'splits'; folder.mkdir()
        metadata = dict(self.dataset.metadata,layers=list(range(28)))
        doc = self.dataset.manifest['documents'][0]
        manifest = {'metadata':metadata,'documents':[doc],
                    'shards':[dict(document_index=0,layer_id=l,heads=4) for l in range(28)]}
        data = folder/'data'; data.mkdir(); json_write(data/'manifest.json',manifest)
        config = {'data':str(data)}; hashes = {}
        for name in ('train','validation'):
            other_doc = dict(doc,text_sha256=name,components=[name])
            path = folder/(name+'.json'); json_write(path,{'metadata':metadata,'documents':[other_doc]})
            config[name+'_manifest'] = str(path); hashes[name] = sha256(path)
        reserved = folder/'reserved.json'; json_write(reserved,dict(manifest,metadata=self.dataset.metadata))
        config['reserved_test_manifest'] = str(reserved)
        declaration = {'split':'independent_test','used_for_training':False,'used_for_checkpoint_selection':False,
                       'test_manifest_sha256':sha256(data/'manifest.json'), 'reserved_manifest_sha256':sha256(reserved),
                       'preparation_note':'all28 rebuilt from previously heldout identities'}
        path = folder/'declaration.json'; json_write(path,declaration); config['test_declaration'] = str(path)
        receipt = {'manifest_hashes':hashes,'reserved_test_manifest_sha256':sha256(reserved)}
        self.assertIn('test',check_independent_test(config,manifest,receipt))
        declaration['used_for_checkpoint_selection'] = True; json_write(path,declaration)
        with self.assertRaisesRegex(ValueError,'independent test declaration'):
            check_independent_test(config,manifest,receipt)
        declaration['used_for_checkpoint_selection'] = False; json_write(path,declaration)
        train = Path(config['train_manifest']); json_write(train,{'metadata':metadata,'documents':[doc]})
        receipt['manifest_hashes']['train'] = sha256(train)
        with self.assertRaisesRegex(ValueError,'overlaps train'):
            check_independent_test(config,manifest,receipt)


if __name__ == '__main__':
    unittest.main()
