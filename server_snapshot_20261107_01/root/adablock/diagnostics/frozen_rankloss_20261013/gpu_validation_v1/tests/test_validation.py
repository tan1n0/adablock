"""Validation mode must be explicit and preserve exact scene/document sums on resume."""
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from rankloss_diagnostic.core import Means
from rankloss_diagnostic.data import sha256
from rankloss_diagnostic.run import json_write
from rankloss_diagnostic.validation import LABEL, check_validation_source, validate_committed_shard


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.workspace=Path(__file__).resolve().parents[1]
        self.temp=tempfile.TemporaryDirectory(dir=self.workspace,prefix='validation-test-')
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.root.resolve().relative_to(self.workspace)
        self.temp.cleanup()

    def test_validation_requires_exact_frozen_manifest_and_all_layers(self):
        manifest=self.root/'manifest.json'; json_write(manifest,{'actual':'selection-val'})
        config={'data':str(self.root)}
        dataset=SimpleNamespace(metadata={'layers':list(range(28))})
        receipt={'manifest_hashes':{'validation':sha256(manifest)}}
        self.assertEqual(check_validation_source(config,dataset,receipt),sha256(manifest))
        receipt['manifest_hashes']['validation']='0'*64
        with self.assertRaisesRegex(ValueError,'exact frozen'):check_validation_source(config,dataset,receipt)
        receipt['manifest_hashes']['validation']=sha256(manifest)
        dataset.metadata['layers']=[12,20,27]
        with self.assertRaisesRegex(ValueError,'all 28'):check_validation_source(config,dataset,receipt)

    def test_merge_preserves_same_document_across_shards_and_missing_values(self):
        one,two,all_rows=Means(),Means(),Means()
        samples=[(0,.1,None,True),(0,.3,5,False),(1,.9,2,False)]
        for index,(doc,cov,required,unreachable) in enumerate(samples):
            row={'coverage':cov,'required_k':required,'unreachable':unreachable,'candidate_shortfall':cov<.2}
            (one if index==0 else two).add(('overall','RankLoss',4),doc,row)
            all_rows.add(('overall','RankLoss',4),doc,row)
        merged=Means()
        # JSON roundtrip also tests integer document IDs converted to strings.
        merged.merge(json.loads(json.dumps(one.export())))
        merged.merge(json.loads(json.dumps(two.export())))
        keys=('group','method','budget')
        for a,b in zip(merged.rows(keys),all_rows.rows(keys)):
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                if isinstance(a[key],float):self.assertAlmostEqual(a[key],b[key])
                else:self.assertEqual(a[key],b[key])
        scene,doc=list(merged.rows(keys))
        self.assertAlmostEqual(scene['coverage'],1.3/3)
        self.assertAlmostEqual(doc['coverage'],.55)
        self.assertAlmostEqual(scene['candidate_shortfall'],1/3)
        self.assertEqual(scene['required_k_available_scenes'],2)

    def test_atomic_commit_validation_detects_corruption_and_signature_change(self):
        with gzip.open(self.root/'scenes.jsonl.gz','wt') as stream:stream.write('{}\n')
        json_write(self.root/'aggregation.json',{'analysis_split':LABEL,'coverage':[],'quality':[]})
        saved={'signature':'same','scene_output_sha256':sha256(self.root/'scenes.jsonl.gz'),
               'aggregation_sha256':sha256(self.root/'aggregation.json')}
        json_write(self.root/'summary.json',saved)
        self.assertEqual(validate_committed_shard(self.root,'same'),saved)
        with self.assertRaisesRegex(ValueError,'signature'):validate_committed_shard(self.root,'different')
        (self.root/'scenes.jsonl.gz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'corrupt'):validate_committed_shard(self.root,'same')


if __name__=='__main__':unittest.main()
