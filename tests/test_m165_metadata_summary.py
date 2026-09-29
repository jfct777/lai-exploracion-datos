"""Synthetic-only metadata joins, summaries, privacy and unchanged-input guards."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.test_m165_sweep_figures import fixture

SCRIPT = Path(__file__).resolve().parents[1]/'bin/m165_metadata_summary.py'
spec = importlib.util.spec_from_file_location('m165_metadata_summary', SCRIPT)
summary = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)
try:
    import numpy
    import sklearn
    HAVE_STATS = True
except ImportError:
    HAVE_STATS = False


def metadata_rows():
    result = []
    for i in range(5):
        result.append(dict(ID=f'PRIVATE_FIXTURE_SAMPLE_{i}', finestructure_clusters='A' if i < 2 else 'B',
            Region='Unknown' if i == 0 else 'REGION_LABEL', State='Unknown' if i == 1 else 'AM_1',
            Cohort='COHORT_LABEL', Autosomes_African_anc=.1*i, Autosomes_European_anc=1-.1*i,
            Autosomes_Indigenous_anc=0, Autosomes_EastAsian_anc=0,
            finestructure_bigclusters='Exclude' if i == 1 else 'BRA_ADM', clinical_out_of_scope='PRIVATE_CLINICAL'))
    return result


def write_metadata(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.path = self.root/'metadata.tsv'
        self.rows = metadata_rows(); write_metadata(self.path, self.rows)
        self.samples = [row['ID'] for row in self.rows]
        self.settings = summary.settings_from_json(json.dumps(dict(expected_samples=5)))

    def read(self):
        return summary.read_metadata(self.path, 'ID', self.samples, self.settings)

    def test_exact_join_reorders_and_collapses_only_identical_full_rows(self):
        extra = dict(self.rows[0], ID='PRIVATE_EXTRA_SAMPLE')
        write_metadata(self.path, list(reversed(self.rows))+[self.rows[0], extra])
        aligned, audit = self.read()
        self.assertEqual([row['Autosomes_African_anc'] for row in aligned], [.1*i for i in range(5)])
        self.assertEqual((audit['n_metadata_rows'], audit['n_unique_metadata_ids'],
                          audit['n_identical_duplicates_collapsed'], audit['n_extra_metadata_ids']), (7, 6, 1, 1))
        self.assertEqual(audit['n_cohort_matched'], 5)
        self.assertFalse(any('ID' in row or 'clinical_out_of_scope' in row for row in aligned))

    def test_conflicting_unselected_column_is_still_rejected(self):
        write_metadata(self.path, self.rows+[dict(self.rows[0], clinical_out_of_scope='changed')])
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'): self.read()

    def test_missing_sample_and_padded_id_are_rejected_without_printing_id(self):
        write_metadata(self.path, self.rows[:-1])
        with self.assertRaisesRegex(ValueError, 'full saved cohort'): self.read()
        write_metadata(self.path, [dict(self.rows[0], ID=' PRIVATE_FIXTURE_SAMPLE_0'), *self.rows[1:]])
        with self.assertRaisesRegex(ValueError, 'whitespace'): self.read()

    def test_unknown_is_missing_but_exclude_in_other_column_does_not_remove_person(self):
        aligned, audit = self.read()
        self.assertIsNone(aligned[0]['Region']); self.assertIsNone(aligned[1]['State'])
        self.assertEqual(audit['fields']['Region']['n_missing'], 1)
        self.assertEqual(audit['fields']['State']['n_missing'], 1)
        self.assertEqual(len(aligned), 5); self.assertTrue(audit['no_cohort_exclusions'])
        self.assertEqual(aligned[2]['State'], 'AM_1')

    def test_ancestry_bounds_nonfinite_and_wrong_fourway_sum_rejected(self):
        for value in (-.1, 1.1, 'inf', 'nonnumeric'):
            with self.subTest(value=value):
                rows=copy.deepcopy(self.rows); rows[0]['Autosomes_African_anc']=value
                write_metadata(self.path, rows)
                with self.assertRaises(ValueError): self.read()
        rows=copy.deepcopy(self.rows); rows[0]['Autosomes_EastAsian_anc']=.4
        write_metadata(self.path, rows)
        with self.assertRaisesRegex(ValueError, 'sum to one'): self.read()

    def test_missing_numeric_values_are_not_imputed(self):
        rows=copy.deepcopy(self.rows); rows[0]['Autosomes_African_anc']='NA'
        write_metadata(self.path, rows); aligned,audit=self.read()
        self.assertIsNone(aligned[0]['Autosomes_African_anc'])
        self.assertEqual(audit['fields']['Autosomes_African_anc']['n_missing'],1)

    def test_only_explicitly_scoped_columns_and_four_ancestries_allowed(self):
        for patch_settings in ({'categorical_columns':['Sex','finestructure_clusters']},
                               {'ancestry_columns':list(summary.ANCESTRY)[:3]},
                               {'disclose_min_n':0}, {'missing_tokens':['']}, {'unexpected':1}):
            with self.subTest(settings=patch_settings):
                with self.assertRaises(ValueError): summary.settings_from_json(json.dumps(patch_settings))
        selected=summary.settings_from_json(json.dumps({'categorical_columns':['finestructure_clusters']}))
        self.assertEqual(selected['categorical_columns'], ['finestructure_clusters'])


@unittest.skipUnless(HAVE_STATS, 'requires NumPy and scikit-learn in the existing M16.5 image')
class SummaryTests(JoinTests):
    def setUp(self):
        super().setUp(); self.source=fixture(self.root/'source')

    def test_assigned_only_scores_match_exact_permutation_and_missing_denominator(self):
        labels=[0,0,1,1,-1,-1]; ref=['A','A','B','B','A',None]
        scores=summary.reference_scores(labels,ref)
        self.assertEqual(scores['n_assigned_with_reference'],4)
        self.assertEqual(scores['reference_ari'],1.)
        self.assertEqual(scores['reference_nmi'],1.)
        self.assertAlmostEqual(scores['reference_ami'],1.)
        self.assertEqual(scores['weighted_purity_assigned'],1.)
        self.assertEqual(scores['dominant_reference_share_assigned'],.5)
        ref[0]=None; scores=summary.reference_scores(labels,ref)
        self.assertEqual(scores['n_assigned_with_reference'],3)
        self.assertEqual(scores['n_assigned_missing_reference'],1)

    def test_degenerate_and_empty_comparisons_are_explicit_not_false_validation(self):
        degenerate=summary.reference_scores([0,0,0], ['A','A','A'])
        self.assertTrue(degenerate['degenerate_reference_or_partition'])
        self.assertEqual(degenerate['agreement_status'],'DESCRIPTIVE_DEGENERATE')
        empty=summary.reference_scores([-1,-1], ['A','B'])
        self.assertIsNone(empty['reference_ari']);self.assertIsNone(empty['weighted_purity_assigned'])

    def test_full_42_outputs_exact_counts_suppression_and_no_identifiers(self):
        before={str(p.relative_to(self.source)):summary.saved.sha256(p) for p in self.source.rglob('*') if p.is_file()}
        output=self.root/'summary'
        result=summary.run(self.source,self.path,output,settings_json_text=json.dumps({'expected_samples':5}))
        self.assertEqual((result['n_cells'],result['n_graphs'],result['n_cohort']),(42,6,5))
        self.assertEqual(len(result['outputs_sha256']),6)
        self.assertEqual(result['input_sha256_before'],result['input_sha256_after'])
        parts=summary.saved.table(output/'partition_metadata_summary.tsv')
        self.assertEqual(len(parts),42)
        self.assertTrue(all(int(r['n_assigned'])==3 and int(r['n_unassigned'])==2 for r in parts))
        sizes=summary.saved.table(output/'community_sizes.tsv')
        self.assertEqual(len(sizes),42)
        self.assertTrue(all(int(r['n_samples'])==3 for r in sizes))
        stats=summary.saved.table(output/'ancestry_summary.tsv')
        small=[r for r in stats if r['scope']=='community']
        self.assertTrue(all(r['mean']=='' and r['suppressed_small_n']=='True' for r in small))
        cohort=[r for r in stats if r['scope']=='cohort' and r['field']=='Autosomes_African_anc']
        self.assertEqual(len(cohort),42)
        for r in cohort:
            self.assertAlmostEqual(float(r['mean']),.2);self.assertAlmostEqual(float(r['median']),.2)
            self.assertAlmostEqual(float(r['q25']),.1);self.assertAlmostEqual(float(r['q75']),.3)
        categorical=summary.saved.table(output/'categorical_distributions.tsv')
        for field in summary.CATEGORICAL:
            subset=[r for r in categorical if r['config_id']==summary.saved.DEFAULT_NETWORKS[0]
                    and float(r['resolution'])==1. and r['scope']=='cohort' and r['field']==field]
            self.assertEqual(sum(int(r['n']) for r in subset),5)
        for path in output.iterdir():
            self.assertNotIn(b'PRIVATE_FIXTURE_SAMPLE',path.read_bytes())
            self.assertNotIn(b'PRIVATE_CLINICAL',path.read_bytes())
        self.assertEqual(before,{str(p.relative_to(self.source)):summary.saved.sha256(p)
                                for p in self.source.rglob('*') if p.is_file()})

    def test_minimum_n_can_be_explicitly_changed_without_changing_people(self):
        data=summary.saved.load_results(self.source); aligned,_=self.read()
        tables=summary.summarize(data,aligned,dict(self.settings,disclose_min_n=3))
        rows=[r for r in tables['ancestry_summary.tsv'] if r['scope']=='community']
        self.assertTrue(all(not r['suppressed_small_n'] and r['n_observed']==3 for r in rows))

    def test_existing_output_and_source_subdirectory_refused(self):
        output=self.root/'existing'; output.mkdir()
        for out in (output,self.source/'new_output'):
            with self.assertRaises(ValueError): summary.run(self.source,self.path,out,settings_json_text='{"expected_samples":5}')

    def test_tampered_graph_rejected_before_creating_output(self):
        first=self.source/summary.saved.DEFAULT_NETWORKS[0]/'graph_nodes.tsv'
        with first.open('a') as handle:handle.write('tampered\n')
        output=self.root/'summary'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            summary.run(self.source,self.path,output,settings_json_text='{"expected_samples":5}')
        self.assertFalse(output.exists())

    def test_metadata_drift_rejected_before_output(self):
        original=summary.summarize
        def change(data, metadata, settings):
            result=original(data,metadata,settings)
            with self.path.open('a') as handle:handle.write('\n')
            return result
        output=self.root/'summary'
        with patch.object(summary,'summarize',side_effect=change):
            with self.assertRaisesRegex(ValueError,'changed during summary'):
                summary.run(self.source,self.path,output,settings_json_text='{"expected_samples":5}')
        self.assertFalse(output.exists())

    def test_cli_errors_do_not_dump_raw_metadata_values(self):
        rows=copy.deepcopy(self.rows);rows[0]['Autosomes_African_anc']='PRIVATE_PERSON_BAD_NUMBER'
        write_metadata(self.path,rows)
        result=subprocess.run([sys.executable,str(SCRIPT),'--results-dir',str(self.source),
            '--metadata-file',str(self.path),'--output-dir',str(self.root/'failed'),
            '--settings-json-text','{"expected_samples":5}'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('PRIVATE_PERSON',result.stdout+result.stderr)
        self.assertNotIn('Traceback',result.stderr)


if __name__=='__main__':
    unittest.main()
