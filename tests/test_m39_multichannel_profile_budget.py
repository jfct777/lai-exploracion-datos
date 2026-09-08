"""Pure-stdlib tests: timing-only budgeting and authenticated synthetic receipts."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bin'))
import m39_multichannel_profile_budget as B


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))
    return B.sha256(path)


class ProfileBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan_path = self.root/'plan'/'plan.json'
        self.outputs = self.root/'primary'
        self.launch_path = self.root/'gpu.launch.json'
        self.completion_path = self.root/'gpu.completion.json'
        self.seal_path = self.root/'source-seal.json'
        self.policy = B.BudgetPolicy(25,46800,.853624312,.0087671232,1.25,240,1,1,600)
        self.build_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def build_fixture(self):
        store = self.root/'store'; store.mkdir()
        header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }\n"
        (store/'radius_cm.npy').write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))
                                           +header+struct.pack('<d',.2))
        manifest = {'arrays':{'radius_cm':{'dtype':'<f8','file':'radius_cm.npy','shape':[1],
                    'nbytes':8,'sha256':B.sha256(store/'radius_cm.npy')}}}
        manifest_hash = write_json(store/'manifest.json',manifest)
        inputs = {'train_manifest_sha256':manifest_hash,'select_manifest_sha256':manifest_hash,
                  'development_sha256':'c'*64}
        groups = []
        for family in B.FAMILIES:
            group_id = f'profile-{family}'
            specs = []
            for arm in B.ARMS:
                cfg = {'schema_version':'m39-ordered-multichannel-training-v1',
                    'scope':'exploratory_chr22_R0_development_anchors_only','device':'cuda:0',
                    'arm':arm,'case_id':f'{group_id}-{arm}','paired_budget_id':group_id,
                    'max_runtime_seconds':600,'steps':64,'evaluate_every_steps':64,'evaluate_initial':True,
                    'anchor_count':64,'batch_size':2,'train_probe_people':4,'cpu_threads':2,
                    'model':{'family':family,'width':32},
                    'multichannel':{'hidden_width':32,'initial_gate':.01},**inputs}
                path = self.plan_path.parent/f'{group_id}-{arm}.json'
                specs.append({'file':path.name,'sha256':write_json(path,cfg)})
            groups.append({'id':group_id,'configs':specs})
        plan = {'schema_version':'m39-ordered-gpu-training-plan-v1',
            'scope':'exploratory_chr22_R0_development_anchors_only','stage':'multichannel_technical',
            'resources':{'max_workers':2,'task_seconds':1500,'controller_seconds':2100},
            'inputs':inputs,'groups':groups}
        plan_hash = write_json(self.plan_path,plan)
        seal = {'profile_sha256':plan_hash,'source_commit':'a'*40,'source_sha256':{'tool.py':'b'*64}}
        seal_hash = write_json(self.seal_path,seal)
        launch = {'schema_version':'m39-gpu-launch-v1','stage':plan['stage'],'plan_sha256':plan_hash,
            'source_seal_sha256':seal_hash,'input_sha256':inputs,'source_commit':seal['source_commit'],
            'group_ids':[group['id'] for group in groups],'run_id':'synthetic-profile',
            'created_utc':'2026-09-08T00:00:00+00:00','resources':{'max_concurrent_workers':2},
            'inputs':{'train_store':str(store),'select_store':str(store)}}
        launch_hash = write_json(self.launch_path,launch)
        write_json(self.completion_path,{'schema_version':'m39-gpu-controller-completion-v1',
            'status':'NEXTFLOW_COMPLETED_NEEDS_PRIMARY_POST','exit_code':0,'run_id':launch['run_id'],
            'launch_sha256':launch_hash,'elapsed_seconds':300})
        for group in groups:
            hashes = {}
            for spec in group['configs']:
                cfg = B.read_json(self.plan_path.parent/spec['file'])
                # Deliberately omit every efficacy field: budgeting must succeed.
                receipt = {'schema_version':cfg['schema_version'],
                    'decision':'COMPLETED_EXPLORATORY_DEVELOPMENT_CASE','config':cfg,
                    'sources':{'config_sha256':spec['sha256'],'code_sha256':{'tool.py':'b'*64},**inputs},
                    'scope':{'SCORE_opened':False},'curve':[{'step':0},{'step':64}],
                    'training_seconds':.64,'evaluation_seconds':1.28,'SELECT_seconds':1.,
                    'TRAIN_probe_seconds':.28,'elapsed_seconds':3.,'available_anchors':660,
                    'TRAIN_exposure':{'available_pairs':48*64},'rss_peak_bytes':1024,
                    'gpu_memory':{'allocated_bytes':1024}}
                path = self.outputs/f"training-{group['id']}"/cfg['arm']/'training.receipt.json'
                hashes[cfg['arm']] = write_json(path,receipt)
            write_json(self.outputs/f"training-{group['id']}"/'group.completion.json',{
                'status':'COMPLETED_DECLARED_PAIRED_ARMS_NEEDS_SCIENTIFIC_POST','exit_code':0,
                'stage':plan['stage'],'group_id':group['id'],'plan_sha256':plan_hash,
                'source_seal_sha256':seal_hash,'source_commit':seal['source_commit'],'SCORE_opened':False,
                'completed_arms':list(B.ARMS),'case_receipt_sha256':hashes,'elapsed_seconds':25.})

    def profile(self):
        return B.load_timing_profile(self.plan_path,self.outputs,self.launch_path,
                                     self.completion_path,self.seal_path)

    def test_full_timing_inventory_requires_no_efficacy_fields(self):
        profile = self.profile()
        self.assertEqual(len(profile['cases']),10)
        self.assertEqual(profile['run_id'],'synthetic-profile')
        self.assertEqual(profile['geometry']['train_people'],48)
        self.assertEqual(profile['geometry']['radius_cm'],.2)
        self.assertAlmostEqual(profile['cases'][0]['setup_seconds'],1.08)
        self.assertEqual(profile['groups'][0]['group_overhead_seconds'],10)
        self.assertNotIn('selected_SELECT_metrics',json.dumps(profile))

    def test_exact_32_fit_formula_and_profile_disk_reserve(self):
        result = B.forecast(self.profile(),self.policy,[128])
        candidate = result['candidates'][0]
        self.assertEqual(candidate['neural_fits'],32)
        self.assertEqual([stage['fits'] for stage in candidate['stages']],[12,20])
        self.assertEqual([stage['groups'] for stage in candidate['stages']],[6,4])
        fit = candidate['stages'][0]['fit_estimates'][0]
        self.assertEqual(fit['steps'],6144)
        self.assertEqual(fit['evaluation_count'],3)
        self.assertAlmostEqual(fit['estimate']['max']['raw_fit_seconds'],66.36)
        self.assertAlmostEqual(fit['estimate']['max']['guarded_fit_seconds'],82.95)
        self.assertAlmostEqual(candidate['science_worker_hours'],6779.4/3600)
        self.assertEqual(candidate['profile_charged_usd'],1)
        self.assertGreater(candidate['science_disks_usd'],0)
        self.assertAlmostEqual(candidate['total_estimated_usd'],2+6779.4/3600*(.853624312+.0087671232))
        self.assertFalse(result['efficacy_fields_used'])
        self.assertFalse(result['training_launched'])

    def test_largest_fitting_candidate_and_time_can_be_binding(self):
        profile = self.profile()
        all_fit = B.forecast(profile,self.policy)
        self.assertEqual(all_fit['selected_anchor_count'],660)
        by_count = {row['anchor_count_A']:row for row in all_fit['candidates']}
        cost = (by_count[660]['total_estimated_usd']+by_count[512]['total_estimated_usd'])/2
        self.assertEqual(B.forecast(profile,replace(self.policy,budget_usd=cost))['selected_anchor_count'],512)
        wall = (by_count[256]['profile_plus_science_wall_seconds']+
                by_count[128]['profile_plus_science_wall_seconds'])/2
        self.assertEqual(B.forecast(profile,replace(self.policy,wall_seconds=wall))['selected_anchor_count'],128)
        result = B.forecast(profile,replace(self.policy,budget_usd=1))
        self.assertIsNone(result['selected_anchor_count'])
        self.assertEqual(result['status'],'NO_CANDIDATE_FITS')

    def test_invalid_policy_inventory_and_missing_arm_fail_closed(self):
        profile = self.profile()
        for counts in ([661],[128,128],[0],[True]):
            with self.assertRaisesRegex(ValueError,'candidate anchors'):
                B.forecast(profile,self.policy,counts)
        for changed in ({'safety_factor':.9},{'worker_hourly_usd':float('nan')},{'disk_hourly_usd':-1}):
            with self.assertRaises(ValueError):
                B.forecast(profile,replace(self.policy,**changed))
        profile['cases'] = profile['cases'][:-1]
        with self.assertRaisesRegex(ValueError,'missing profile timing'):
            B.forecast(profile,self.policy)

    def test_controller_and_case_hash_corruption_are_rejected(self):
        completion = B.read_json(self.completion_path)
        completion['run_id'] = 'another-run'
        write_json(self.completion_path,completion)
        with self.assertRaisesRegex(ValueError,'controller not complete/bound'):
            self.profile()
        completion['run_id'] = 'synthetic-profile'
        write_json(self.completion_path,completion)
        path = self.outputs/'training-profile-cnn'/'none'/'training.receipt.json'
        receipt = B.read_json(path); receipt['training_seconds'] = -1
        write_json(path,receipt)
        with self.assertRaisesRegex(ValueError,'receipt hash differs'):
            self.profile()
        group_path = self.outputs/'training-profile-cnn'/'group.completion.json'
        group = B.read_json(group_path)
        group['case_receipt_sha256']['none'] = B.sha256(path)
        write_json(group_path,group)
        with self.assertRaisesRegex(ValueError,'invalid profile training time'):
            self.profile()

    def test_observed_profile_elapsed_is_charged_when_above_reserve(self):
        profile = self.profile()
        profile['controller_elapsed_seconds'] = 7200
        result = B.forecast(profile,self.policy,[128])
        expected = 4*(self.policy.worker_hourly_usd+self.policy.disk_hourly_usd)
        self.assertAlmostEqual(result['candidates'][0]['profile_charged_usd'],expected)
        self.assertGreater(result['candidates'][0]['profile_plus_science_wall_seconds'],7200)

    def test_impossible_group_and_controller_timings_are_rejected(self):
        path = self.outputs/'training-profile-cnn'/'group.completion.json'
        group = B.read_json(path)
        group['elapsed_seconds'] = 14.
        write_json(path,group)
        with self.assertRaisesRegex(ValueError,'shorter than case timers'):
            self.profile()
        group['elapsed_seconds'] = 301.
        write_json(path,group)
        with self.assertRaisesRegex(ValueError,'shorter than a worker group'):
            self.profile()

    def test_asynchronous_mixed_group_bound_is_not_the_optimistic_average(self):
        profile = self.profile()
        for case in profile['cases']:
            if case['family'] == 'attention':
                case['training_seconds_per_step'] *= 2
        result = B.forecast(profile,replace(self.policy,provision_seconds_per_worker=600),[128])
        for stage in result['candidates'][0]['stages']:
            times = stage['group_worker_seconds']
            bound = sum(times)/2+max(times)/2
            self.assertEqual(stage['wall_seconds_upper_estimate'],bound)
            self.assertGreater(bound,sum(times)/2)
            # Any order of two-family, two-worker work-conserving scheduling is
            # bounded, including adverse ordering, not just balanced family lanes.
            import itertools
            for order in set(itertools.permutations(times)):
                lanes = [0.,0.]
                for duration in order:
                    lane = min(range(2),key=lanes.__getitem__)
                    lanes[lane] += duration
                self.assertLessEqual(max(lanes),bound+1e-9)

    def test_native_task_eight_hour_ceiling_is_independent_of_dollar_and_wall_limits(self):
        profile = self.profile()
        for case in profile['cases']:
            case['training_seconds_per_step'] = 10.
        result = B.forecast(profile,replace(self.policy,budget_usd=10000,wall_seconds=10**8),[128])
        self.assertTrue(result['candidates'][0]['fits_cost_ceiling'])
        self.assertTrue(result['candidates'][0]['fits_time_ceiling'])
        self.assertFalse(result['candidates'][0]['fits_native_task_ceiling'])
        self.assertIsNone(result['selected_anchor_count'])

    def test_science_can_omit_initial_evaluation_without_silently_reducing_forecast(self):
        profile = self.profile()
        original = B.forecast(profile,self.policy)
        explicit = B.forecast(profile,self.policy,scientific_evaluate_initial=False)
        self.assertEqual(explicit['candidates'],original['candidates'])
        self.assertEqual(explicit['selected_anchor_count'],original['selected_anchor_count'])
        schedule = explicit['evaluation_schedule']
        self.assertTrue(schedule['profile_evaluate_initial'])
        self.assertFalse(schedule['declared_scientific_evaluate_initial'])
        self.assertEqual(schedule['conservative_extra_evaluations_per_fit'],1)
        self.assertEqual([stage['fit_estimates'][0]['evaluation_count']
                          for stage in explicit['candidates'][0]['stages']],[3,5])

    def test_import_and_forecast_module_are_stdlib_only(self):
        script = ('import sys; sys.path.insert(0,'+repr(str(ROOT/'bin'))+'); '
                  'import m39_multichannel_profile_budget; '
                  'assert not any(x in sys.modules for x in ("numpy","torch","scipy"))')
        result = subprocess.run([sys.executable,'-I','-c',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
