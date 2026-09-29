"""Synthetic historical-visual reproduction, no DNABR algorithms or data jobs."""
import csv
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

BIN=Path(__file__).resolve().parents[1]/"bin"
sys.path.insert(0,str(BIN))
sys.path.insert(0,str(Path(__file__).parent))
import m165_spectral_figures as renderer
from test_m165_sweep_figures import fixture, recertify, write_tsv

try:
    import numpy as np
    import scipy.sparse as sp
    import umap
    import ibd_community_enhanced as core
    HAVE_STACK=True
except ImportError:
    HAVE_STACK=False


class ContractTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.base=Path(temp.name)

    def test_json_inline_and_file_equivalent(self):
        path=self.base/'parameters.json';path.write_text('{}')
        self.assertEqual(renderer.settings_from_json(path),renderer.settings_from_json(text='{}'))

    def test_inline_legend_is_opt_in_and_wrapping_is_bounded(self):
        defaults=renderer.settings_from_json(text='{}')
        self.assertFalse(defaults['inline_metadata_legend'])
        self.assertEqual(defaults['legend_wrap_chars'],56)
        explicit=renderer.settings_from_json(text='{"inline_metadata_legend":true,"legend_wrap_chars":64}')
        self.assertTrue(explicit['inline_metadata_legend'])
        for bad in ({'inline_metadata_legend':1},{'legend_wrap_chars':31},
                    {'legend_wrap_chars':101},{'legend_wrap_chars':56.0}):
            with self.assertRaises(ValueError):
                renderer.settings_from_json(text=json.dumps(bad))

    def test_settings_reject_unknown_or_changed_scientific_controls(self):
        for entry in ({'gamma':1},{'seed':43},{'n_spectral':14},{'n_neighbors':14},
                      {'min_dist':0.1},{'resolution':.75},{'config_ids':[]},{'prefix':'../x'}):
            with self.assertRaises(ValueError):renderer.settings_from_json(text=json.dumps(entry))
        with self.assertRaises(ValueError):renderer.settings_from_json()
        with self.assertRaises(ValueError):renderer.settings_from_json('x','{}')

    def test_all_authenticated_graphs_and_resolutions_and_scalar_compatibility(self):
        settings=renderer.settings_from_json(text=json.dumps(dict(config_ids=list(renderer.KNOWN_CONFIGS),
                         resolutions=list(renderer.saved.GAMMAS),combined_pdf=False)))
        self.assertEqual(len(settings['config_ids'])*len(renderer.selected_resolutions(settings)),42)
        self.assertFalse(settings['combined_pdf'])
        scalar=renderer.settings_from_json(text='{"resolution": 1.2}')
        self.assertEqual(renderer.selected_resolutions(scalar),[1.2])
        for change in ({'resolutions':[]},{'resolutions':[1,1]},{'resolutions':[.75]},
                       {'resolutions':'1'},{'config_ids':['unknown']},
                       {'config_ids':[renderer.KNOWN_CONFIGS[0]]*2},{'combined_pdf':1}):
            with self.assertRaises(ValueError):renderer.settings_from_json(text=json.dumps(change))

    def metadata(self,rows):
        path=self.base/'metadata.txt';write_tsv(path,rows);return path

    def test_metadata_exact_duplicates_only_and_node_order(self):
        rows=[dict(ID='private_b',finestructure_clusters='B',extra='x'),
              dict(ID='private_a',finestructure_clusters='A',extra='y'),
              dict(ID='outside',finestructure_clusters='Z',extra='z')]
        path=self.metadata(rows+[rows[1].copy()])
        values,info=renderer.read_metadata(path,'ID','finestructure_clusters',['private_a','private_b'])
        self.assertEqual(values,['A','B']);self.assertEqual(info['n_metadata_rows'],4)
        self.assertEqual(info['n_identical_duplicates_collapsed'],1)
        self.assertEqual(info['n_extra_metadata_ids'],1)
        self.assertNotIn('private',json.dumps(info))

    def test_metadata_conflict_even_outside_colour_column_fails(self):
        path=self.metadata([dict(ID='a',color='A',extra='x'),dict(ID='a',color='A',extra='y')])
        with self.assertRaisesRegex(ValueError,'Conflicting duplicate'):
            renderer.read_metadata(path,'ID','color',['a'])

    def test_metadata_missing_people_and_bad_columns_fail(self):
        path=self.metadata([dict(ID='a',color='A')])
        with self.assertRaisesRegex(ValueError,'full cohort'):renderer.read_metadata(path,'ID','color',['b'])
        with self.assertRaises(ValueError):renderer.read_metadata(path,'ID','absent',['a'])
        with self.assertRaises(ValueError):renderer.read_metadata(path,None,'color',['a'])

    def test_no_overwrite_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError,'no overwrite'):
            renderer.run('absent',self.base,parameters_json_text='{}')

    def test_cli_parameters_sources_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            renderer.main(['--results-dir','x','--output-dir','y','--parameters-json','a',
                           '--parameters-json-text','{}'])


@unittest.skipUnless(HAVE_STACK,'requires existing pinned M16.5 image')
class NumericalTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.base=Path(temp.name);self.source=fixture(self.base/'source')
        self.matrix=sp.csr_matrix(np.array([[0,1,1,0,0],[1,0,1,0,0],[1,1,0,0,0],
                                          [0,0,0,0,0],[0,0,0,0,0]],dtype=float))
        for directory in self.source.iterdir():
            sp.save_npz(directory/'graph_sharing_matrix.npz',self.matrix)
            recertify(directory,'graph_sharing_matrix.npz')
        self.data=renderer.saved.load_results(self.source)
        self.graph=self.data['graphs'][0];self.directory=self.source/self.graph['config_id']
        self.settings=dict(renderer.DEFAULTS,expected_samples=5,dpi=72,prefix='synthetic')

    def tiny_render(self,core,matrix,coords,labels,graph,diag,settings,output,stem,metadata=None,metadata_name=None):
        """Cheap synthetic inventory fixture; default-path test exercises the real renderer."""
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(1,1));ax.scatter(coords[:,0],coords[:,1]);ax.axis('off')
        fig.savefig(output/f'{stem}.png',dpi=72);fig.savefig(output/f'{stem}.pdf');plt.close(fig)
        (output/f'{stem}.descripcion.md').write_text('Synthetic inventory fixture\n')

    def cached_coordinates(self):
        output=self.base/'saved_coordinates'
        with patch.object(renderer,'render_projection',side_effect=self.tiny_render):
            result=renderer.run(self.source,output,parameters_json_text=json.dumps(self.settings))
        # Reproduce the original scalar stage10 schema, before coordinates_file was explicit.
        for diagnostic in result['diagnostics']:diagnostic.pop('coordinates_file',None)
        (output/'manifest.json').write_text(json.dumps(result))
        return output,renderer.saved.sha256(output/'manifest.json')

    def test_original_matrix_matches_edges_and_weighted_degrees(self):
        matrix,samples,digest=renderer.authenticated_matrix(self.directory,self.graph)
        self.assertEqual((matrix-self.matrix).nnz,0);self.assertEqual(len(samples),5)
        self.assertEqual(len(digest),64)

    def test_changed_matrix_rejected_by_hash_then_edge_parity(self):
        sp.save_npz(self.directory/'graph_sharing_matrix.npz',self.matrix*2)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            renderer.authenticated_matrix(self.directory,self.graph)
        recertify(self.directory,'graph_sharing_matrix.npz')
        with self.assertRaisesRegex(ValueError,'saved edges'):
            renderer.authenticated_matrix(self.directory,self.graph)

    def test_projection_real_umap_keeps_isolates_all_nodes_and_source_matrix(self):
        before=renderer.matrix_hash(self.matrix)
        first=renderer.project_matrix(self.matrix,self.settings,core=core)
        second=renderer.project_matrix(self.matrix,self.settings,core=core)
        for result in (first,second):
            self.assertEqual(result[0].shape,(5,2));self.assertEqual(result[1].shape,(5,3))
            self.assertTrue(np.isfinite(result[0]).all())
            self.assertEqual(result[3]['umap_parameters']['random_state'],42)
            self.assertEqual(result[3]['umap_parameters']['min_dist'],.3)
            self.assertEqual(result[3]['umap_parameters']['n_neighbors'],4)
        self.assertEqual(first[3]['n_connected_components_active'],1)
        self.assertEqual(first[3]['n_stationary_candidates'],1)
        self.assertEqual(first[3]['n_modes_lambda_near_one'],1)
        self.assertEqual(first[3]['n_isolated'],2)
        self.assertFalse(first[3]['isolated_positions_informative'])
        self.assertEqual(renderer.matrix_hash(self.matrix),before)

    def test_degenerate_fixture_records_two_stages_without_false_bitwise_contract(self):
        """Regression recipe: triangle + two isolates, retained spectrum [1,0,0].

        Both eigsh stages can choose different bases despite a fixed UMAP seed.
        A run need not exhibit the difference; asserting inequality would also
        be flaky. Preserve the diagnostic instead of promising exact coordinates.
        """
        from scipy.spatial.distance import pdist
        before=renderer.matrix_hash(self.matrix)
        bases=[core._spectral_embedding(self.matrix,15,True) for _ in range(2)]
        eigenvalues=[np.sum(b*(core.laplacian_normalize(self.matrix)@b),axis=0) for b in bases]
        for values in eigenvalues:np.testing.assert_allclose(values,[1,0,0],atol=1e-12)
        np.testing.assert_allclose(pdist(bases[0]),pdist(bases[1]),atol=1e-12)
        frozen=bases[0].copy();source_hash=renderer.numeric_hash(frozen,'<f8')
        models=[umap.UMAP(n_components=2,random_state=42,n_neighbors=4,min_dist=.3,
                          metric='euclidean',n_jobs=1) for _ in range(2)]
        coords=[model.fit_transform(frozen.copy()) for model in models]
        graphs=[model.graph_.tocsr() for model in models]
        self.assertEqual(renderer.matrix_hash(graphs[0]),renderer.matrix_hash(graphs[1]))
        self.assertEqual(renderer.numeric_hash(frozen,'<f8'),source_hash)
        self.assertEqual(renderer.matrix_hash(self.matrix),before)
        self.assertTrue(all(c.shape==(5,2) and np.isfinite(c).all() for c in coords))
        print(json.dumps(dict(synthetic_degenerate_reproduction_diagnostic=True,
            retained_eigenvalues=[values.tolist() for values in eigenvalues],
            spectral_hashes=[renderer.numeric_hash(b,'<f8') for b in bases],
            fixed_umap_input_sha256=source_hash,
            identical_knn_graph_sha256=renderer.matrix_hash(graphs[0]),
            output_coordinate_hashes=[renderer.numeric_hash(c,'<f8') for c in coords],
            bitwise_reproducibility_claimed=False)),flush=True)

    def test_connected_nondegenerate_control_repeats_in_pinned_runtime(self):
        n=24;weights=np.random.default_rng(817).uniform(.25,3,n-1)
        matrix=sp.diags([weights,weights],[-1,1],shape=(n,n),format='csr')
        first=renderer.project_matrix(matrix,self.settings,core=core)
        second=renderer.project_matrix(matrix,self.settings,core=core)
        np.testing.assert_array_equal(first[1],second[1])
        np.testing.assert_array_equal(first[0],second[0])

    def test_umap_failure_is_fatal_no_fallback(self):
        class Broken:
            def __init__(self,**kwargs):pass
            def fit_transform(self,embedding):raise ValueError('synthetic UMAP failure')
        with self.assertRaisesRegex(RuntimeError,'no spectral fallback'):
            renderer.project_matrix(self.matrix,self.settings,core=core,umap_factory=Broken)

    def test_projection_receives_no_community_or_metadata_labels(self):
        import inspect
        self.assertNotIn('labels',inspect.signature(renderer.project_matrix).parameters)
        calls=[]
        class Recording:
            def __init__(self,**kwargs):self.kwargs=kwargs
            def fit_transform(self,embedding):
                calls.append((embedding.copy(),self.kwargs));return embedding[:,:2]
            def get_params(self):return self.kwargs
        renderer.project_matrix(self.matrix,self.settings,core=core,umap_factory=Recording)
        self.assertEqual(calls[0][1]['n_jobs'],1)
        self.assertNotIn('y',calls[0][1])

    def test_42_figures_share_one_geometry_per_graph_and_reuse_three_exact_sources(self):
        # Make a different saved label at gamma3; counts stay valid and no reclustering occurs.
        for directory in self.source.iterdir():
            path=directory/'leiden_assignments.tsv';rows=renderer.saved.table(path)
            for row in rows:
                if row['community_res_3']!='-1':row['community_res_3']='7'
            write_tsv(path,rows);recertify(directory,path.name)
        cache,cache_sha=self.cached_coordinates()
        output=self.base/'all_42';settings=dict(self.settings,config_ids=list(renderer.KNOWN_CONFIGS),
                                               resolutions=list(renderer.saved.GAMMAS))
        seen={}
        def inspect_render(*args,**kwargs):
            coords,labels,graph,diag=args[2:6]
            key=(graph['config_id'],diag['resolution'])
            self.assertNotIn(key,seen)
            seen[key]=(renderer.numeric_hash(coords,'<f8'),tuple(labels.tolist()))
            return self.tiny_render(*args,**kwargs)
        with (patch.object(renderer,'render_projection',side_effect=inspect_render),
              patch.object(renderer,'project_matrix',wraps=renderer.project_matrix) as projection):
            result=renderer.run(self.source,output,parameters_json_text=json.dumps(settings),
                coordinates_source_dir=cache,coordinates_source_manifest_sha256=cache_sha)
        self.assertEqual(projection.call_count,3)
        self.assertEqual(result['n_coordinate_sets_reused'],3);self.assertEqual(result['n_projections_computed'],3)
        self.assertEqual((result['n_png'],result['n_pdf'],result['n_private_npz']),(42,49,6))
        self.assertEqual(len(result['figures']),42)
        self.assertEqual(len(result['input_sha256_before']),54)
        self.assertEqual(result['input_sha256_before'],result['input_sha256_after'])
        self.assertEqual(len(result['coordinates_source']['input_sha256_before']),4)
        self.assertEqual(result['coordinates_source']['input_sha256_before'],result['coordinates_source']['input_sha256_after'])
        source_manifest=json.loads((cache/'manifest.json').read_text())
        for config in settings['config_ids']:
            self.assertEqual(len({seen[(config,g)][0] for g in renderer.saved.GAMMAS}),1)
            self.assertEqual(seen[(config,3.)][1][:3],(7,7,7))
            self.assertEqual(seen[(config,1.)][1][:3],(0,0,0))
            if config in renderer.saved.DEFAULT_NETWORKS:
                source_diag=next(d for d in source_manifest['diagnostics'] if d['config_id']==config)
                self.assertEqual(seen[(config,1.)][0],source_diag['coordinates_sha256'])
        for filename,pages in result['pdf_pages'].items():
            self.assertEqual(len(re.findall(rb'/Type\s*/Page\b',(output/filename).read_bytes())),pages)
        rows=renderer.saved.table(output/'synthetic_invariants.tsv')
        self.assertEqual(len(rows),42)
        self.assertTrue(all(float(row['median_ari'])==.8 and row['ari_seed_pairs']=='300' for row in rows))
        for diagnostic in result['diagnostics']:
            with np.load(output/diagnostic['coordinates_file'],allow_pickle=False) as archive:
                self.assertEqual(renderer.numeric_hash(archive['coordinates'],'<f8'),diagnostic['coordinates_sha256'])

    def test_coordinate_source_manifest_hash_and_graph_labels_fail_closed(self):
        cache,cache_sha=self.cached_coordinates()
        spec_hash=renderer.hashlib.sha256(inspect.getsource(core._spectral_embedding).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError,'manifest hash'):
            renderer.load_coordinate_source(cache,'0'*64,self.settings,spec_hash)
        with self.assertRaisesRegex(ValueError,'explicit manifest'):
            renderer.load_coordinate_source(cache,None,self.settings,spec_hash)
        manifest=json.loads((cache/'manifest.json').read_text())
        manifest['diagnostics'][0]['config_id']='unknown'
        (cache/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'mislabeled'):
            renderer.load_coordinate_source(cache,renderer.saved.sha256(cache/'manifest.json'),self.settings,spec_hash)

    def test_legend_only_revision_requires_metadata_and_never_recomputes(self):
        cache,cache_sha=self.cached_coordinates()
        settings=dict(self.settings,inline_metadata_legend=True)
        metadata=self.base/'metadata.tsv'
        write_tsv(metadata,[dict(ID=f'PRIVATE_FIXTURE_SAMPLE_{i}',finestructure_clusters='BrazilA')
                            for i in range(5)])
        with patch.object(renderer,'project_matrix',side_effect=AssertionError('Must not project')):
            with self.assertRaisesRegex(ValueError,'requires metadata'):
                renderer.run(self.source,self.base/'no_metadata',parameters_json_text=json.dumps(settings),
                    coordinates_source_dir=cache,coordinates_source_manifest_sha256=cache_sha)
            with self.assertRaisesRegex(ValueError,'requires metadata'):
                renderer.run(self.source,self.base/'no_coords',parameters_json_text=json.dumps(settings),
                    metadata_file=metadata,metadata_sample_column='ID')
            missing=dict(settings,config_ids=[renderer.KNOWN_CONFIGS[-1]])
            with self.assertRaisesRegex(ValueError,'cannot compute a new projection'):
                renderer.run(self.source,self.base/'missing_graph',parameters_json_text=json.dumps(missing),
                    metadata_file=metadata,metadata_sample_column='ID',coordinates_source_dir=cache,
                    coordinates_source_manifest_sha256=cache_sha)
            output=self.base/'inline_success'
            with patch.object(renderer,'render_projection',side_effect=self.tiny_render):
                result=renderer.run(self.source,output,parameters_json_text=json.dumps(settings),
                    metadata_file=metadata,metadata_sample_column='ID',coordinates_source_dir=cache,
                    coordinates_source_manifest_sha256=cache_sha)
        self.assertEqual(result['n_projections_computed'],0)
        self.assertEqual(result['n_coordinate_sets_reused'],3)
        self.assertEqual(result['coordinates_source']['input_sha256_before'],
                         result['coordinates_source']['input_sha256_after'])

    def test_coordinate_source_corruption_or_wrong_matrix_never_recomputes(self):
        cache,cache_sha=self.cached_coordinates()
        spec_hash=renderer.hashlib.sha256(inspect.getsource(core._spectral_embedding).encode()).hexdigest()
        source=renderer.load_coordinate_source(cache,cache_sha,self.settings,spec_hash)
        graph=self.data['graphs'][0];config=graph['config_id']
        hashes=dict(self.data['input_sha256'])
        hashes[f'{config}/graph_sharing_matrix.npz']=renderer.saved.sha256(self.source/config/'graph_sharing_matrix.npz')
        with self.assertRaisesRegex(ValueError,'graph/node-order'):
            renderer.reuse_projection(source,graph,self.matrix*2,self.settings,hashes)
        diagnostic=source['diagnostics'][config]
        relative=f"private/{self.settings['prefix']}_{config}_gamma1.coordinates.npz"
        with (cache/relative).open('ab') as handle:handle.write(b'changed')
        with self.assertRaisesRegex(ValueError,'artifact hash'):
            renderer.reuse_projection(source,graph,self.matrix,self.settings,hashes)

    def test_two_resolutions_without_global_pdf(self):
        settings=dict(self.settings,config_ids=[renderer.KNOWN_CONFIGS[-1]],resolutions=[.5,3.],combined_pdf=False)
        output=self.base/'two'
        with patch.object(renderer,'render_projection',side_effect=self.tiny_render):
            result=renderer.run(self.source,output,parameters_json_text=json.dumps(settings))
        self.assertEqual(result['n_pdf'],3);self.assertEqual(result['n_png'],2)
        self.assertEqual(result['n_projections_computed'],1)
        self.assertFalse((output/'synthetic_comparacion.pdf').exists())

    def test_unchanged_legacy_functions_used_and_inventory_privacy(self):
        output=self.base/'output'
        metadata=self.base/'metadata.tsv'
        write_tsv(metadata,[dict(ID=f'PRIVATE_FIXTURE_SAMPLE_{i}',finestructure_clusters='LevelA' if i<3 else 'LevelB')
                            for i in range(5)])
        getsource=inspect.getsource
        with (patch.object(core,'_plot_network_static',wraps=core._plot_network_static) as plot,
              patch.object(core,'_spectral_embedding',wraps=core._spectral_embedding) as spectral,
              patch.object(renderer.inspect,'getsource',side_effect=lambda obj:
                           getsource(getattr(obj,'_mock_wraps',None) or obj))):
            result=renderer.run(self.source,output,parameters_json_text=json.dumps(self.settings),
                                metadata_file=metadata,metadata_sample_column='ID')
        self.assertEqual(plot.call_count,3);self.assertEqual(spectral.call_count,3)
        for call in plot.call_args_list:
            self.assertEqual(len(call.kwargs['membership']),5)
            self.assertIn('sin posición informativa',call.kwargs['figure_note'])
            self.assertEqual(call.kwargs['metadata_name'],'finestructure_clusters')
        self.assertEqual(result['n_png'],3);self.assertEqual(result['n_pdf'],4)
        self.assertEqual(result['input_sha256_before'],result['input_sha256_after'])
        self.assertEqual(len(result['input_sha256_before']),51)
        self.assertFalse(result['projection_uses_labels']);self.assertTrue(result['no_reclustering'])
        self.assertFalse(result['bitwise_reproducibility_claimed'])
        self.assertTrue(result['coordinate_artifacts_persisted'])
        table=renderer.saved.table(output/'synthetic_invariants.tsv')
        self.assertEqual(len(table),3)
        self.assertTrue(all(int(row['n_nodes'])==5 and int(row['n_edges'])==3 for row in table))
        for path in (output/'private').glob('*.npz'):
            with np.load(path,allow_pickle=False) as archive:
                self.assertEqual(archive['coordinates'].shape,(5,2))
                np.testing.assert_array_equal(archive['node_index'],np.arange(5))
        self.assertEqual(len(re.findall(rb'/Type\s*/Page\b',(output/'synthetic_comparacion.pdf').read_bytes())),3)
        for name,digest in result['outputs_sha256'].items():
            self.assertEqual(renderer.saved.sha256(output/name),digest)
        for path in output.rglob('*'):
            if path.is_file():self.assertNotIn(b'PRIVATE_FIXTURE_SAMPLE',path.read_bytes())


if __name__=='__main__':
    unittest.main()
