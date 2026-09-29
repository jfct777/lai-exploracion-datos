"""Synthetic-only authenticated companion figures; no human rows or genotypes."""
import csv
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from tests.test_m165_sweep_figures import fixture, write_tsv
from tests.test_m165_metadata_summary import metadata_rows, write_metadata

BASE=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,BASE/'bin'/f'{name}.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
presentation=module('m165_metadata_presentation')
try:
    import matplotlib
    import numpy
    import sklearn
    HAVE_STATS=True
except ImportError:
    HAVE_STATS=False


def synthetic_bundles(root):
    """Tiny source graphs plus honestly generated summaries; placeholder source PNGs."""
    metadata_helper=module('m165_metadata_summary');kinship_helper=module('m165_graph_kinship')
    source=fixture(root/'source');metadata_path=root/'metadata.tsv';write_metadata(metadata_path,metadata_rows())
    metadata_dir=root/'metadata_summary'
    mm=metadata_helper.run(source,metadata_path,metadata_dir,settings_json_text='{"expected_samples":5}')
    data=presentation.saved.load_results(source)
    kinship_dir=root/'kinship';kinship_dir.mkdir()
    gr,pr=kinship_helper.summarize(data,{(0,1):.05,(0,2):.001,(1,2):None},.0221)
    write_tsv(kinship_dir/'graph_kinship_summary.tsv',gr)
    write_tsv(kinship_dir/'partition_kinship_summary.tsv',pr)
    km=dict(status='COMPLETE_DESCRIPTIVE_EDGE_KINSHIP',no_reclustering=True,n_partitions=42,
        input_sha256=data['input_sha256'],source_core_sha256=data['core_sha256'],
        contains_individual_identifiers=False,
        outputs_sha256={p.name:presentation.saved.sha256(p) for p in kinship_dir.iterdir()})
    (kinship_dir/'manifest.json').write_text(json.dumps(km))
    figure_dir=root/'figures';figure_dir.mkdir();records=[]
    for row in data['rows']:
        stem=f"synthetic_{row['config_id']}_gamma{row['resolution']:g}"
        record=dict(config_id=row['config_id'],resolution=row['resolution'],length_bp=row['length_bp'],
                    threshold_bp=row['min_edge_bp'],n_nodes=5,n_active=3,n_isolated=2,n_assigned=3,
                    n_communities=1,n_edges=3,median_ari=.8,q25_ari=.7,q75_ari=.9)
        records.append(dict(record,stem=stem))
        (figure_dir/(stem+'.png')).write_bytes(b'SYNTHETIC_SOURCE_IMAGE_NOT_READ_AS_PIXELS')
    write_tsv(figure_dir/'synthetic_invariants.tsv',[{k:v for k,v in r.items() if k!='stem'} for r in records])
    fm=dict(status='COMPLETE_HISTORICAL_VISUAL_REPRODUCTION',no_reclustering=True,n_cells=42,n_png=42,
        figures=records,parameters=dict(prefix='synthetic'),source_core_sha256=data['core_sha256'],
        metadata=dict(sha256=mm['metadata_sha256']),input_sha256_before=data['input_sha256'],
        input_sha256_after=data['input_sha256'],
        outputs_sha256={p.name:presentation.saved.sha256(p) for p in figure_dir.iterdir()})
    (figure_dir/'manifest.json').write_text(json.dumps(fm))
    return figure_dir,metadata_dir,kinship_dir


def recertify(root,name):
    path=root/'manifest.json';meta=json.loads(path.read_text())
    meta['outputs_sha256'][name]=presentation.saved.sha256(root/name)
    path.write_text(json.dumps(meta))


class StructureTests(unittest.TestCase):
    def test_settings_enforce_readable_bounds(self):
        for text in ('{"dpi":30}','{"rows_per_page":100}','{"top_communities":99}','{"unknown":1}',
                     '{"original_figure_link_prefix":"https://outside"}'):
            with self.assertRaises(ValueError):presentation.settings_from_json(text)

    def test_every_community_and_category_survives_pagination(self):
        rows=[]
        for label in range(35):
            for cat,n in (('A',3),('B',2)):
                rows.append(dict(community_local=label,denominator=5,n=n,fineSTRUCTURE=cat))
        blocks=presentation.correspondence_blocks(rows)
        pages=presentation.paginate(blocks,presentation.settings_from_json())
        self.assertEqual(len(pages),3)
        self.assertEqual([b['community'] for p in pages for b in p],list(range(35)))
        self.assertTrue(all('3/5 (60.0 %)' in b['lines'][0] and '2/5 (40.0 %)' in b['lines'][0] for b in blocks))

    def test_zero_denominator_is_not_zero_percent(self):
        self.assertEqual(presentation.percent(0,0),'no definido')
        self.assertEqual(presentation.number('NA'),'no disponible')

    def test_path_traversal_and_link_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inside=root/'inside';inside.mkdir();outside=root/'outside.txt';outside.write_text('test')
            (inside/'escape.txt').symlink_to(outside)
            for name in ('../outside.txt','/etc/passwd','escape.txt'):
                with self.assertRaises(ValueError):presentation.safe_file(inside,name)

    def test_inverse_uses_entire_category_including_unassigned_and_missing(self):
        rows=[]
        for scope,denom,values in [('cohort',10,[('BrazilA',6),('BrazilB',3),('__MISSING__',1)]),
                                  ('unassigned',4,[('BrazilA',2),('BrazilB',1),('__MISSING__',1)]),
                                  ('community',6,[('BrazilA',4),('BrazilB',2),('__MISSING__',0)])]:
            rows.extend(dict(field='finestructure_clusters',scope=scope,community_local='0' if scope=='community' else '',
                             denominator_all=denom,category=cat,n=n,is_missing=str(cat=='__MISSING__')) for cat,n in values)
        result=presentation.inverse_reference_rows(rows,('fixture',1),10,4,'fixture.png')
        a=[r for r in result if r['fineSTRUCTURE']=='BrazilA']
        self.assertEqual([(r['community_local'],r['n'],r['denominator_finestructure_cohort']) for r in a],[(0,4,6),('sin_asignar',2,6)])
        md='\n'.join(presentation.inverse_markdown(result))
        self.assertIn('C0: 4/6 (66.7 %); sin asignar: 2/6 (33.3 %)',md)
        self.assertIn('Sin dato de fineSTRUCTURE | 1 | sin asignar: 1/1 (100.0 %)',md)
        rows[-2]['n']=1
        with self.assertRaisesRegex(ValueError,'inverse correspondence'):presentation.inverse_reference_rows(rows,('fixture',1),10,4,'fixture.png')


@unittest.skipUnless(HAVE_STATS,'requires existing image with NumPy, matplotlib and scikit-learn')
class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.inputs=synthetic_bundles(self.root)

    def test_authenticates_42_cells_and_exact_complete_correspondences(self):
        data=presentation.load_inputs(*self.inputs)
        self.assertEqual(len(data['records']),42)
        self.assertEqual(len(data['correspondences']),126)
        for key in presentation.EXPECTED_CELLS:
            rows=[r for r in data['correspondences'] if (r['config_id'],r['resolution'])==key]
            self.assertEqual(sum(r['n'] for r in rows),3)
            self.assertTrue(all(r['denominator']==3 for r in rows))
            inverse=[r for r in data['inverse_correspondences'] if (r['config_id'],r['resolution'])==key]
            self.assertEqual(sum(r['n'] for r in inverse),5)
            for category in {r['fineSTRUCTURE'] for r in inverse}:
                members=[r for r in inverse if r['fineSTRUCTURE']==category]
                self.assertEqual(sum(r['n'] for r in members),members[0]['denominator_finestructure_cohort'])
                self.assertEqual(sum(r['community_local']=='sin_asignar' for r in members),1)

    def test_notes_preserve_reference_vs_seed_ari_and_edge_denominators(self):
        data=presentation.load_inputs(*self.inputs);key=sorted(presentation.EXPECTED_CELLS)[0]
        text=presentation.interpretation(data,key,presentation.settings_from_json())
        self.assertIn('300 comparaciones dependientes',text)
        self.assertIn('NO todos los pares posibles',text)
        self.assertIn('1 sobre el corte, 1 por debajo y 1 sin valor',text)
        self.assertIn('mediana/IQR suprimidos por N<5',text)
        self.assertIn('Este asiático',text)
        self.assertIn('las cuatro medianas no tienen por qué sumar 100 %',text)
        self.assertIn('gris también atenúa no asignados',text)
        self.assertIn('fineSTRUCTURE → comunidades y personas sin asignar',text)
        self.assertIn('N es el TOTAL de esa categoría en la cohorte completa',text)
        self.assertNotIn('PRIVATE_FIXTURE_SAMPLE',text)
        stem=data['records'][key]['stem']
        self.assertIn(f']({stem}.png)',text)
        custom=presentation.interpretation(data,key,presentation.settings_from_json('{"original_figure_link_prefix":"../figures"}'))
        self.assertIn(f'](../figures/{stem}.png)',custom)

    def test_tampered_output_rejected(self):
        path=self.inputs[1]/'community_sizes.tsv'
        with path.open('a') as handle:handle.write('changed\n')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):presentation.load_inputs(*self.inputs)

    def test_recertified_wrong_category_denominator_is_rejected(self):
        root=self.inputs[1];name='categorical_distributions.tsv';rows=presentation.saved.table(root/name)
        row=next(r for r in rows if r['scope']=='community' and r['field']=='finestructure_clusters')
        row['denominator_all']='4';write_tsv(root/name,rows);recertify(root,name)
        with self.assertRaisesRegex(ValueError,'Category denominator'):presentation.load_inputs(*self.inputs)

    def test_mixed_upstream_source_rejected(self):
        path=self.inputs[2]/'manifest.json';meta=json.loads(path.read_text())
        meta['input_sha256']['a_different_input']='a'*64;path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'share unchanged'):presentation.load_inputs(*self.inputs)

    def test_missing_cell_rejected(self):
        path=self.inputs[0]/'manifest.json';meta=json.loads(path.read_text());meta['figures'].pop()
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'Missing or duplicated'):presentation.load_inputs(*self.inputs)

    def test_full_pipeline_42_notes_only_new_outputs_and_final_hashes(self):
        before={str(p):presentation.saved.sha256(p) for root in self.inputs for p in root.rglob('*') if p.is_file()}
        output=self.root/'presentation'
        def small_render(rows,output,stem,title,settings):
            (output/(stem+'.correspondencia.png')).write_bytes(b'SYNTHETIC_NEW_IMAGE')
            (output/(stem+'.correspondencia.pdf')).write_bytes(b'SYNTHETIC_NEW_PDF')
            return dict(pngs=[stem+'.correspondencia.png'],pdf=stem+'.correspondencia.pdf',n_pages=1,n_communities=1)
        with patch.object(presentation,'render_correspondence',side_effect=small_render):
            result=presentation.run(*self.inputs,output,settings_json_text='{"dpi":72}')
        self.assertEqual((result['n_interpretations'],result['n_correspondence_pdfs'],result['n_correspondence_pngs']),(42,42,42))
        self.assertEqual(len(list(output.glob('*.interpretacion.md'))),42)
        self.assertTrue((output/'finestructure_a_comunidades.tsv').is_file())
        for name,sha in result['outputs_sha256'].items():self.assertEqual(presentation.saved.sha256(output/name),sha)
        for path in output.iterdir():self.assertNotIn(b'PRIVATE_FIXTURE_SAMPLE',path.read_bytes())
        self.assertEqual(before,{str(p):presentation.saved.sha256(p) for root in self.inputs for p in root.rglob('*') if p.is_file()})

    def test_existing_output_refused(self):
        output=self.root/'exists';output.mkdir()
        with self.assertRaisesRegex(ValueError,'no overwrite'):presentation.run(*self.inputs,output)

    def test_real_synthetic_multipage_render_complete_and_legible_geometry(self):
        output=self.root/'preview';output.mkdir();rows=[]
        for label in range(17):
            for name,n in (('BrazilA',3),('BrazilB',2)):
                rows.append(dict(community_local=label,denominator=5,n=n,fineSTRUCTURE=name))
        settings=presentation.settings_from_json('{"dpi":72}')
        result=presentation.render_correspondence(rows,output,'synthetic','Fixture de prueba · sin datos humanos',settings)
        self.assertEqual(result['n_pages'],2)
        self.assertEqual(result['n_communities'],17)
        self.assertEqual(result['pngs'],['synthetic.correspondencia.png','synthetic.correspondencia.p002.png'])
        self.assertEqual(len(re.findall(rb'/Type\s*/Page\b',(output/result['pdf']).read_bytes())),2)
        from PIL import Image
        for name in result['pngs']:
            with Image.open(output/name) as img:self.assertEqual(img.size,(1008,720))


if __name__=='__main__':unittest.main()
