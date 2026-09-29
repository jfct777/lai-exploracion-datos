"""Synthetic-only figures, denominators, parity and fail-closed regressions."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/"bin/m165_sweep_figures.py"
spec=importlib.util.spec_from_file_location("m165_sweep_figures",SCRIPT)
figures=importlib.util.module_from_spec(spec);spec.loader.exec_module(figures)
try:
    import igraph
    import matplotlib
    import numpy as np
    HAVE_RENDERER=True
except ImportError:
    HAVE_RENDERER=False


def write_tsv(path,rows):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"wt",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def fixture(root):
    samples=[f"PRIVATE_FIXTURE_SAMPLE_{i}" for i in range(5)]
    for length,threshold in figures.GRID:
        source=f"L{length}_G50000_N20";config=f"{source}_T{threshold}_U0"
        d=root/config;d.mkdir(parents=True)
        write_tsv(d/"graph_nodes.tsv",[dict(node_id=i,sample_id=s,degree=2 if i<3 else 0,
                  weighted_degree=2 if i<3 else 0) for i,s in enumerate(samples)])
        write_tsv(d/"graph_edges.tsv.gz",[dict(sample_a=samples[a],sample_b=samples[b],weight=1.)
                                        for a,b in ((0,1),(0,2),(1,2))])
        write_tsv(d/"leiden_assignments.tsv",[dict(sample_id=s,**{f"community_res_{g:g}":0 if i<3 else -1
                  for g in figures.GAMMAS}) for i,s in enumerate(samples)])
        write_tsv(d/"resolution_summary.tsv",[dict(config_id=config,resolution=g,n_cohort=5,n_active=3,
                  n_isolated=2,n_assigned=3,n_active_unassigned=0,n_communities=1,largest_community=3)
                  for g in figures.GAMMAS])
        write_tsv(d/"leiden_ari_by_resolution.tsv",[dict(resolution=g,n_pairs=300,n_nodes=3,
                  median_ari=.8,q25_ari=.7,q75_ari=.9,min_ari=.6,max_ari=1.) for g in figures.GAMMAS])
        write_tsv(d/"leiden_modularity.tsv",[dict(resolution=g,seed=i+1,rb_quality=i,
                  is_representative=i==24) for g in figures.GAMMAS for i in range(25)])
        (d/"graph_summary.json").write_text(json.dumps(dict(n_nodes=5,n_edges=3,n_isolated=2,
                  min_edge_bp=threshold,min_max_segment_bp=0,weight_transform="log1p")))
        manifest=dict(status="COMPLETE_DESCRIPTIVE",chromosome="22",n_cohort=5,n_active=3,n_edges=3,
                  configuration=dict(config_id=config,source_config_id=source,length_bp=length,gap_bp=50000,
                                     min_shared=20,min_shared_effective=20,min_edge_bp=threshold,min_max_segment_bp=0),
                  parameters=dict(resolutions=list(figures.GAMMAS),min_community_size=3,consensus_resolution=1.,
                                  n_seeds=25,expected_samples=5),no_nmf=True,no_founder_classification=True,
                  source_sha256={"fixture":"1"*64},core_sha256="2"*64,
                  outputs_sha256={n:figures.sha256(d/n) for n in figures.REQUIRED})
        (d/"manifest.json").write_text(json.dumps(manifest))
    return root


def recertify(directory,name):
    path=directory/"manifest.json";data=json.loads(path.read_text())
    data["outputs_sha256"][name]=figures.sha256(directory/name);path.write_text(json.dumps(data))


class FigureContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.source=fixture(self.base/"source")
        self.first=self.source/"L250000_G50000_N20_T250000_U0"

    def test_all_42_rows_reproduce_counts_and_denominators(self):
        data=figures.load_results(self.source)
        self.assertEqual(len(data["graphs"]),6);self.assertEqual(len(data["rows"]),42)
        for row in data["rows"]:
            self.assertEqual(row["n_assigned"]+row["n_active_unassigned"]+row["n_isolated"],5)
            self.assertEqual(row["percent_assigned"],60.)
            self.assertNotIn("sample_id",row)
        self.assertEqual(len(data["input_sha256"]),48)

    def test_missing_graph_fails_before_output(self):
        (self.first/"manifest.json").rename(self.first/"missing.json")
        with self.assertRaises(FileNotFoundError):figures.load_results(self.source)

    def test_checksum_change_fails(self):
        with (self.first/"graph_nodes.tsv").open("a") as handle:handle.write("unexpected\n")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):figures.load_results(self.source)

    def test_missing_resolution_is_not_zero_filled(self):
        path=self.first/"resolution_summary.tsv";rows=figures.table(path)
        write_tsv(path,rows[:-1]);recertify(self.first,path.name)
        with self.assertRaisesRegex(ValueError,"Missing or duplicate resolution"):figures.load_results(self.source)

    def test_duplicate_resolution_fails(self):
        path=self.first/"leiden_ari_by_resolution.tsv";rows=figures.table(path)
        rows[-1]=rows[0].copy();write_tsv(path,rows);recertify(self.first,path.name)
        with self.assertRaisesRegex(ValueError,"Missing or duplicate resolution"):figures.load_results(self.source)

    def test_wrong_denominator_fails(self):
        path=self.first/"resolution_summary.tsv";rows=figures.table(path);rows[0]["n_cohort"]=6
        write_tsv(path,rows);recertify(self.first,path.name)
        with self.assertRaisesRegex(ValueError,"resolution counts"):figures.load_results(self.source)

    def test_isolated_assignment_fails(self):
        path=self.first/"leiden_assignments.tsv";rows=figures.table(path);rows[-1]["community_res_1"]=0
        write_tsv(path,rows);recertify(self.first,path.name)
        with self.assertRaisesRegex(ValueError,"isolated node"):figures.load_results(self.source)

    def test_unmatched_sample_order_fails(self):
        path=self.first/"leiden_assignments.tsv";rows=figures.table(path);rows.reverse()
        write_tsv(path,rows);recertify(self.first,path.name)
        with self.assertRaisesRegex(ValueError,"Assignment order"):figures.load_results(self.source)

    def test_ari_iqr_is_checked_and_never_called_confidence_interval(self):
        path=self.first/"leiden_ari_by_resolution.tsv";rows=figures.table(path);rows[0]["q25_ari"]=.95
        write_tsv(path,rows);recertify(self.first,path.name)
        with self.assertRaisesRegex(ValueError,"ARI quantiles"):figures.load_results(self.source)

    def test_representative_must_maximize_saved_objective(self):
        path=self.first/"leiden_modularity.tsv";rows=figures.table(path)
        rows[0]["is_representative"]="True";rows[24]["is_representative"]="False"
        write_tsv(path,rows);recertify(self.first,path.name)
        with self.assertRaisesRegex(ValueError,"RB-quality maximum"):figures.load_results(self.source)

    def test_existing_output_and_traversal_fail_closed(self):
        output=self.base/"existing";output.mkdir()
        with self.assertRaisesRegex(ValueError,"no overwrite"):figures.run(self.source,output)
        with self.assertRaisesRegex(ValueError,"Unsafe"):figures.run(self.source,self.base/"other",prefix="../outside")
        self.assertFalse((self.base/"other").exists())

    def test_default_network_selection_is_three_gamma_one_not_42(self):
        selected=figures.select_networks()
        self.assertEqual(selected,[(c,1.) for c in figures.DEFAULT_NETWORKS])
        self.assertEqual(len(figures.select_networks(all_networks=True)),42)
        self.assertEqual(figures.select_networks(figures.DEFAULT_NETWORKS[0],.8),
                         [(figures.DEFAULT_NETWORKS[0],.8)])

    def test_explicit_selection_rejects_missing_duplicate_and_unobserved(self):
        for selection in ("", "unknown", ",".join([figures.DEFAULT_NETWORKS[0]]*2)):
            with self.assertRaises(ValueError):figures.select_networks(selection)
        with self.assertRaises(ValueError):figures.select_networks(resolution=.75)
        with self.assertRaises(ValueError):figures.select_networks(figures.DEFAULT_NETWORKS[0],all_networks=True)

    @unittest.skipUnless(HAVE_RENDERER,"requires the existing M16.5 image")
    def test_common_layout_is_deterministic(self):
        data=figures.load_results(self.source)
        first,shown,limits=figures.shared_layout(data,42,10)
        second,shown2,limits2=figures.shared_layout(data,42,10)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(shown,shown2);self.assertEqual(limits,limits2)
        self.assertTrue(np.isnan(first[[3,4]]).all())

    @unittest.skipUnless(HAVE_RENDERER,"requires the existing M16.5 image")
    def test_network_size_panel_shows_all_32_groups_without_node_labels(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PathCollection
        data=figures.load_results(self.source)
        graph=data["graphs"][0]
        sizes={label:3+label%5 for label in range(32)}
        labels=np.array([label for label,count in sizes.items() for _ in range(count)])
        n=len(labels)
        graph.update(n_cohort=n,n_active=n,active=[True]*n,edges=[(0,1),(1,2)],
                     labels={g:labels.tolist() for g in figures.GAMMAS})
        data.update(graphs=[graph],n_cohort=n)
        positions=np.column_stack((np.linspace(-1,1,n),np.linspace(1,-1,n)))
        before=positions.copy();selection=[(graph["config_id"],1.)]
        output=self.base/"network_panels";output.mkdir()
        captured=[]
        def inspect(fig,output,stem,dpi,description,book=None):
            net,bars=fig.axes
            self.assertEqual(tuple(fig.get_size_inches()),(14.,9.))
            self.assertEqual(len(net.texts),0)
            self.assertEqual(len(bars.patches),32)
            ranked=sorted(sizes,key=lambda label:(-sizes[label],label))
            self.assertEqual([t.get_text() for t in bars.get_yticklabels()],
                             [f"C{label}" for label in ranked])
            self.assertEqual([p.get_width() for p in bars.patches],[sizes[label] for label in ranked])
            self.assertEqual([t.get_text() for t in bars.texts],[str(sizes[label]) for label in ranked])
            self.assertEqual(bars.get_xlabel(),"Personas")
            self.assertEqual(net.get_xlim(),(-1.2,1.2));self.assertEqual(net.get_ylim(),(-1.2,1.2))
            edge_collection=next(c for c in net.collections if isinstance(c,LineCollection))
            self.assertEqual(len(edge_collection.get_segments()),2)
            nodes=next(c for c in net.collections if isinstance(c,PathCollection))
            self.assertEqual(len(nodes.get_offsets()),n)
            for rank,label in enumerate(ranked):
                index=int(np.flatnonzero(labels==label)[0])
                np.testing.assert_allclose(bars.patches[rank].get_facecolor()[:3],nodes.get_facecolors()[index,:3])
            self.assertIn("TODOS los grupos",description)
            self.assertIn("no es una partición consensuada",description)
            self.assertIn("G = 50 kb; N mínimo = 20",description)
            self.assertTrue(any("25 semillas" in t.get_text() for t in fig.texts))
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            self.assertLess(fig.legends[0].get_window_extent(renderer).y1,
                            bars.xaxis.label.get_window_extent(renderer).y0)
            captured.append(stem);plt.close(fig)
        with patch.object(figures,"save_figure",side_effect=inspect):
            rows=figures.network_figures(data,output,"synthetic",72,positions,list(range(n)),
                                        (-1.2,1.2,-1.2,1.2),selection)
        self.assertEqual(len(captured),1);self.assertEqual(len(rows),32)
        self.assertTrue(all(not row["annotated"] and row["shown_in_size_panel"] for row in rows))
        np.testing.assert_array_equal(positions,before)

    @unittest.skipUnless(HAVE_RENDERER,"requires the existing M16.5 image")
    def test_complete_synthetic_render_inventory_pages_and_privacy(self):
        output=self.base/"figures"
        result=figures.run(self.source,output,prefix="synthetic",dpi=72,layout_iterations=10)
        self.assertEqual(result["n_png"],6);self.assertEqual(result["n_pdf"],7)
        self.assertEqual(result["n_network_figures"],3)
        self.assertTrue(result["no_reclustering"])
        self.assertEqual(result["n_union_isolated"],2)
        self.assertEqual(len(figures.table(output/"synthetic_tabla_42.tsv")),42)
        sizes=figures.table(output/"synthetic_tamaños_comunidades.tsv")
        self.assertEqual(len(sizes),3)
        self.assertTrue(all(row["annotated"]=="False" and row["shown_in_size_panel"]=="True" for row in sizes))
        self.assertEqual(len(list(output.glob("*.descripcion.md"))),6)
        comparison=output/"syntheticD_comparacion_redes.pdf"
        self.assertEqual(len(re.findall(rb"/Type\s*/Page\b",comparison.read_bytes())),3)
        for name,expected in result["outputs_sha256"].items():
            self.assertEqual(figures.sha256(output/name),expected)
        for path in output.iterdir():
            self.assertNotIn(b"PRIVATE_FIXTURE_SAMPLE",path.read_bytes())


if __name__=="__main__":
    unittest.main()
