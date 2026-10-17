#!/usr/bin/env python3
"""Shared private affinity table authentication and emission for K03.

This reader exposes only FIT edges. It authenticates an admitted upstream
product; it does not certify ascertainment, fit a graph or open query outcomes.
"""
from pathlib import Path
import csv
import math
from collections import Counter
import shutil
import numpy as np
import r02_structure_discrete as discrete

require, sha, load_json = discrete.require, discrete.sha, discrete.load_json
CODE_FILES = ('r02_structure_affinity_io.py', *discrete.CODE_FILES)
FIT_FILES = {'persons.tsv', 'fit_affinities.private.tsv', 'exclusion.json', 'affinity_inputs.json'}
NEW_SCHEMA = 'r02_structure_rare_affinity_inputs_v1'
STAGE = 'COMPLETE_AFFINITY_INPUTS_COMMUNITIES_AND_COMPARATOR_PENDING'
FAMILY_SCHEMA = 'r02_structure_family_affinity_inputs_v1'
COMPOSITE_SCHEMA = 'r02_structure_lazy_composite_affinity_v1'
FAMILY_EDGE_COLUMNS = ['left_index', 'right_index', 'observed', 'affinity',
                       'support_numerator', 'support_denominator']
FAMILY_CORE_FILES = {'persons.tsv', 'fit_affinities.private.tsv',
                     'query_affinities.private.tsv', 'affinity_inputs.json'}


def pair_indices(persons, roles, policy):
    require(policy in {'INCLUDE', 'EXCLUDE'}, 'Explicit FIT component policy required')
    fit = np.asarray(roles['FIT'])
    left, right = np.triu_indices(len(fit), 1); left, right = fit[left], fit[right]
    if policy == 'EXCLUDE':
        components = np.asarray([p['component'] for p in persons])
        keep = components[left] != components[right]; left, right = left[keep], right[keep]
    query = np.sort(np.concatenate([roles['SELECT'], roles['EVALUATE']]))
    return (left, right), (np.repeat(query, len(fit)), np.tile(fit, len(query)))


def write_pair_tables(output, keys, totals, nsites, resources):
    """Emit the existing pair format from additive I/U/Q, preserving U=0 NA."""
    output = Path(output); require(output.is_dir() and type(nsites) is int and nsites > 0,
                                 'Existing private output and positive candidate count required')
    require(len(keys) == len(totals) == 2, 'FIT and query pair arrays required')
    written = 0
    for filename, indices, counts in zip(('fit_affinities.private.tsv', 'query_affinities.private.tsv'), keys, totals):
        require(counts.shape == (3, len(indices[0])) and len(indices[0]) == len(indices[1]) and
                counts.dtype.kind in 'iu', 'Pair/count dimensions or integer dtype differ')
        with (output/filename).open('x') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n'); writer.writerow(discrete.EDGE_COLUMNS)
            for row, (i, j) in enumerate(zip(*indices)):
                intersection, union, joint = map(int, counts[:, row])
                require(0 <= intersection <= union <= joint <= nsites, 'Aggregate denominator invariant failed')
                writer.writerow([int(i), int(j), int(union > 0), format(intersection/union, '.17g') if union else 'NA',
                                 joint, union, nsites])
                if row % 4096 == 0:
                    require(written+stream.tell() <= resources['max_output_bytes'] and
                            shutil.disk_usage(output).free >= resources['min_free_bytes'], 'Output/disk reserve exceeded')
        written += (output/filename).stat().st_size
    require(written <= resources['max_output_bytes'], 'Output byte bound exceeded')


class AuthenticatedFitAffinities:
    def __init__(self, inputs, *, evidence_scope, max_input_bytes, max_metadata_bytes):
        require(set(inputs) == FIT_FILES and evidence_scope in {'SYNTHETIC_INSTRUMENT_ONLY', 'DNABR_R02'},
                'Exact FIT affinity inputs and evidence scope required')
        require(type(max_input_bytes) is int and type(max_metadata_bytes) is int and
                max_input_bytes > 0 and max_metadata_bytes > 0, 'Explicit positive affinity byte bounds required')
        self.inputs = {}; self.pins = {}; total = metadata = 0
        for name, desc in inputs.items():
            require(set(desc) == {'path', 'sha256'} and discrete.valid_hash(desc['sha256']), 'Exact FIT artifact pin required')
            path = Path(desc['path'])
            require(path.is_absolute() and path.is_file(), 'Absolute existing FIT source required')
            total += path.stat().st_size
            if name != 'fit_affinities.private.tsv': metadata += path.stat().st_size
            require(total <= max_input_bytes and metadata <= max_metadata_bytes and sha(path) == desc['sha256'],
                    'FIT source changed or exceeds byte bound')
            self.inputs[name] = path; self.pins[name] = desc['sha256']
        m = self.upstream = load_json(self.inputs['affinity_inputs.json'])
        require(m['schema'] in {'r02_structure_discrete_reuse_v1', NEW_SCHEMA} and m['stage'] == STAGE and
                m['exclusion_scope'] in discrete.SCOPES and m['same_component_fit_edges'] == 'EXCLUDE' and
                m['missing_edge_policy'] == 'UNKNOWN_NOT_ZERO' and m['affinity']['source_measure'] == 'SOURCE_MINOR_JACCARD' and
                m['affinity']['transformation'] == 'IDENTITY_NONNEGATIVE_INPUT' and m['affinity']['maximum'] == 1. and
                type(m['candidate_sites']) is int and m['candidate_sites'] > 0 and
                discrete.valid_hash(m['source_contract_sha256']), 'Reviewed rare affinity semantics required')
        if m['schema'] == 'r02_structure_discrete_reuse_v1':
            require(m['exclusion_scope'] == 'WHOLE_CHROMOSOME', 'Old count producer supports whole chromosome only')
        for name in FIT_FILES-{'affinity_inputs.json'}:
            require(m['sha256'][name] == self.pins[name], 'Upstream FIT artifact binding differs')
        self.persons = discrete.shared.table(self.inputs['persons.tsv'], ['sample_id', 'component', 'role'])
        self.roles = discrete.shared.validate_roles(self.persons)
        require(discrete.materialize.digest(self.persons) == m['persons_content_sha256'], 'Upstream roster changed')
        if evidence_scope == 'DNABR_R02':
            require(len(self.persons) == 2619 and {k: len(v) for k, v in self.roles.items()} ==
                    {'FIT': 1460, 'SELECT': 476, 'EVALUATE': 683}, 'FIT learner must retain original R02 roster')
        actual = load_json(self.inputs['exclusion.json'])
        if m['schema'] == NEW_SCHEMA:
            expected = m['exclusion']
        else:
            recipes = [r for r in m['recipes'] if r['block_id'] == m['block_id']]
            require(len(recipes) == 1, 'Exactly one original whole-chromosome recipe required')
            expected = {k: v for k, v in recipes[0].items() if k != 'included_chromosomes'}
        require(actual == expected and actual['block_id'] == m['block_id'], 'Affinity exclusion geometry differs')
        self.source_sha256 = self.pins['affinity_inputs.json']
        self.fit_set = set(map(int, self.roles['FIT']))
        sizes = Counter(self.persons[i]['component'] for i in self.fit_set)
        self.expected_edges = len(self.fit_set)*(len(self.fit_set)-1)//2-sum(n*(n-1)//2 for n in sizes.values())

    def fit_edges(self):
        previous = None; seen = 0
        with self.inputs['fit_affinities.private.tsv'].open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            require(reader.fieldnames == discrete.EDGE_COLUMNS, 'FIT edge columns differ')
            for row in reader:
                require(None not in row and None not in row.values(), 'Malformed FIT affinity row')
                i, j = int(row['left_index']), int(row['right_index'])
                require(i in self.fit_set and j in self.fit_set and i < j and
                        (previous is None or previous < (i, j)) and
                        self.persons[i]['component'] != self.persons[j]['component'], 'FIT edge roster/order/component differs')
                previous = (i, j); seen += 1
                joint, union, sites = (int(row[k]) for k in discrete.EDGE_COLUMNS[4:])
                require(0 < union <= joint <= sites and sites == self.upstream['candidate_sites'] and row['observed'] == '1',
                        'Unknown FIT support cannot become zero or a cohort filter')
                value = float(row['affinity']); require(math.isfinite(value) and 0 <= value <= 1, 'FIT affinity range differs')
                yield i, j, value
        require(seen == self.expected_edges, 'Complete eligible FIT edge support required')
        self.verify()

    def verify(self):
        require(all(sha(self.inputs[n]) == h for n, h in self.pins.items()), 'FIT affinity input changed after admission')


class AuthenticatedRareView(AuthenticatedFitAffinities):
    """Explicit extension for query-to-FIT features, never query outcomes.

    The FIT-only reader remains unchanged for learners/profilers. Query rows are
    streamed in the fixed complete roster order; U=0 is None, observed J=0 is 0.
    Completion is authenticated only after the iterator is exhausted.
    """
    def __init__(self, inputs, *, evidence_scope, max_input_bytes, max_metadata_bytes):
        filename = 'query_affinities.private.tsv'
        require(set(inputs) == FIT_FILES | {filename}, 'Exact FIT and query affinity inputs required')
        super().__init__({k:v for k,v in inputs.items() if k != filename}, evidence_scope=evidence_scope,
                         max_input_bytes=max_input_bytes, max_metadata_bytes=max_metadata_bytes)
        desc = inputs[filename]
        require(set(desc) == {'path','sha256'} and discrete.valid_hash(desc['sha256']), 'Exact query artifact pin required')
        path = Path(desc['path'])
        require(path.is_absolute() and path.is_file(), 'Absolute existing query source required')
        require(sum(p.stat().st_size for p in self.inputs.values())+path.stat().st_size <= max_input_bytes,
                'FIT and query inputs exceed combined byte bound')
        require(sha(path) == desc['sha256'] == self.upstream['sha256'][filename], 'Query source binding changed')
        self.inputs[filename] = path; self.pins[filename] = desc['sha256']
        self.query_indices = sorted(int(i) for r in ('SELECT','EVALUATE') for i in self.roles[r])

    def query_rows(self, *, roles=('SELECT','EVALUATE')):
        require(isinstance(roles, tuple) and roles and len(set(roles)) == len(roles)
                and set(roles) <= {'SELECT','EVALUATE'}, 'Explicit held-out feature roles required')
        self.verify(); fit = sorted(self.fit_set)
        with self.inputs['query_affinities.private.tsv'].open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            require(reader.fieldnames == discrete.EDGE_COLUMNS, 'Query edge columns differ')
            for query in self.query_indices:
                observations = {}; joint_sum = union_sum = masked = 0
                for reference in fit:
                    row = next(reader, None)
                    require(row is not None and None not in row and None not in row.values(),
                            'Missing or malformed query affinity row')
                    require((int(row['left_index']),int(row['right_index'])) == (query,reference),
                            'Query-to-FIT roster/order differs; missing edges cannot be silently dropped')
                    joint, union, sites = (int(row[k]) for k in discrete.EDGE_COLUMNS[4:])
                    require(0 <= union <= joint <= sites and sites == self.upstream['candidate_sites'],
                            'Query denominator or candidate territory differs')
                    if row['observed'] == '1':
                        value = float(row['affinity'])
                        require(union > 0 and math.isfinite(value) and 0 <= value <= 1,
                                'Observed query affinity or support invalid')
                    else:
                        require(row['observed'] == '0' and row['affinity'] == 'NA' and union == 0,
                                'Unknown query affinity must preserve NA and zero union')
                        value = None; masked += 1
                    observations[reference] = value; joint_sum += joint; union_sum += union
                if self.persons[query]['role'] in roles:
                    yield query, observations, dict(observed_fit_count=len(fit)-masked,
                        total_fit_count=len(fit), joint_callable_sites_sum=joint_sum,
                        value_denominator_sites_sum=union_sum,
                        candidate_sites=self.upstream['candidate_sites'],
                        denominator_scope='SUM_OVER_QUERY_TO_FIT_EDGES_NOT_UNIQUE_LOCI',
                        target_outcomes_read=False)
            require(next(reader, None) is None, 'Unexpected extra query affinity row')
        self.verify()


class AuthenticatedAffinityView:
    """Authenticate family-explicit FIT and query-to-FIT affinities.

    This adapter deliberately does not infer Jaccard, rare-variant semantics, a
    unit, or an upper bound from filenames.  The manifest states those meanings;
    family-specific producers retain sufficient statistics and source pins.
    Missing query affinities remain ``None``.  FIT support must be complete for
    the current dense spectral consumer and this operational restriction is
    recorded as capacity, not as a biological cohort filter.
    """
    def __init__(self, inputs, *, evidence_scope, max_input_bytes, max_metadata_bytes,
                 expected_family=None):
        require(FAMILY_CORE_FILES <= set(inputs)
                and evidence_scope in {'SYNTHETIC_INSTRUMENT_ONLY', 'DNABR_R02'},
                'Family affinity core inputs and explicit evidence scope required')
        require(type(max_input_bytes) is int and type(max_metadata_bytes) is int
                and max_input_bytes > 0 and max_metadata_bytes > 0,
                'Explicit positive family-affinity byte bounds required')
        self.inputs = {}; self.pins = {}; total = metadata = 0
        for name, desc in inputs.items():
            require(isinstance(name, str) and name and '/' not in name and name not in {'.', '..'}
                    and set(desc) == {'path', 'sha256'} and discrete.valid_hash(desc['sha256']),
                    'Safe exact family artifact pin required')
            path = Path(desc['path'])
            require(path.is_absolute() and path.is_file(), 'Absolute existing family source required')
            total += path.stat().st_size
            if not name.endswith('.private.tsv') and not name.endswith('.private.npz'):
                metadata += path.stat().st_size
            require(total <= max_input_bytes and metadata <= max_metadata_bytes
                    and sha(path) == desc['sha256'],
                    'Family source changed or exceeds byte bound')
            self.inputs[name] = path; self.pins[name] = desc['sha256']
        m = self.upstream = load_json(self.inputs['affinity_inputs.json'])
        required = {'schema', 'stage', 'family', 'affinity', 'missing_edge_policy',
                    'same_component_fit_edges', 'territory', 'role_counts',
                    'persons_content_sha256', 'normalization', 'support', 'sources',
                    'capacity', 'sha256', 'target_outcomes_read', 'evaluate_scored',
                    'scientific_admission', 'biological_validation'}
        require(set(m) == required and m['schema'] == FAMILY_SCHEMA
                and m['stage'] == 'FAMILY_AFFINITIES_READY_NOT_SCIENTIFICALLY_ADMITTED'
                and isinstance(m['family'], str) and m['family']
                and (expected_family is None or m['family'] == expected_family)
                and m['missing_edge_policy'] == 'UNKNOWN_NOT_ZERO'
                and m['same_component_fit_edges'] == 'EXCLUDE'
                and m['target_outcomes_read'] is False and m['evaluate_scored'] is False
                and m['scientific_admission'] is False and m['biological_validation'] is False,
                'Explicit non-admitted family affinity semantics required')
        affinity = m['affinity']
        require(set(affinity) == {'measure', 'units', 'transformation', 'minimum', 'maximum'}
                and all(isinstance(affinity[k], str) and affinity[k]
                        for k in ('measure', 'units', 'transformation'))
                and type(affinity['minimum']) in (int, float) and math.isfinite(affinity['minimum'])
                and affinity['minimum'] >= 0
                and (affinity['maximum'] is None
                     or (type(affinity['maximum']) in (int, float)
                         and math.isfinite(affinity['maximum'])
                         and affinity['maximum'] >= affinity['minimum'])),
                'Family affinity measure/unit/transformation/range required')
        require(set(m['sha256']) == set(inputs)-{'affinity_inputs.json'}
                and all(m['sha256'][name] == self.pins[name] for name in m['sha256']),
                'Family manifest artifact inventory differs')
        self.persons = discrete.shared.table(self.inputs['persons.tsv'],
                                             ['sample_id', 'component', 'role'])
        self.roles = discrete.shared.validate_roles(self.persons)
        require(discrete.materialize.digest(self.persons) == m['persons_content_sha256']
                and m['role_counts'] == {k: len(v) for k, v in self.roles.items()},
                'Family roster content or role counts differ')
        if evidence_scope == 'DNABR_R02':
            require(len(self.persons) == 2619 and m['role_counts'] ==
                    {'FIT': 1460, 'SELECT': 476, 'EVALUATE': 683},
                    'Family adapter must retain the frozen R02 roster')
        self.fit_set = set(map(int, self.roles['FIT']))
        sizes = Counter(self.persons[i]['component'] for i in self.fit_set)
        self.expected_edges = (len(self.fit_set)*(len(self.fit_set)-1)//2
                               - sum(n*(n-1)//2 for n in sizes.values()))
        self.query_indices = sorted(int(i) for role in ('SELECT', 'EVALUATE')
                                    for i in self.roles[role])
        capacity = m['capacity']
        require(set(capacity) == {'fit_edges', 'query_fit_edges', 'fit_support',
                                  'query_support', 'dense_spectrum_memory_order'}
                and capacity['fit_edges'] == self.expected_edges
                and capacity['query_fit_edges'] == len(self.query_indices)*len(self.fit_set)
                and capacity['fit_support'] == 'COMPLETE_OBSERVED_REQUIRED'
                and capacity['query_support'] == 'EXPLICIT_OBSERVED_OR_UNKNOWN'
                and capacity['dense_spectrum_memory_order'] == 'O_N_FIT_SQUARED',
                'Family affinity capacity/dimensions differ')
        self.family = m['family']; self.affinity = affinity
        self.source_sha256 = self.pins['affinity_inputs.json']

    def _value(self, row):
        if row['observed'] == '0':
            require(row['affinity'] == row['support_numerator'] == row['support_denominator'] == 'NA',
                    'Unknown affinity and support must stay NA, not zero')
            return None
        require(row['observed'] == '1', 'Observed flag must be zero or one')
        value = float(row['affinity'])
        require(math.isfinite(value) and value >= self.affinity['minimum']
                and (self.affinity['maximum'] is None or value <= self.affinity['maximum']),
                'Observed family affinity is outside declared range')
        support = self.upstream['support']
        if support['row_encoding'] == 'NUMERIC_SOURCE_COMPONENT':
            numerator, denominator = float(row['support_numerator']), float(row['support_denominator'])
            require(math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0,
                    'Observed source support must have a positive denominator')
        else:
            require(support['row_encoding'] == 'COMPOSITE_SOURCES_PINNED'
                    and row['support_numerator'] == row['support_denominator'] == 'NA',
                    'Composite affinity support must be retained by pinned components')
        return value

    def fit_edges(self):
        previous = None; seen = 0
        with self.inputs['fit_affinities.private.tsv'].open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            require(reader.fieldnames == FAMILY_EDGE_COLUMNS, 'Family FIT edge columns differ')
            for row in reader:
                require(None not in row and None not in row.values(), 'Malformed family FIT row')
                i, j = int(row['left_index']), int(row['right_index'])
                require(i in self.fit_set and j in self.fit_set and i < j
                        and (previous is None or previous < (i, j))
                        and self.persons[i]['component'] != self.persons[j]['component'],
                        'Family FIT edge roster/order/component differs')
                value = self._value(row)
                require(value is not None, 'Unknown FIT support cannot become a graph zero')
                previous = (i, j); seen += 1
                yield i, j, value
        require(seen == self.expected_edges, 'Complete eligible family FIT edges required')
        self.verify()

    def query_rows(self, *, roles=('SELECT', 'EVALUATE')):
        require(isinstance(roles, tuple) and roles and len(set(roles)) == len(roles)
                and set(roles) <= {'SELECT', 'EVALUATE'},
                'Explicit family held-out feature roles required')
        self.verify(); fit = sorted(self.fit_set)
        with self.inputs['query_affinities.private.tsv'].open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            require(reader.fieldnames == FAMILY_EDGE_COLUMNS, 'Family query edge columns differ')
            for query in self.query_indices:
                observations = {}; masked = 0
                for reference in fit:
                    row = next(reader, None)
                    require(row is not None and None not in row and None not in row.values(),
                            'Missing or malformed family query row')
                    require((int(row['left_index']), int(row['right_index'])) == (query, reference),
                            'Family query-to-FIT roster/order differs')
                    value = self._value(row); observations[reference] = value
                    masked += value is None
                if self.persons[query]['role'] in roles:
                    yield query, observations, dict(
                        observed_fit_count=len(fit)-masked, total_fit_count=len(fit),
                        family=self.family, measure=self.affinity['measure'],
                        units=self.affinity['units'], target_outcomes_read=False)
            require(next(reader, None) is None, 'Unexpected extra family query row')
        self.verify()

    def graph_semantics(self):
        return dict(family=self.family, measure=self.affinity['measure'],
                    units=self.affinity['units'], minimum=self.affinity['minimum'],
                    maximum=self.affinity['maximum'])

    def verify(self):
        require(all(sha(self.inputs[name]) == digest for name, digest in self.pins.items()),
                'Family affinity input changed after admission')


def component_pair_positive_scale(edges, persons):
    """Positive FIT mean with equal total opportunity per component pair."""
    sizes = Counter(p['component'] for p in persons if p['role'] == 'FIT')
    total = mass = 0.0; count = 0
    for i, j, value in edges:
        if value > 0:
            weight = 1.0/(sizes[persons[i]['component']]*sizes[persons[j]['component']])
            total += weight*value; mass += weight; count += 1
    require(count > 0 and math.isfinite(total) and total > 0
            and math.isfinite(mass) and mass > 0,
            'FIT-positive component-pair scale is undefined')
    return total/mass, count, mass


class AuthenticatedCompositeAffinityView:
    """Lazy R+C view over one authenticated C table and one rare view.

    The tiny composite specification avoids copying the same 2.75 million pair
    rows for every mixture weight.  It exposes the identical ``fit_edges`` and
    ``query_rows`` interface as materialized families, while preserving unknown
    rare query support as unknown in the mixture.
    """
    def __init__(self, descriptor, *, evidence_scope, max_input_bytes,
                 max_metadata_bytes, expected_family='R_PLUS_C'):
        require(set(descriptor) == {'path', 'sha256'} and discrete.valid_hash(descriptor['sha256']),
                'Exact lazy composite specification pin required')
        path = Path(descriptor['path'])
        require(path.is_absolute() and path.is_file() and path.stat().st_size <= max_metadata_bytes
                and sha(path) == descriptor['sha256'], 'Lazy composite specification changed')
        m = self.upstream = load_json(path)
        required = {'schema','stage','family','mix_weight','normalization_policy','normalization',
                    'common_inputs','rare_inputs','territory','role_counts','persons_content_sha256',
                    'capacity','target_outcomes_read','evaluate_scored','scientific_admission',
                    'biological_validation'}
        require(set(m) == required and m['schema'] == COMPOSITE_SCHEMA
                and m['stage'] == 'LAZY_COMPOSITE_READY_NOT_SCIENTIFICALLY_ADMITTED'
                and m['family'] == expected_family == 'R_PLUS_C'
                and type(m['mix_weight']) in (int,float) and math.isfinite(m['mix_weight'])
                and 0 <= m['mix_weight'] <= 1
                and m['normalization_policy'] == 'COMPONENT_PAIR_EQUAL_POSITIVE_MEAN'
                and m['target_outcomes_read'] is False and m['evaluate_scored'] is False
                and m['scientific_admission'] is False and m['biological_validation'] is False,
                'Exact non-admitted lazy R+C semantics required')
        self.common = AuthenticatedAffinityView(m['common_inputs'], evidence_scope=evidence_scope,
            max_input_bytes=max_input_bytes, max_metadata_bytes=max_metadata_bytes, expected_family='C')
        self.rare = AuthenticatedRareView(m['rare_inputs'], evidence_scope=evidence_scope,
            max_input_bytes=max_input_bytes, max_metadata_bytes=max_metadata_bytes)
        require(self.common.persons == self.rare.persons
                and self.common.upstream['territory'] == m['territory']
                and self.rare.upstream['block_id'] == m['territory']['block_id']
                and self.rare.upstream['exclusion_scope'] == m['territory']['exclusion_scope']
                and m['role_counts'] == self.common.upstream['role_counts']
                and m['persons_content_sha256'] == self.common.upstream['persons_content_sha256'],
                'Lazy R+C roster/territory binding differs')
        rs, rn, rm = component_pair_positive_scale(self.rare.fit_edges(), self.rare.persons)
        cs, cn, cm = component_pair_positive_scale(self.common.fit_edges(), self.common.persons)
        expected = dict(policy='COMPONENT_PAIR_EQUAL_POSITIVE_MEAN',
            scope='ELIGIBLE_CROSS_COMPONENT_FIT_EDGES_ONLY',
            fit_positive_rare_scale=rs, fit_positive_common_scale=cs,
            n_positive_rare_fit_edges=rn, n_positive_common_fit_edges=cn,
            positive_rare_fit_component_pair_mass=rm,
            positive_common_fit_component_pair_mass=cm, queries_used_for_scale=False)
        require(m['normalization'] == expected, 'Lazy R+C FIT-only scales differ from pinned components')
        self.persons=self.common.persons;self.roles=self.common.roles
        self.fit_set=self.common.fit_set;self.query_indices=self.common.query_indices
        self.expected_edges=self.common.expected_edges;self.family='R_PLUS_C'
        self.affinity=dict(measure='FIT_NORMALIZED_RARE_PLUS_COMMON_AFFINITY',
            units='WEIGHTED_SUM_OF_COMPONENTS_IN_FIT_POSITIVE_MEAN_UNITS',
            transformation='(1-w)*R/sR+w*C/sC_WITH_C_POSITIVE_COMMON_KERNEL',
            minimum=0.0,maximum=None)
        self.source_sha256=descriptor['sha256'];self.spec_path=path

    def fit_edges(self):
        weight=self.upstream['mix_weight'];norm=self.upstream['normalization']
        rare=iter(self.rare.fit_edges());common=iter(self.common.fit_edges())
        while True:
            r=next(rare,None);c=next(common,None)
            require((r is None)==(c is None),'Lazy R+C FIT edge counts differ')
            if r is None: break
            require(r[:2]==c[:2],'Lazy R+C FIT edge order differs')
            yield r[0],r[1],(1-weight)*r[2]/norm['fit_positive_rare_scale']+weight*c[2]/norm['fit_positive_common_scale']
        self.verify()

    def query_rows(self, *, roles=('SELECT','EVALUATE')):
        weight=self.upstream['mix_weight'];norm=self.upstream['normalization']
        rare=iter(self.rare.query_rows(roles=roles));common=iter(self.common.query_rows(roles=roles))
        while True:
            r=next(rare,None);c=next(common,None)
            require((r is None)==(c is None),'Lazy R+C query counts differ')
            if r is None: break
            require(r[0]==c[0] and set(r[1])==set(c[1]),'Lazy R+C query/FIT order differs')
            observations={}
            for reference in r[1]:
                observations[reference]=(None if r[1][reference] is None else
                    (1-weight)*r[1][reference]/norm['fit_positive_rare_scale']
                    +weight*c[1][reference]/norm['fit_positive_common_scale'])
            yield r[0],observations,dict(observed_fit_count=sum(v is not None for v in observations.values()),
                total_fit_count=len(observations),family='R_PLUS_C',measure=self.affinity['measure'],
                units=self.affinity['units'],target_outcomes_read=False)
        self.verify()

    def graph_semantics(self):
        return dict(family=self.family,measure=self.affinity['measure'],units=self.affinity['units'],
                    minimum=self.affinity['minimum'],maximum=self.affinity['maximum'])

    def verify(self):
        require(sha(self.spec_path)==self.source_sha256,'Lazy R+C specification changed during use')
        self.common.verify();self.rare.verify()
