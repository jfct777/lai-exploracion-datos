"""Deterministic FIT person/cell traversal and recoverable masked-genotype updates.

This library executes an explicitly supplied budget; it does not choose a
scientific budget, perform SELECT, or certify convergence or biological validity.
An example hides one whole cell. Cells with an empty channel stay in the
declared update; the existing loss rejects an update lacking either denominator.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from r02_context_checkpoint import CheckpointStore, _atomic_bytes, _canonical, _sha_file
from r02_masked_genotype_update import (MaskedUpdateStore, masked_fragment_record,
    masked_streaming_update_identity, masked_update_code_hashes,
    run_masked_update_streaming)

SCHEDULE_VERSION = 'PERSON_BALANCED_CYCLIC_CELLS_V2'
COVERAGE_SCHEDULE_VERSION = 'PERSON_COVERAGE_HASHED_CELLS_V1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def ordered(values, seed, epoch, kind):
    """Hash-sort small coordinate/identity lists, never genotype values."""
    return sorted(values, key=lambda value: (digest([seed, epoch, kind, value]), value))


@dataclass(frozen=True)
class CellExample:
    epoch: int
    chromosome: int
    cell: int
    person_ids: tuple[str, ...]


@dataclass(frozen=True)
class MaskedSchedule:
    person_ids: tuple[str, ...]
    chromosome_cell_counts: tuple[tuple[int, int], ...]
    seed: int
    epochs: int
    people_per_fragment: int
    fragments_per_update: int

    def __post_init__(self):
        require(bool(self.person_ids) and len(set(self.person_ids)) == len(self.person_ids)
                and all(isinstance(p, str) and p for p in self.person_ids),
                'unique explicit FIT identities required')
        require(type(self.seed) is int, 'explicit integer schedule seed required')
        for key in ('epochs', 'people_per_fragment', 'fragments_per_update'):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0,
                    'positive integer ' + key + ' required')
        counts = self.chromosome_cell_counts
        require(bool(counts) and len(dict(counts)) == len(counts)
                and all(type(c) is int and 1 <= c <= 22 and type(n) is int and n > 0
                        for c, n in counts), 'unique autosomes and cell counts required')

    def updates(self):
        # No person-by-cell product is materialized; every declared pair appears
        # once per epoch, including the final short person/fragment groups.
        cells = [(c, i) for c, n in self.chromosome_cell_counts for i in range(n)]
        for epoch in range(self.epochs):
            people = ordered(self.person_ids, self.seed, epoch, 'people')
            coordinates = ordered(cells, self.seed, epoch, 'cells')
            group = []
            person_groups = [tuple(people[start:start+self.people_per_fragment])
                             for start in range(0, len(people), self.people_per_fragment)]
            # A truncated budget must not traverse every cell for the first
            # people before visiting anybody else. Each round visits every
            # group once, with a fixed cyclic offset on the shuffled cell axis.
            # This is a traversal rule, not independent biological sampling.
            for round_index in range(len(coordinates)):
                for group_index, ids in enumerate(person_groups):
                    chromosome, cell = coordinates[(round_index+group_index) % len(coordinates)]
                    group.append(CellExample(epoch, chromosome, cell, ids))
                    if len(group) == self.fragments_per_update:
                        yield tuple(group)
                        group = []
            if group:
                yield tuple(group)

    @property
    def update_count(self):
        fragments = math.ceil(len(self.person_ids)/self.people_per_fragment) * sum(
            n for _, n in self.chromosome_cell_counts)
        return self.epochs * math.ceil(fragments/self.fragments_per_update)

    @property
    def minimum_updates_for_all_people(self):
        """Updates needed for one visit per person; not adequate learning."""
        return math.ceil(math.ceil(len(self.person_ids)/self.people_per_fragment)
                         / self.fragments_per_update)


@dataclass(frozen=True)
class CoverageSchedule:
    """A bounded number of visits per FIT person, not a genomic epoch.

    One visit traverses every person group exactly once and assigns each group
    one cell from a hash ordering of the complete, frozen cell axis.  Cell
    choice therefore uses seed, visit number and coordinates only.  It never
    retries or substitutes a cell after inspecting genotypes or losses.
    """

    person_ids: tuple[str, ...]
    chromosome: int
    cell_count: int
    seed: int
    visits_per_person: int
    people_per_fragment: int
    fragments_per_update: int

    def __post_init__(self):
        require(bool(self.person_ids) and len(set(self.person_ids)) == len(self.person_ids)
                and all(isinstance(p, str) and p for p in self.person_ids),
                'unique explicit FIT identities required')
        require(type(self.chromosome) is int and 1 <= self.chromosome <= 22,
                'one explicit autosome required')
        for key in ('cell_count', 'visits_per_person', 'people_per_fragment',
                    'fragments_per_update'):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0,
                    'positive integer ' + key + ' required')
        require(type(self.seed) is int, 'explicit integer schedule seed required')
        require(self.fragment_count % self.fragments_per_update == 0,
                'coverage visits must end on a complete optimizer-update boundary')

    @property
    def person_groups(self):
        people = ordered(self.person_ids, self.seed, 0, 'coverage-people')
        return tuple(tuple(people[start:start+self.people_per_fragment])
                     for start in range(0, len(people), self.people_per_fragment))

    @property
    def fragment_count(self):
        return self.visits_per_person * math.ceil(
            len(self.person_ids) / self.people_per_fragment)

    @property
    def update_count(self):
        return self.fragment_count // self.fragments_per_update

    @property
    def minimum_updates_for_all_people(self):
        return math.ceil(len(self.person_groups) / self.fragments_per_update)

    def updates(self):
        cells = tuple(range(self.cell_count))
        group = []
        for visit in range(self.visits_per_person):
            coordinates = ordered(cells, self.seed, visit, 'coverage-cells')
            require(len(self.person_groups) <= len(coordinates),
                    'one visit needs at least one distinct cell per person group')
            for group_index, ids in enumerate(self.person_groups):
                group.append(CellExample(visit, self.chromosome,
                                         coordinates[group_index], ids))
                if len(group) == self.fragments_per_update:
                    yield tuple(group)
                    group = []
        require(not group, 'coverage schedule ended inside an optimizer update')

    def identity_payload(self):
        return {'schema': 'r02_masked_coverage_schedule_v1',
                'schedule_version': COVERAGE_SCHEDULE_VERSION, **asdict(self)}


def fit_cell_loader(readers, model_config, *, context_exclusion_cm, device,
                    fragment_transform=None):
    """Use authenticated source roles; reject SELECT before reading genotypes."""
    from r02_masked_training_batch import load_masked_cell
    from r02_masked_store import role_person_ids

    def load(example):
        reader = readers[example.chromosome]
        require(reader.role == 'FIT', 'training requires a FIT reader')
        require(set(example.person_ids) <= role_person_ids(reader.reader, 'FIT'),
                'scheduled person is outside authenticated FIT')
        fragment, lineage = load_masked_cell(
            reader, model_config, person_ids=example.person_ids,
            cell_index=example.cell, context_exclusion_cm=context_exclusion_cm,
            device=device)
        if fragment_transform is not None:
            fragment = fragment_transform(fragment=fragment, lineage=lineage,
                                          example=example)
        return fragment
    return load


def run_masked_campaign(model, optimizer, schedule, load_fragment, checkpoint_root,
                        *, source_identity_sha256, protocol_sha256,
                        information_condition, max_updates, max_seconds):
    """Restore the last full update and any subsequent partial gradients.

    Caller must authenticate source identity and scientific protocol separately.
    A data-only checkpoint conserves completed work; crossing a time bound stops
    before the next update, with no assertion that the scientific search ended.
    Current outer checkpoint supports CPU RNG; reject CUDA rather than silently
    dropping its random state. The existing per-update implementation supports it.
    """
    require(type(max_updates) is int and 0 < max_updates <= schedule.update_count,
            'explicit update budget must fit declared traversal')
    require(type(max_seconds) in (int, float) and math.isfinite(max_seconds)
            and max_seconds > 0, 'positive finite time limit required')
    for name, value in [('source', source_identity_sha256), ('protocol', protocol_sha256)]:
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), name + ' SHA256 required')
    require(all(p.device.type == 'cpu' for p in model.parameters()),
            'outer campaign checkpoint currently supports CPU RNG only')
    root = Path(checkpoint_root)
    schedule_identity = (schedule.identity_payload() if hasattr(schedule, 'identity_payload')
                         else {'schedule_version': SCHEDULE_VERSION, **asdict(schedule)})
    identity = {'schema': 'r02_masked_campaign_v1', 'schedule': schedule_identity,
        'model': asdict(model.config), 'source_sha256': source_identity_sha256,
        'protocol_sha256': protocol_sha256, 'condition': information_condition,
        'max_updates': max_updates, 'max_seconds': max_seconds,
        'optimizer': type(optimizer).__module__+'.'+type(optimizer).__qualname__,
        'optimizer_options': [{k:v for k,v in g.items() if k != 'params'}
                              for g in optimizer.param_groups],
        'code_sha256': {**masked_update_code_hashes(),
            'r02_masked_training_campaign.py': _sha_file(Path(__file__))}}
    # Compare the same data representation before and after JSON persistence
    # (schedule tuples and Adam betas otherwise become unequal lists on load).
    identity = json.loads(_canonical(identity))
    store = CheckpointStore(root, identity)
    started = time.monotonic()
    try:
        deadline_path = root/'DEADLINE.json'
        if deadline_path.exists():
            deadline = json.loads(deadline_path.read_text())
            require(deadline.get('identity_sha256') == digest(identity),
                    'campaign deadline belongs to another problem')
        else:
            deadline = {'identity_sha256': digest(identity),
                        'expires_at_unix': time.time()+max_seconds}
            _atomic_bytes(deadline_path, _canonical(deadline))
        step, history = store.restore(model, optimizer)
        for index, examples in enumerate(schedule.updates()):
            if index < step:
                continue
            if step >= max_updates or time.time() >= deadline['expires_at_unix']:
                break
            # Preflight authenticates all inputs and frozen denominators, while
            # releasing each tensor before the next fragment is loaded.  The
            # second pass performs forward/backward one fragment at a time.
            records = []; owned = set()
            for example in examples:
                fragment = load_fragment(example)
                record, keys = masked_fragment_record(
                    fragment, model.config.max_loci_per_channel)
                require(owned.isdisjoint(keys),
                        'owned target appears in more than one streamed fragment')
                owned.update(keys); records.append(record)
                del fragment
            part_identity = masked_streaming_update_identity(
                model, optimizer, records,
                information_condition=information_condition)
            partial = MaskedUpdateStore(root/f'partial-{index+1:08d}', part_identity)
            cleared = False
            try:
                result = run_masked_update_streaming(
                    model, optimizer, lambda fragment_index: load_fragment(
                        examples[fragment_index]), records,
                    information_condition=information_condition, checkpoint=partial)
                stats = {'schema':'r02_masked_campaign_update_v1', 'step': index+1,
                    'campaign_sha256': digest(identity), 'examples': [asdict(e) for e in examples],
                    'result': asdict(result)}
                _atomic_bytes(root/f'update-{index+1:08d}.json', _canonical(stats))
                history.append(float(result.loss)); step = index+1
                store.save(model, optimizer, step, history)
                # Full state is durable first. Each update has its own directory;
                # a crash here leaves the prior state, never reused as a new one.
                partial.clear()
                cleared = True
            finally:
                partial.close()
                if cleared:
                    (partial.root/'.LOCK').unlink()
                    partial.root.rmdir()
        return {'status': 'DECLARED_UPDATE_BUDGET_COMPLETE' if step == max_updates
                else 'TIME_LIMIT_INCOMPLETE', 'updates_completed': step,
                'updates_declared': max_updates, 'traversal_updates': schedule.update_count,
                'minimum_updates_for_all_people': schedule.minimum_updates_for_all_people,
                'all_people_visited': step >= schedule.minimum_updates_for_all_people,
                'history': history, 'elapsed_this_invocation_seconds':time.monotonic()-started,
                'deadline_includes_interrupted_time_and_downtime':True,
                'scientific_convergence_certified':False, 'evaluate_opened':False}
    finally:
        store.close()
