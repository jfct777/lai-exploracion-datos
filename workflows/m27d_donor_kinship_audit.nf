nextflow.enable.dsl=2

include {
    WRITE_DONOR_KINSHIP_RUN_PROVENANCE;
    VERIFY_DONOR_KINSHIP_PREPARED_INPUTS;
    PREPARE_DONOR_KINSHIP_RESOURCES;
    BENCHMARK_DONOR_KINSHIP_RESOURCES;
    RESOLVE_DONOR_KINSHIP_STRATA;
    AUDIT_BASELINE_DONOR_IDENTITY;
    RUN_DONOR_KINSHIP_PASS0;
    FIT_DONOR_KINSHIP_PCA;
    RUN_DONOR_KINSHIP_CONFIGURATION;
    SELECT_DONOR_KINSHIP_CANDIDATES;
    COMPARE_DONOR_KINSHIP_PC_COUNT
} from '../modules/27D_DONOR_KINSHIP_AUDIT'

// The M27D phases are launched one at a time on purpose.  An incomplete preparation
// must never flow straight into a paid PC-Relate pass, and the audit must never start
// without somebody reading the preparation hashes first.
// The phase list and the subset that requires an explicit human authorization are read
// from the preregistration, never restated here.  They were restated once, in the gate
// and again in the provenance record, and the copies drifted far enough that an
// authorized pass0 published itself as unauthorized.  bin/m27d_run_provenance.py reads
// the same two lists from the same file, so there is nothing left to fall out of step.
def authorizationPolicy(contract) {
    def block = contract?.authorization
    if( !(block instanceof Map) ) {
        throw new IllegalStateException('M27D preregistration declares no authorization block.')
    }
    def phases = block.phases
    def gated = block.phases_requiring_explicit_authorization
    if( !(phases instanceof List) || !phases ) {
        throw new IllegalStateException('M27D preregistration declares no authorization.phases.')
    }
    if( !(gated instanceof List) ) {
        throw new IllegalStateException(
            'M27D preregistration declares no authorization.phases_requiring_explicit_authorization.'
        )
    }
    def unknown = (gated as Set) - (phases as Set)
    if( unknown ) {
        throw new IllegalStateException("M27D gated phases are not declared phases: ${unknown.sort().join(', ')}")
    }
    [phases: phases.collect { it.toString() }, gated: gated.collect { it.toString() }]
}

def sortedAutosomes(pattern, description) {
    channel
        .fromPath(pattern, checkIfExists: true)
        .collect()
        .map { paths ->
            def numbered = paths.collect { path ->
                def matcher = (path.getName() =~ /(?:^|[._])(?:chr)?(\d{1,2})[._]/)
                if( !matcher.find() ) {
                    throw new IllegalStateException("M27D could not parse a chromosome from ${description}: ${path.getName()}")
                }
                [(matcher.group(1) as int), path]
            }
            def chromosomes = numbered.collect { it[0] }
            if( chromosomes.toSorted() != (1..22).toList() ) {
                throw new IllegalStateException("M27D expects exactly autosomes 1-22 for ${description}; found ${chromosomes.toSorted()}")
            }
            numbered.toSorted { left, right -> left[0] <=> right[0] }.collect { it[1] }
        }
}

workflow {
    def repoDir = projectDir.resolve('..')
    // Overridable so a test can point at a synthetic contract without rewriting the
    // repository file in place. Swapping the real preregistration on disk to run a test
    // risks leaving a fixture value committed, which would quietly destroy the evidence
    // that the configurations were fixed before the results existed.
    def preregistration = file(
        params.donor_kinship_preregistration ?: "${repoDir}/conf/m27d_donor_kinship_preregistration.json",
        checkIfExists: true,
    )
    def contract = new groovy.json.JsonSlurper().parse(preregistration)
    if( contract.pcrelate.king_allowed ) {
        throw new IllegalStateException('M27D preregistration must forbid KING.')
    }
    def policy = authorizationPolicy(contract)
    // The phases this file actually implements, checked against the contract before
    // anything runs.  A phase declared in the preregistration without a branch used to
    // pass the launch gate and fall through into the audit block: it would have re-run
    // pass0, audited baseline identity over twenty-two VCFs and executed all four
    // configurations, while the provenance record described a narrow technical phase.
    def IMPLEMENTED_PHASES = ['prepare', 'benchmark', 'strata', 'pass0', 'pc_sensitivity', 'audit']
    def declared = policy.phases as Set
    if( declared != (IMPLEMENTED_PHASES as Set) ) {
        throw new IllegalStateException(
            'M27D phases drifted between the preregistration and the workflow. Declared but ' +
            "not implemented: ${(declared - (IMPLEMENTED_PHASES as Set)).sort()}; implemented " +
            "but not declared: ${((IMPLEMENTED_PHASES as Set) - declared).sort()}."
        )
    }

    def phase = params.donor_kinship_phase?.toString()
    if( !(phase in policy.phases) ) {
        throw new IllegalStateException("M27D phase must be one of ${policy.phases.join(', ')}.")
    }
    def phaseConsumesAuthorization = phase in policy.gated
    if( !phaseConsumesAuthorization && !params.donor_kinship_smoke_only ) {
        throw new IllegalStateException(
            'M27D technical phases run with --donor_kinship_smoke_only true.'
        )
    }
    if( phaseConsumesAuthorization ) {
        if( params.donor_kinship_smoke_only ) {
            throw new IllegalStateException(
                "M27D ${phase} is part of the full donor run. Set --donor_kinship_smoke_only false."
            )
        }
        if( !params.donor_kinship_full_run_authorized ) {
            throw new IllegalStateException(
                "M27D ${phase} needs an explicit human authorization: --donor_kinship_full_run_authorized true."
            )
        }
    }

    def pcrelateSmokeR = file("${repoDir}/bin/m27d_resource_smoke.R", checkIfExists: true)
    def verifyPreparedPy = file("${repoDir}/bin/verify_m27d_prepared_inputs.py", checkIfExists: true)
    def manifestPy = file("${repoDir}/bin/write_stage_manifest.py", checkIfExists: true)
    def sampleStrataPy = file("${repoDir}/bin/m27d_prepare_sample_strata.py", checkIfExists: true)
    def bridgePy = file("${repoDir}/bin/audit_rare_scaffold_bridge.py", checkIfExists: true)
    def commonR = file("${repoDir}/bin/m27d_common.R", checkIfExists: true)
    def kinshipGraphPy = file("${repoDir}/bin/m27d_kinship_graph.py", checkIfExists: true)

    def gitCommit = System.getenv('DNABR_GIT_COMMIT') ?: 'unknown'
    def provenance = [
        git_commit       : gitCommit,
        nextflow_version : workflow.nextflow.version.toString(),
        container_path   : params.donor_kinship_container_image,
        container_sha256 : params.donor_kinship_container_digest,
        run_id           : params.cloud_run_id,
    ]
    def provenanceB64 = groovy.json.JsonOutput.toJson(provenance).bytes.encodeBase64().toString()
    def runProvenance = provenance + [
        nextflow_command : workflow.commandLine,
        launch_dir       : workflow.launchDir.toString(),
        project_dir      : projectDir.toString(),
        scientific_scope : phase == 'audit'
            ? 'M27D donor kinship and disjointness audit; PC-Relate without KING; no Gnomix, simulation, training or TEST'
            : "M27D ${phase} technical phase only; PC-Relate without KING; no donor certification, Gnomix, simulation, training or TEST",
        compute_region   : params.cloud_region,
        panel_vcf_glob   : params.donor_kinship_panel_vcf_glob,
        sample_metadata  : params.donor_kinship_metadata,
        exclude_regions  : params.donor_kinship_exclude_regions_bed,
        thread_grid      : params.donor_kinship_thread_grid,
        phase            : phase,
        architecture     : 'persistent marker preparation, reusable benchmark, then a single-training-set donor audit',
    ]
    // The authorization fields are deliberately absent here.  They are added by
    // bin/m27d_run_provenance.py, which derives them from the same preregistration the
    // gate above consulted, so the record cannot disagree with the gate that let the run
    // start.  Restating them in Groovy is what produced the discrepancy.
    def runProvenanceB64 = groovy.json.JsonOutput.toJson(runProvenance).bytes.encodeBase64().toString()

    WRITE_DONOR_KINSHIP_RUN_PROVENANCE(
        channel.value(runProvenanceB64),
        channel.value(phase),
        channel.value(params.donor_kinship_full_run_authorized ? 'true' : 'false'),
        channel.value(preregistration),
        channel.value(file("${repoDir}/bin/m27d_run_provenance.py", checkIfExists: true)),
    )

    if( phase == 'prepare' ) {
        def panelVcfs = sortedAutosomes(params.donor_kinship_panel_vcf_glob, 'the official panel')
        def metadata = file(params.donor_kinship_metadata, checkIfExists: true)
        def excludeBed = file(params.donor_kinship_exclude_regions_bed, checkIfExists: true)
        def prepareResourcesR = file("${repoDir}/bin/m27d_prepare_genotype_resources.R", checkIfExists: true)
        PREPARE_DONOR_KINSHIP_RESOURCES(
            panelVcfs,
            channel.value(metadata),
            channel.value(excludeBed),
            channel.value(preregistration),
            channel.value(sampleStrataPy),
            channel.value(prepareResourcesR),
            channel.value(bridgePy),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )
        return
    }

    if( phase == 'strata' ) {
        def panelVcfs = sortedAutosomes(params.donor_kinship_panel_vcf_glob, 'the official panel')
        RESOLVE_DONOR_KINSHIP_STRATA(
            panelVcfs.map { paths -> paths[0] },
            channel.value(file(params.donor_kinship_metadata, checkIfExists: true)),
            channel.value(preregistration),
            channel.value(sampleStrataPy),
            channel.value(bridgePy),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )
        return
    }

    // Both remaining phases reuse the published preparation instead of rebuilding the
    // GDS, so they must name it explicitly and prove it is the reviewed one.
    def requiredPrepared = [
        donor_kinship_prepared_gds: params.donor_kinship_prepared_gds,
        donor_kinship_prepared_anchor_rds: params.donor_kinship_prepared_anchor_rds,
        donor_kinship_prepared_strict_rds: params.donor_kinship_prepared_strict_rds,
        donor_kinship_preparation_manifest: params.donor_kinship_preparation_manifest,
        donor_kinship_preparation_manifest_sha256: params.donor_kinship_preparation_manifest_sha256,
    ]
    if( phase == 'benchmark' ) {
        requiredPrepared.donor_kinship_prepared_strata = params.donor_kinship_prepared_strata
    }
    def missingPrepared = requiredPrepared.findAll { key, value -> !value }.keySet()
    if( missingPrepared ) {
        throw new IllegalStateException("M27D ${phase} is missing prepared inputs: ${missingPrepared.join(', ')}")
    }

    def preparedGds = file(params.donor_kinship_prepared_gds, checkIfExists: true)
    def anchorRds = file(params.donor_kinship_prepared_anchor_rds, checkIfExists: true)
    def strictRds = file(params.donor_kinship_prepared_strict_rds, checkIfExists: true)
    def preparationManifest = file(params.donor_kinship_preparation_manifest, checkIfExists: true)

    if( phase == 'benchmark' ) {
        BENCHMARK_DONOR_KINSHIP_RESOURCES(
            channel.value(preparedGds),
            channel.value(anchorRds),
            channel.value(strictRds),
            channel.value(file(params.donor_kinship_prepared_strata, checkIfExists: true)),
            channel.value(preparationManifest),
            channel.value(params.donor_kinship_preparation_manifest_sha256),
            channel.value(preregistration),
            channel.value(pcrelateSmokeR),
            channel.value(verifyPreparedPy),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )
        return
    }


    // ---- audit, pass0 and pc_sensitivity ---------------------------------------
    // All three consume a GDS produced by a run that finished days earlier, so the
    // manifest hash is re-checked here rather than trusted from the parameter alone.
    // Requiring the hash and never verifying it would make the parameter decorative.
    //
    // pc_sensitivity additionally reuses the pass0 training set instead of recomputing it,
    // so its hash is checked too: the two configurations must provably fit their shared
    // PCA on the same individuals, and a repointed path would leave every downstream
    // number self-consistent and wrong.
    def reusesPass0TrainingSet = phase == 'pc_sensitivity'
    if( reusesPass0TrainingSet ) {
        def missingReuse = [
            donor_kinship_pass0_training_set: params.donor_kinship_pass0_training_set,
            donor_kinship_pass0_training_set_sha256: params.donor_kinship_pass0_training_set_sha256,
        ].findAll { key, value -> !value }.keySet()
        if( missingReuse ) {
            throw new IllegalStateException(
                "M27D pc_sensitivity reuses the pass0 training set and is missing: ${missingReuse.join(', ')}"
            )
        }
    }
    def reusedTrainingSet = reusesPass0TrainingSet
        ? file(params.donor_kinship_pass0_training_set, checkIfExists: true)
        : preregistration
    VERIFY_DONOR_KINSHIP_PREPARED_INPUTS(
        channel.value(preparedGds),
        channel.value(anchorRds),
        channel.value(strictRds),
        channel.value(preparationManifest),
        channel.value(params.donor_kinship_preparation_manifest_sha256),
        channel.value(verifyPreparedPy),
        channel.value(reusedTrainingSet),
        channel.value(reusesPass0TrainingSet ? params.donor_kinship_pass0_training_set_sha256 : ''),
    )

    def panelVcfs = sortedAutosomes(params.donor_kinship_panel_vcf_glob, 'the official panel')
    def pass0R = file("${repoDir}/bin/m27d_pass0_pcrelate.R", checkIfExists: true)
    def pcaR = file("${repoDir}/bin/m27d_pca_projection.R", checkIfExists: true)
    def configurationR = file("${repoDir}/bin/m27d_pcrelate_configuration.R", checkIfExists: true)
    def baselineIdentityR = file("${repoDir}/bin/m27d_baseline_identity.R", checkIfExists: true)
    def selectionPy = file("${repoDir}/bin/m27d_candidate_selection.py", checkIfExists: true)

    // Strata are recomputed inside the audit rather than passed in, so the corrected
    // resolution and the kinship result always come from the same run and the same code.
    RESOLVE_DONOR_KINSHIP_STRATA(
        panelVcfs.map { paths -> paths[0] },
        channel.value(file(params.donor_kinship_metadata, checkIfExists: true)),
        channel.value(preregistration),
        channel.value(sampleStrataPy),
        channel.value(bridgePy),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )
    def strata = RESOLVE_DONOR_KINSHIP_STRATA.out.private_strata

    // The configuration table is read from the preregistration, never restated here, so
    // the code cannot drift from the contract it claims to implement.  The mapping from
    // r2 to a prepared marker set is an explicit lookup rather than an if/else: with a
    // fallback branch, a configuration asking for an r2 nobody prepared would quietly
    // borrow the strict set and report a result for a pruning that never happened.
    def PREPARED_MARKER_SETS = [(0.2d): 'anchor', (0.1d): 'strict']
    def markerSetFor = { r2 ->
        def match = PREPARED_MARKER_SETS.find { threshold, _id -> Math.abs((r2 as double) - threshold) < 1e-9 }
        if( !match ) {
            throw new IllegalStateException(
                "M27D has no prepared marker set for r2=${r2}; prepared: ${PREPARED_MARKER_SETS.keySet()}"
            )
        }
        match.value
    }
    def snpRdsFor = [anchor: anchorRds, strict: strictRds]

    if( phase == 'pc_sensitivity' ) {
        // A component-count comparison is only readable if the component count is the only
        // thing that differs, so the pair is validated against the contract here rather
        // than trusted from its name.  Which pair to compare stays a parameter: the
        // spectrum, not the code, decides which contrast is informative.
        def requested = params.donor_kinship_pc_sensitivity_configurations
            .toString().split(',').collect { it.trim() }.findAll { it }
        if( requested.size() != 2 ) {
            throw new IllegalStateException(
                "M27D pc_sensitivity compares exactly two configurations, got ${requested}"
            )
        }
        def selected = requested.collect { id ->
            def match = contract.configurations.find { it.id == id }
            if( !match ) {
                throw new IllegalStateException(
                    "Configuration '${id}' is not preregistered; available: " +
                    contract.configurations.collect { it.id }.join(', ')
                )
            }
            match
        }
        if( (selected[0].ld_r2_max as double) != (selected[1].ld_r2_max as double) ) {
            throw new IllegalStateException(
                'The two configurations differ in the LD threshold as well as in the component count.'
            )
        }
        if( (selected[0].n_pcs as int) == (selected[1].n_pcs as int) ) {
            throw new IllegalStateException('The two configurations use the same component count.')
        }
        def markerSetId = markerSetFor(selected[0].ld_r2_max)

        // One PCA fit, sliced twice.  Fitting per configuration would refit the axes and
        // change two things at once, which is the failure this phase exists to avoid.
        FIT_DONOR_KINSHIP_PCA(
            channel.of([markerSetId, snpRdsFor[markerSetId]]),
            channel.value(preparedGds),
            strata.first(),
            VERIFY_DONOR_KINSHIP_PREPARED_INPUTS.out.verification.map { reusedTrainingSet },
            channel.value(preregistration),
            channel.value(pcaR),
            channel.value(commonR),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )

        def sensitivityInputs = channel.fromList(selected.collect { [markerSetId, it.id] })
            .combine(FIT_DONOR_KINSHIP_PCA.out.scores, by: 0)
            .map { setId, configurationId, scores ->
                tuple(configurationId, setId, snpRdsFor[setId], scores)
            }
        RUN_DONOR_KINSHIP_CONFIGURATION(
            sensitivityInputs,
            channel.value(preparedGds),
            strata.first(),
            channel.value(reusedTrainingSet),
            channel.value(preregistration),
            channel.value(configurationR),
            channel.value(commonR),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )

        COMPARE_DONOR_KINSHIP_PC_COUNT(
            RUN_DONOR_KINSHIP_CONFIGURATION.out.pairs.collect(),
            RUN_DONOR_KINSHIP_CONFIGURATION.out.inbreeding.collect(),
            RUN_DONOR_KINSHIP_CONFIGURATION.out.summary.collect(),
            strata.first(),
            channel.value(file(params.donor_kinship_pass0_sample_universe, checkIfExists: true)),
            channel.value(file(params.donor_kinship_pass0_call_rates, checkIfExists: true)),
            channel.value(selected[0].id),
            channel.value(preregistration),
            channel.value(file("${repoDir}/bin/m27d_pc_comparison.py", checkIfExists: true)),
            channel.value(kinshipGraphPy),
            channel.value(manifestPy),
            channel.value(provenanceB64),
        )
        return
    }

    def baselineVcfs = sortedAutosomes(params.donor_kinship_baseline_vcf_glob, 'the frozen baseline')

    RUN_DONOR_KINSHIP_PASS0(
        VERIFY_DONOR_KINSHIP_PREPARED_INPUTS.out.verification.map { preparedGds },
        channel.value(anchorRds),
        strata,
        channel.value(preregistration),
        channel.value(pass0R),
        channel.value(commonR),
        channel.value(kinshipGraphPy),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )
    def trainingSet = RUN_DONOR_KINSHIP_PASS0.out.training_set

    // The contract calls pass0 the real checkpoint, so it is launchable on its own.
    // Committing the whole DAG in one go would pay for four PC-Relate configurations
    // before anybody had read the number of related pairs the first pass found.
    if( phase == 'pass0' ) {
        return
    }

    AUDIT_BASELINE_DONOR_IDENTITY(
        channel.value(preparedGds),
        channel.value(anchorRds),
        strata,
        baselineVcfs,
        channel.value(preregistration),
        channel.value(baselineIdentityR),
        channel.value(commonR),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )

    // One PCA fit per LD-pruned marker set.  The number of components is a slice of a
    // single fit, so a configuration that only changes it still changes one factor.
    def markerSets = channel.of(
        ['anchor', anchorRds],
        ['strict', strictRds],
    )
    FIT_DONOR_KINSHIP_PCA(
        markerSets,
        channel.value(preparedGds),
        strata.first(),
        trainingSet.first(),
        channel.value(preregistration),
        channel.value(pcaR),
        channel.value(commonR),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )

    def configurations = channel.fromList(
        contract.configurations.collect { config -> [markerSetFor(config.ld_r2_max), config.id] }
    )
    def snpRdsByMarkerSet = channel.of(['anchor', anchorRds], ['strict', strictRds])
    def configurationInputs = configurations
        .combine(snpRdsByMarkerSet, by: 0)
        .combine(FIT_DONOR_KINSHIP_PCA.out.scores, by: 0)
        .map { markerSetId, configurationId, snpRds, scores ->
            tuple(configurationId, markerSetId, snpRds, scores)
        }

    RUN_DONOR_KINSHIP_CONFIGURATION(
        configurationInputs,
        channel.value(preparedGds),
        strata.first(),
        trainingSet.first(),
        channel.value(preregistration),
        channel.value(configurationR),
        channel.value(commonR),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )

    SELECT_DONOR_KINSHIP_CANDIDATES(
        RUN_DONOR_KINSHIP_CONFIGURATION.out.pairs.collect(),
        strata.first(),
        RUN_DONOR_KINSHIP_PASS0.out.sample_universe,
        RUN_DONOR_KINSHIP_PASS0.out.call_rates,
        AUDIT_BASELINE_DONOR_IDENTITY.out.identities,
        AUDIT_BASELINE_DONOR_IDENTITY.out.summary
            .mix(FIT_DONOR_KINSHIP_PCA.out.summary)
            .mix(RUN_DONOR_KINSHIP_PASS0.out.summary)
            .collect(),
        channel.value(preregistration),
        channel.value(selectionPy),
        channel.value(kinshipGraphPy),
        channel.value(manifestPy),
        channel.value(provenanceB64),
    )
}
