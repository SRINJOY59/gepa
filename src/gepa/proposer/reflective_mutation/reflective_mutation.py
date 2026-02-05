# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    MinibatchSampledEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.strategies.bandit import (
    MutationStrategy,
    MultiPromptGenerator,
    ThompsonSamplingBandit,
)
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature


class ReflectiveMutationProposer(ProposeNewCandidate[DataId]):
    """
    Implements current reflective mutation flow:
    - Select candidate via selector
    - Select minibatch via sampler
    - capture_traces_and_eval -> trajectories, subsample_scores
    - skip if all scores==perfect and skip_perfect_score
    - reflection + mutate -> new candidate
    - evaluate new candidate on same minibatch -> new_subsample_scores
    - Return proposal if improved; else None
    """

    def __init__(
        self,
        logger: Any,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        perfect_score: float,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | None = None,
        callbacks: list[GEPACallback] | None = None,
        # Bandit-based multi-prompt mutation parameters
        use_bandit_mutation: bool = False,
        num_prompt_variants: int = 10,
        bandit: ThompsonSamplingBandit | None = None,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.reflection_lm = reflection_lm
        self.callbacks = callbacks

        InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)
        self.reflection_prompt_template = reflection_prompt_template

        # Bandit-based multi-prompt mutation
        self.use_bandit_mutation = use_bandit_mutation
        self.num_prompt_variants = num_prompt_variants
        self.bandit = bandit if bandit is not None else ThompsonSamplingBandit()
        self.prompt_generator = MultiPromptGenerator()
        self._last_selected_strategy: MutationStrategy | None = None

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")
        new_texts: dict[str, str] = {}
        for name in components_to_update:
            # Gracefully handle cases where a selected component has no data in reflective_dataset
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(f"Component '{name}' is not in reflective dataset. Skipping.")
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]
            new_texts[name] = InstructionProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                    "prompt_template": self.reflection_prompt_template,
                },
            )["new_instruction"]
        return new_texts

    def propose_new_texts_with_bandit(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, list[str]]:
        """
        Generate K prompt variants for each component using bandit-selected mutation strategy.
        
        Returns:
            Dictionary mapping component names to lists of K prompt variants.
        """
        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided for bandit-based mutation.")
        
        # Select mutation strategy using Thompson Sampling
        strategy = self.bandit.select_strategy()
        self._last_selected_strategy = strategy
        self.logger.log(f"Bandit selected mutation strategy: {strategy.value}")
        
        # Log bandit statistics
        stats = self.bandit.get_strategy_stats()
        self.logger.log(f"Bandit statistics: {stats}")
        
        all_variants: dict[str, list[str]] = {}
        
        for name in components_to_update:
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(f"Component '{name}' is not in reflective dataset. Skipping.")
                continue
            
            base_instruction = candidate[name]
            dataset_with_feedback = list(reflective_dataset[name])
            
            # Format feedback for the prompt
            feedback_text = self.prompt_generator.format_feedback(dataset_with_feedback)
            
            # Generate K variants using the selected strategy
            self.logger.log(f"\n[MUTATION CONTEXT for component '{name}']")
            self.logger.log(f"Feedback being used for mutation:\n{feedback_text}\n{'-'*20}")
            
            self.logger.log(f"Generating {self.num_prompt_variants} variants for component '{name}' using strategy '{strategy.value}'...")
            variants = self.prompt_generator.generate_variants(
                base_instruction=base_instruction,
                feedback_text=feedback_text,
                strategy=strategy,
                lm=self.reflection_lm,
                k=self.num_prompt_variants,
            )
            
            all_variants[name] = variants
            self.logger.log(f"Generated {len(variants)} variants for component '{name}':")
            for v_idx, variant_text in enumerate(variants):
                self.logger.log(f"  [Variant {v_idx+1}]\n{variant_text}\n{'-'*40}")
        
        return all_variants

    def _evaluate_variants_and_select_best(
        self,
        candidate: dict[str, str],
        variants_by_component: dict[str, list[str]],
        minibatch: list[DataInst],
        subsample_ids: list[DataId],
        state: GEPAState,
        i: int,
    ) -> tuple[dict[str, str], list[float], bool]:
        """
        Evaluate all K variants on the minibatch and select the best one.
        
        Returns:
            Tuple of (best_new_texts, best_scores, has_variants)
        """
        if not variants_by_component:
            return {}, [], False
        
        # Get the first component's variants count (all should be the same)
        first_component = next(iter(variants_by_component))
        k = len(variants_by_component[first_component])
        
        # Evaluate each variant combination
        best_score_sum = float('-inf')
        best_new_texts: dict[str, str] = {}
        best_scores: list[float] = []
        
        for variant_idx in range(k):
            # Create candidate with this variant
            test_candidate = candidate.copy()
            for component_name, variants in variants_by_component.items():
                test_candidate[component_name] = variants[variant_idx]
            
            # Evaluate this variant
            def evaluator(b, c):
                r = self.adapter.evaluate(b, c, capture_traces=False)
                return r.outputs, r.scores, list(r.objective_scores) if r.objective_scores else None
            
            outputs_by_id, scores_by_id, _, actual_evals = state.cached_evaluate_full(
                test_candidate, subsample_ids, self.trainset.fetch, evaluator
            )
            state.increment_evals(actual_evals)
            
            scores = [scores_by_id[eid] for eid in subsample_ids]
            score_sum = sum(scores)
            
            self.logger.log(f"Variant {variant_idx + 1}/{k}: score_sum = {score_sum:.4f}")
            
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_scores = scores
                best_new_texts = {
                    name: variants[variant_idx] 
                    for name, variants in variants_by_component.items()
                }
        
        self.logger.log(f"Best variant score: {best_score_sum:.4f}")
        
        return best_new_texts, best_scores, True

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        i = state.i + 1

        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        
        self.logger.log(f"\n{'='*40}\nSTARTING BANDIT MUTATION: ITERATION {i}\n{'='*40}")
        self.logger.log(
            f"Selected parent candidate {curr_prog_id} with score: {state.program_full_scores_val_set[curr_prog_id]}"
        )

        # Notify candidate selected
        notify_callbacks(
            self.callbacks,
            "on_candidate_selected",
            CandidateSelectedEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                candidate=curr_prog,
                score=state.program_full_scores_val_set[curr_prog_id],
            ),
        )

        self.experiment_tracker.log_metrics(
            {"iteration": i, "selected_program_candidate": curr_prog_id, "total_metric_calls": state.total_num_evals},
            step=i,
        )

        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

        # Notify minibatch sampled
        notify_callbacks(
            self.callbacks,
            "on_minibatch_sampled",
            MinibatchSampledEvent(
                iteration=i,
                minibatch_ids=subsample_ids,
                trainset_size=len(self.trainset),
            ),
        )

        # 1) Evaluate current program with traces
        # Note: We don't use cache for capture_traces=True evaluations since we need fresh traces for reflection
        curr_parent_ids = [p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None]
        is_seed_candidate = curr_prog_id == 0
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                batch_size=len(minibatch),
                capture_traces=True,
                parent_ids=curr_parent_ids,
                inputs=minibatch,
                is_seed_candidate=is_seed_candidate,
            ),
        )
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        state.increment_evals(len(subsample_ids))
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores
        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                scores=eval_curr.scores,
                has_trajectories=bool(eval_curr.trajectories),
                parent_ids=curr_parent_ids,
                outputs=eval_curr.outputs,
                trajectories=eval_curr.trajectories,
                objective_scores=eval_curr.objective_scores,
                is_seed_candidate=is_seed_candidate,
            ),
        )

        # Update cache with current program evaluation results (for future reuse when capture_traces=False)
        if state.evaluation_cache is not None:
            objective_scores_list = list(eval_curr.objective_scores) if eval_curr.objective_scores else None
            state.evaluation_cache.put_batch(
                curr_prog, subsample_ids, eval_curr.outputs, eval_curr.scores, objective_scores_list
            )

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="no_trajectories",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        if self.skip_perfect_score and all(s >= self.perfect_score for s in eval_curr.scores):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            notify_callbacks(
                self.callbacks,
                "on_evaluation_skipped",
                EvaluationSkippedEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    reason="all_scores_perfect",
                    scores=eval_curr.scores,
                    is_seed_candidate=is_seed_candidate,
                ),
            )
            return None

        self.experiment_tracker.log_metrics(
            {"subsample_score": sum(eval_curr.scores), "total_metric_calls": state.total_num_evals}, step=i
        )

        # 2) Decide which predictors to update
        predictor_names_to_update = self.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
        )

        # 3) Build reflective dataset and propose texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)

            # Convert to concrete types for callback
            reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
                k: [dict(item) for item in v] for k, v in reflective_dataset.items()
            }

            # Notify reflective dataset built
            notify_callbacks(
                self.callbacks,
                "on_reflective_dataset_built",
                ReflectiveDatasetBuiltEvent(
                    iteration=i,
                    candidate_idx=curr_prog_id,
                    components=predictor_names_to_update,
                    dataset=reflective_dataset_concrete,
                ),
            )

            # Notify proposal start
            notify_callbacks(
                self.callbacks,
                "on_proposal_start",
                ProposalStartEvent(
                    iteration=i,
                    parent_candidate=curr_prog,
                    components=predictor_names_to_update,
                    reflective_dataset=reflective_dataset_concrete,
                ),
            )

            # Choose between bandit-based multi-prompt vs. single prompt generation
            if self.use_bandit_mutation:
                # Bandit-based: generate K variants, evaluate all, select best
                variants_by_component = self.propose_new_texts_with_bandit(
                    curr_prog, reflective_dataset, predictor_names_to_update
                )
                
                if not variants_by_component:
                    self.logger.log(f"Iteration {i}: No variants generated. Skipping.")
                    return None
                
                # Evaluate all K variants and select the best
                new_texts, new_scores, has_variants = self._evaluate_variants_and_select_best(
                    curr_prog, variants_by_component, minibatch, subsample_ids, state, i
                )
                
                if not has_variants or not new_texts:
                    self.logger.log(f"Iteration {i}: No valid variants found. Skipping.")
                    return None
                
                # Log selected best candidate
                self.logger.log(f"\n[SELECTED BEST CANDIDATE for Iteration {i}]")
                for name, text in new_texts.items():
                    self.logger.log(f"Component '{name}':\n{text}\n{'-'*20}")
                self.logger.log(f"Best Variant Score: {max(new_scores)}\n")
                
                # Update bandit based on whether best variant improved over parent
                old_sum = sum(eval_curr.scores)
                new_sum = sum(new_scores)
                improved = new_sum > old_sum
                
                if self._last_selected_strategy is not None:
                    self.bandit.update(self._last_selected_strategy, improved)
                    self.logger.log(
                        f"Bandit updated: strategy={self._last_selected_strategy.value}, "
                        f"improved={improved} (old={old_sum:.4f}, new={new_sum:.4f})"
                    )
                
                # Notify proposal end
                notify_callbacks(
                    self.callbacks,
                    "on_proposal_end",
                    ProposalEndEvent(
                        iteration=i,
                        new_instructions=new_texts,
                    ),
                )
                
                for pname, text in new_texts.items():
                    self.logger.log(f"Iteration {i}: Best variant for {pname}: {text[:200]}...")
                self.experiment_tracker.log_metrics(
                    {f"new_instruction_{pname}": text for pname, text in new_texts.items()}, step=i
                )
                self.experiment_tracker.log_metrics(
                    {
                        "bandit_strategy": self._last_selected_strategy.value if self._last_selected_strategy else "none",
                        "bandit_improved": improved,
                        "num_variants_evaluated": self.num_prompt_variants,
                    },
                    step=i,
                )
                
                # Create new candidate from best variant
                new_candidate = curr_prog.copy()
                for pname, text in new_texts.items():
                    assert pname in new_candidate, f"{pname} missing in candidate"
                    new_candidate[pname] = text
                
                state.full_program_trace[-1]["new_subsample_scores"] = new_scores
                
                new_sum = sum(new_scores)
                self.experiment_tracker.log_metrics(
                    {"new_subsample_score": new_sum, "total_metric_calls": state.total_num_evals}, step=i
                )
                
                return CandidateProposal(
                    candidate=new_candidate,
                    parent_program_ids=[curr_prog_id],
                    subsample_indices=subsample_ids,
                    subsample_scores_before=eval_curr.scores,
                    subsample_scores_after=new_scores,
                    tag="reflective_mutation_bandit",
                )
            
            else:
                # Original single-prompt approach
                new_texts = self.propose_new_texts(curr_prog, reflective_dataset, predictor_names_to_update)

                # Notify proposal end
                notify_callbacks(
                    self.callbacks,
                    "on_proposal_end",
                    ProposalEndEvent(
                        iteration=i,
                        new_instructions=new_texts,
                    ),
                )

                for pname, text in new_texts.items():
                    self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
                self.experiment_tracker.log_metrics(
                    {f"new_instruction_{pname}": text for pname, text in new_texts.items()}, step=i
                )
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            import traceback

            self.logger.log(traceback.format_exc())
            return None

        # 4) Create candidate, evaluate on same minibatch (no need to capture traces)
        # (Only reached for non-bandit path; bandit path returns earlier)
        new_candidate = curr_prog.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"{pname} missing in candidate"
            new_candidate[pname] = text

        def evaluator(b, c):
            r = self.adapter.evaluate(b, c, capture_traces=False)
            return r.outputs, r.scores, list(r.objective_scores) if r.objective_scores else None

        # Evaluate new candidate (not yet in state)
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=None,
                batch_size=len(minibatch),
                capture_traces=False,
                parent_ids=[curr_prog_id],
                inputs=minibatch,
                is_seed_candidate=False,
            ),
        )

        outputs_by_id, scores_by_id, objective_by_id, actual_evals_count = state.cached_evaluate_full(
            new_candidate, subsample_ids, self.trainset.fetch, evaluator
        )
        new_scores = [scores_by_id[eid] for eid in subsample_ids]
        outputs = [outputs_by_id[eid] for eid in subsample_ids]

        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=None,
                scores=new_scores,
                has_trajectories=False,
                parent_ids=[curr_prog_id],
                outputs=outputs,
                trajectories=None,
                objective_scores=[objective_by_id[eid] for eid in subsample_ids] if objective_by_id else None,
                is_seed_candidate=False,
            ),
        )

        state.increment_evals(actual_evals_count)
        state.full_program_trace[-1]["new_subsample_scores"] = new_scores

        new_sum = sum(new_scores)
        self.experiment_tracker.log_metrics(
            {"new_subsample_score": new_sum, "total_metric_calls": state.total_num_evals}, step=i
        )

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=new_scores,
            tag="reflective_mutation",
        )

