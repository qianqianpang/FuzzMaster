use std::marker::PhantomData;
use libafl::mutators::{ComposedByMutations, MutationId, MutationResult, Mutator, MutatorsTuple, ScheduledMutator, StdScheduledMutator};
use libafl::state::{HasCorpus, HasRand};
use libafl::Error;


use libafl_bolts::Named;
use libafl_bolts::tuples::NamedTuple;
use crate::dqn_alogritm::get_mutator_selection;
use crate::global_info::{P_TABLE,RANDOM_P, select_mutation_action};


pub struct StdScheduledMutatorQQ<I, MT, S>
where
    MT: MutatorsTuple<I, S> + NamedTuple,
    S: HasRand ,
{
    inner: StdScheduledMutator<I, MT, S>,
    phantom: PhantomData<(I, MT, S)>,
}

impl<I, MT, S> ComposedByMutations<I, MT, S> for StdScheduledMutatorQQ<I, MT, S> where MT: MutatorsTuple<I, S> + NamedTuple, S:  HasRand {
    fn mutations(&self) -> &MT {
        self.inner.mutations()
    }

    fn mutations_mut(&mut self) -> &mut MT {
        self.inner.mutations_mut()
    }
}

impl<I, MT, S> Named for StdScheduledMutatorQQ<I, MT, S> where MT: MutatorsTuple<I, S> + NamedTuple, S: HasRand {
    fn name(&self) -> &str {
        "ScheduledMutatorQQ"
    }
}

impl<I, MT, S> Mutator<I, S> for StdScheduledMutatorQQ<I, MT, S> where MT: MutatorsTuple<I, S> + NamedTuple, S:  HasRand {
    fn mutate(&mut self, state: &mut S, input: &mut I, stage_idx: i32) -> Result<MutationResult, Error> {
        self.scheduled_mutate(state, input, stage_idx)
    }
}


impl<I, MT, S> ScheduledMutator<I, MT, S> for StdScheduledMutatorQQ<I, MT, S>
where
    MT: MutatorsTuple<I, S> + NamedTuple,
    S: HasRand,
{
    fn iterations(&self, state: &mut S, input: &I) -> u64 {
        self.inner.iterations(state, input)
    }

    fn schedule(&self, state: &mut S, input: &I) -> MutationId {
        // self.inner.schedule(state, input)
        // println!("我重写的2");
        debug_assert!(!self.mutations().is_empty());
        let mutator_selection = get_mutator_selection();
        // println!(",,,,,{:?}", mutator_selection);
        let byte_expansion=mutator_selection["5_byte_expansion"];
        let detail_mutation=mutator_selection["6_detail_mutation"];
        if detail_mutation ==1{
            let idx = (detail_mutation-1) as usize;
            idx.into()
        }else if byte_expansion==2{
            let idx = (detail_mutation-1) as usize;
            idx.into()
        }else{
            let idx = 2usize;
            idx.into()
        }
    }

    fn scheduled_mutate(
        &mut self,
        state: &mut S,
        input: &mut I,
        stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let mut r = MutationResult::Skipped;
        let num = self.iterations(state, input);
        for _ in 0..num {
            let idx = self.schedule(state, input);
            let outcome = self
                .mutations_mut()
                .get_and_mutate(idx, state, input, stage_idx)?;
            if outcome == MutationResult::Mutated {
                r = MutationResult::Mutated;
            }
        }
        Ok(r)
    }
}

impl<I, MT, S> StdScheduledMutatorQQ<I, MT, S>
    where
        MT: MutatorsTuple<I, S> + NamedTuple,
        S: HasRand ,
{
    pub fn new(mutations: MT) -> Self {
        StdScheduledMutatorQQ {
            inner: StdScheduledMutator::with_max_stack_pow(mutations, 4),
            phantom: PhantomData,
        }
    }
}