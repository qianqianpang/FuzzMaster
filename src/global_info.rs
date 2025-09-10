use crate::evm::{BRANCH_COVERAGE, BRANCH_COVERAGE_LIST, INSTRUCTION_COVERAGE, INSTRUCTION_COVERAGE_LIST, MUTATE_COUNT};
use lazy_static::lazy_static;
use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Mutex;

pub static IS_OBJECTIVE: AtomicBool = AtomicBool::new(false);
pub static IS_CMP_INTERESTING: AtomicBool = AtomicBool::new(false);
pub static IS_DATAFLOW_INTERESTING: AtomicBool = AtomicBool::new(false);
pub static IS_INSTRUCTION_INTERESTING: AtomicI32 = AtomicI32::new(0);
pub static IS_BRANCH_COVERAGE_INTERESTING: AtomicBool = AtomicBool::new(false);
pub static IS_INSTRUCTION_COVERAGE_INTERESTING: AtomicBool = AtomicBool::new(false);
// pub static DATAFLOW_INTERESTINGNESS:AtomicI32   = AtomicI32::new(0);
// pub static CMP_INTERESTINGNESS:AtomicI32   = AtomicI32::new(0);
pub static VALUE: AtomicI32 = AtomicI32::new(0);

//超参数
pub static mut RANDOM_P: f64 = 0.2;
const A: f64 = 3.0;//is_objective
const B: f64 = 2.0;//is_cmp_interesting
const C: f64 = 2.0;//is_dataflow_interesting
const D: f64 = 1.0;//is_instruction_interesting
const E: f64 = 3.0;//is_branch_coverage_interesting
const F: f64 = 3.0;//is_instruction_coverage_interesting

lazy_static! {
    pub static ref USE_MULTI_ARMED_BANDIT: Mutex<bool> = Mutex::new(true);
    pub static ref USE_DQN: Mutex<bool> = Mutex::new(false);
}


//ptable
lazy_static! {
    //这里的初始值，也可以随机初始化  为和=1的值
    pub static ref P_TABLE: Mutex<HashMap<&'static str, HashMap<&'static str, f64>>> = {
        let mut table = HashMap::new();

         // 添选择变异env  or abi-args
        let mut input_mutate_map = HashMap::new();
        input_mutate_map.insert("INPUT_MUTATE_ENV", 0.5355183689358782);
        input_mutate_map.insert("INPUT_MUTATE_ARGS", 0.4644816310641217);
        table.insert("INPUT_MUTATE", input_mutate_map);

        //变异ABI-args
        let mut t256_address_map = HashMap::new();
        t256_address_map.insert("T256_ADDRESS_RANDOM", 0.4920636834743438);
        t256_address_map.insert("T256_ADDRESS_SELF", 0.5079363165256562);
        table.insert("T256_ADDRESS", t256_address_map);

        let mut tarray_dynamic_map = HashMap::new();
        tarray_dynamic_map.insert("TARRAY_DYNAMIC_RANDOM", 0.15670148765488864);
        tarray_dynamic_map.insert("TARRAY_DYNAMIC_INCREASE", 0.2950168574272999);
        tarray_dynamic_map.insert("TARRAY_DYNAMIC_DECREASE", 0.5482816549178113);
        table.insert("TARRAY_DYNAMIC", tarray_dynamic_map);

        let mut tunknown = HashMap::new();
        tunknown.insert("TUNKNOWN_SLOT", 0.29614490719018544);
        tunknown.insert("TUNKNOWN_ABI", 0.7038550928098146);
        table.insert("TUNKNOWN", tunknown);

        //变异env
        let mut env_map = HashMap::new();
        env_map.insert("ENV_CALLER", 0.10628936761790735);
        env_map.insert("ENV_BALANCE", 0.012087685458521048);
        env_map.insert("ENV_GASPRICE", 0.024347533568454243);
        env_map.insert("ENV_BASEFEE", 0.16699855358421933);
        env_map.insert("ENV_TIMESTAMP", 0.06928435516295682);
        env_map.insert("ENV_COINBASE", 0.10595742814793209);
        env_map.insert("ENV_GASLIMIT", 0.006449580092650405);
        env_map.insert("ENV_NUMBER", 0.16210237119036905);
        env_map.insert("ENV_CALLVALUE", 0.06042721599214044);
        env_map.insert("ENV_PREVRANDAO", 0.2196002076744771);
        env_map.insert("ENV_DIFFICULTY", 0.06645570151037233);
        table.insert("ENV", env_map);


        //具体的byte_mutations操作
        let mut byte_mutations_map = HashMap::new();
        byte_mutations_map.insert("ByteRandMutator", 0.04859009581991758);
        byte_mutations_map.insert("ByteInterestingMutator", 0.10802973675191048);
        byte_mutations_map.insert("DwordAddMutator", 0.04342976642042233);
        byte_mutations_map.insert("ByteNegMutator", 0.08530105944006433);
        byte_mutations_map.insert("ByteIncMutator", 0.07497920490450842);
        byte_mutations_map.insert("ByteAddMutator", 0.11014066253434186);
        byte_mutations_map.insert("ByteFlipMutator", 0.020403173323224978);
        byte_mutations_map.insert("ByteDecMutator", 0.09477373713374436);
        byte_mutations_map.insert("BytesSwapMutator", 0.021522091252595292);
        byte_mutations_map.insert("BitFlipMutator", 0.06271614938134926);
        byte_mutations_map.insert("ConstantHintedMutator", 0.040649101030637465);
        byte_mutations_map.insert("BytesRandSetMutator", 0.040372126465043574);
        byte_mutations_map.insert("WordInterestingMutator", 0.030164440054414102);
        byte_mutations_map.insert("DwordInterestingMutator", 0.11150981503928818);
        byte_mutations_map.insert("BytesSetMutator", 0.04896178921838225);
        byte_mutations_map.insert("WordAddMutator", 0.015502141527304095);
        byte_mutations_map.insert("QwordAddMutator", 0.04295490970285134);
        table.insert("BYTE_MUTATIONS", byte_mutations_map);

        //选择mutate with template,state,data,bibao
        let mut mutate_template_map = HashMap::new();
        mutate_template_map.insert("USE_TEMPLATE", 0.8007821388257128);
        mutate_template_map.insert("NOT_USE", 0.19921786117428725);
        table.insert("MUTATE_TEMPLATE", mutate_template_map);

        let mut mutate_state_map = HashMap::new();
        mutate_state_map.insert("USE_STATE", 0.23152363034373602);
        mutate_state_map.insert("NOT_USE", 0.7684763696562639);
        table.insert("MUTATE_STATE", mutate_state_map);

        let mut mutate_data_map = HashMap::new();
        mutate_data_map.insert("USE_DATA", 0.5314618084604269);
        mutate_data_map.insert("NOT_USE", 0.4685381915395731);
        table.insert("MUTATE_DATA", mutate_data_map);

        let mut mutate_byte_map = HashMap::new();
        mutate_byte_map.insert("MUTATE_LIQUIDATE", 0.23296299140488302);
        mutate_byte_map.insert("MUTATE_NORMAL", 0.7670370085951169);
        table.insert("MUTATE_BYTE", mutate_byte_map);

         let mut mutate_borrow_map = HashMap::new();
        mutate_borrow_map.insert("MUTATE_RANDOMNESS", 0.8621228379832364);
        mutate_borrow_map.insert("MUTATE_NORMAL", 0.13787716201676353);
        table.insert("MUTATE_BORROW", mutate_borrow_map);

        let mut mutate_all_map = HashMap::new();
        mutate_all_map.insert("MUTATE_LIQUIDATION", 0.10454835683618816);
        mutate_all_map.insert("MUTATE_RANDOMNESSL", 0.04699016083378935);
        mutate_all_map.insert("MUTATE_NORMAL", 0.8484614823300225);
        table.insert("MUTATE_ALL", mutate_all_map);

        Mutex::new(table)
    };
}
pub fn reset_p_table_ave() {
    let mut p_table = P_TABLE.lock().unwrap();
    for (_key, value_map) in p_table.iter_mut() {
        let equal_value = 1.0 / value_map.len() as f64;
        for (_sub_key, sub_value) in value_map.iter_mut() {
            *sub_value = equal_value;
        }
    }
}

fn generate_random_probabilities(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut probabilities: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let sum: f64 = probabilities.iter().sum();
    probabilities.iter_mut().for_each(|x| *x /= sum);
    probabilities
}

pub fn reset_p_table_ran() {
    let mut p_table = P_TABLE.lock().unwrap();
    for (_key, value_map) in p_table.iter_mut() {
        let probabilities = generate_random_probabilities(value_map.len());
        for ((_sub_key, sub_value), &probability) in value_map.iter_mut().zip(probabilities.iter()) {
            *sub_value = probability;
        }
    }
}
pub fn select_mutation_action(p_table: &Mutex<HashMap<&'static str, HashMap<&'static str, f64>>>, action_type: &'static str, p: f64) -> &'static str {
    let p_table = p_table.lock().unwrap();
    let action_map = p_table.get(action_type).unwrap();
    let mut rng = rand::thread_rng();
    let random_number: f64 = rng.gen(); // generates a float between 0 and 1

    if random_number < p {
        // select the action with the maximum probability
        let max_probability_action = action_map.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        max_probability_action.0
    } else {
        // select a random action
        let actions: Vec<&&str> = action_map.keys().collect();
        let random_action = actions[rng.gen_range(0..actions.len())];
        *random_action
    }
}
pub fn adjust_p_table() {
    // print_mutation_op();
    let value =  VALUE.load(Ordering::SeqCst);
    if value >= 1 {
        let mutation_op = MUTATION_OP.lock().unwrap();
        let mut p_table = P_TABLE.lock().unwrap();
        for (key, value_map) in mutation_op.iter() {
            let mut total = 0.0;
            for (sub_key, sub_value) in value_map.iter() {
                if *sub_value != 0 {
                    if let Some(p_value_map) = p_table.get_mut(*key) {
                        if let Some(p_value) = p_value_map.get_mut(*sub_key) {
                            *p_value += *sub_value as f64 /100.0; // 增加的概率等于MUTATION_OP中的值
                            total += *p_value;
                        }
                    }
                }
            }
            // Normalize the probabilities so that the sum is 1
            if let Some(p_value_map) = p_table.get_mut(*key) {
                if total == 0.0 {
                    let equal_value = 1.0 / p_value_map.len() as f64;
                    for p_value in p_value_map.values_mut() {
                        *p_value = equal_value;
                    }
                } else {
                    for p_value in p_value_map.values_mut() {
                        *p_value /= total;
                    }
                }
            }
        }
    }
    VALUE.store(0, Ordering::SeqCst);
    // print_p_table();
    // Reset all values in MUTATION_OP to 0
    let mut mutation_op = MUTATION_OP.lock().unwrap();
    for (_key, value_map) in mutation_op.iter_mut() {
        for (_sub_key, sub_value) in value_map.iter_mut() {
            *sub_value = 0;
        }
    }
}
pub fn print_p_table() {
    // println!("================P_TABLE======");
    let p_table = P_TABLE.lock().unwrap();
    for (key, value_map) in p_table.iter() {
        // println!("{}:", key);
        for (sub_key, sub_value) in value_map.iter() {
            // println!("     {}: {}", sub_key, sub_value);
        }
    }
}
//MUTATION_OP=============================================================================================
lazy_static! {
    pub static ref MUTATION_OP: Mutex<HashMap<&'static str, HashMap<&'static str, i32>>> = {
        let mut table = HashMap::new();

        let p_table = P_TABLE.lock().unwrap();
        for (key, value_map) in p_table.iter() {
            let mut inner_map = HashMap::new();
            for sub_key in value_map.keys() {
                inner_map.insert(*sub_key, 0);
            }
            table.insert(*key, inner_map);
        }

        Mutex::new(table)
    };
}
pub fn increment_mutation_op(key: &'static str, sub_key: &'static str) {
    let mut mutation_op = MUTATION_OP.lock().unwrap();
    if let Some(value_map) = mutation_op.get_mut(key) {
        if let Some(value) = value_map.get_mut(sub_key) {
            *value += 1;
        }
    }
}
pub fn print_mutation_op() {
    // println!("================MUTATION_OP======");
    let mutation_op = MUTATION_OP.lock().unwrap();
    for (key, value_map) in mutation_op.iter() {
        // println!("{}:", key);
        for (sub_key, sub_value) in value_map.iter() {
            // println!("     {}: {}", sub_key, sub_value);
        }
    }
}
//feedback=============================================================================================
pub fn get_feedback_info() -> [bool; 6] {
    let is_objective = IS_OBJECTIVE.load(Ordering::SeqCst);
    let is_cmp_interesting = IS_CMP_INTERESTING.load(Ordering::SeqCst);
    let is_dataflow_interesting = IS_DATAFLOW_INTERESTING.load(Ordering::SeqCst);
    let is_instruction_interesting = IS_INSTRUCTION_INTERESTING.load(Ordering::SeqCst) != 0;
    let is_branch_coverage_interesting = IS_BRANCH_COVERAGE_INTERESTING.load(Ordering::SeqCst);
    let is_instruction_coverage_interesting = IS_INSTRUCTION_COVERAGE_INTERESTING.load(Ordering::SeqCst);

    [is_objective, is_cmp_interesting, is_dataflow_interesting, is_instruction_interesting, is_branch_coverage_interesting, is_instruction_coverage_interesting]
}

pub(crate) fn print_feedback_info() {
    println!("{:?}", get_feedback_info());
}


//value相关函数=========================================================================================
pub fn calculate_value() {
    let is_objective = IS_OBJECTIVE.load(Ordering::SeqCst) as i32;
    let is_cmp_interesting = IS_CMP_INTERESTING.load(Ordering::SeqCst) as i32;
    let is_dataflow_interesting = IS_DATAFLOW_INTERESTING.load(Ordering::SeqCst) as i32;
    let is_instruction_interesting = IS_INSTRUCTION_INTERESTING.load(Ordering::SeqCst) ;

    let mutate_success_count = MUTATE_COUNT.load(Ordering::SeqCst);
    if mutate_success_count == 1 {
        IS_BRANCH_COVERAGE_INTERESTING.store(true, Ordering::SeqCst);
        IS_INSTRUCTION_COVERAGE_INTERESTING.store(true, Ordering::SeqCst);
    }
    else{
        let mut branch_coverage_list = BRANCH_COVERAGE_LIST.lock().unwrap();
        let mut instruction_coverage_list = INSTRUCTION_COVERAGE_LIST.lock().unwrap();
        if mutate_success_count != branch_coverage_list.len() {
            let last_branch_coverage = match branch_coverage_list.last() {
                Some(last_coverage) => *last_coverage,
                None => {
                    0.0
                }
            };
            let last_instruction_coverage = match instruction_coverage_list.last() {
                Some(last_coverage) => *last_coverage,
                None => {
                    0.0
                }
            };
            branch_coverage_list.push(last_branch_coverage);
            instruction_coverage_list.push(last_instruction_coverage);
        }
        // println!("mutate_success_count============{}",mutate_success_count );
        // println!("branch_coverage_list len============{}", branch_coverage_list.len());
        let branch_coverage = branch_coverage_list.last().unwrap();
        let instruction_coverage = instruction_coverage_list.last().unwrap();
        let branch_coverage_last = if branch_coverage_list.len() > 1 {
            Some(*branch_coverage_list.get(branch_coverage_list.len() - 2).unwrap())
        } else {
            None
        };

        let instruction_coverage_last = if instruction_coverage_list.len() > 1 {
            Some(*instruction_coverage_list.get(instruction_coverage_list.len() - 2).unwrap())
        } else {
            None
        };

        match branch_coverage_last {
            Some(last_coverage) => {
                if branch_coverage > &last_coverage {
                    IS_BRANCH_COVERAGE_INTERESTING.store(true, Ordering::SeqCst);
                }
            },
            None => {
            }
        }

        match instruction_coverage_last {
            Some(last_coverage) => {
                if instruction_coverage > &last_coverage {
                    IS_INSTRUCTION_COVERAGE_INTERESTING.store(true, Ordering::SeqCst);
                }
            },
            None => {
            }
        }
    }
    let is_branch_coverage_interesting = IS_BRANCH_COVERAGE_INTERESTING.load(Ordering::SeqCst) as i32;
    let is_instruction_coverage_interesting = IS_INSTRUCTION_COVERAGE_INTERESTING.load(Ordering::SeqCst) as i32;

    let instruction_cov = *INSTRUCTION_COVERAGE.lock().unwrap();
    let branch_cov = *BRANCH_COVERAGE.lock().unwrap();
    println!("instruction_cov:{},branch_cov:{}", instruction_cov,branch_cov);
    print_feedback_info();
    let value = A as i32 * is_objective
        + B as i32 * is_cmp_interesting
        + C as i32 * is_dataflow_interesting
        + D as i32 * is_instruction_interesting
        + E as i32 * is_branch_coverage_interesting
        + F as i32 * is_instruction_coverage_interesting
        + (instruction_cov * 0.1) as i32
        + (branch_cov * 0.1) as i32;

    VALUE.store(value, Ordering::SeqCst);
}
pub fn store_value(value: i32) {
    VALUE.store(value, Ordering::SeqCst);
}
pub fn get_value() -> i32 {
    VALUE.load(Ordering::SeqCst)
}
pub fn load_value() -> i32 {
    VALUE.load(Ordering::SeqCst)
}



