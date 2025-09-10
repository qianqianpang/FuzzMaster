use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::convert::TryInto;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use bytes::Bytes;
use csv::Reader;
use lazy_static::lazy_static;
use rand::prelude::IteratorRandom;
use rand::Rng;
use revm_primitives::{Env, U256};
use tch::nn::VarStore;
use tch::{nn, nn::Module, nn::Optimizer, nn::OptimizerConfig, no_grad, Kind, Tensor};

use crate::evm::abi::BoxedABI;
use crate::evm::input::{EVMInput, EVMInputTy};
use crate::evm::mutator::AccessPattern;
use crate::evm::types::EVMAddress;
use crate::evm::{ACTION_COUNTS, ACTION_FREQUENCY, LOSS_VALUES, OPCODE_COUNTS, OPCODE_FREQUENCY, REWARD_VALUES};
use crate::global_info::get_value;
use crate::input::VMInputT;
use crate::state_input::StagedVMState;
lazy_static! {
    pub static ref GLOBAL_INPUT: Mutex<EVMInput> = Mutex::new(EVMInput {
        input_type: EVMInputTy::ABI,
        caller: EVMAddress::default(),
        contract: EVMAddress::default(),
        data: None,
        sstate: StagedVMState::default(),
        sstate_idx: 0,
        txn_value: None,
        step: false,
        env: Env::default(),
        access_pattern: Arc::new(Mutex::new(AccessPattern::default())),
        liquidation_percent: 0,
        direct_data: Bytes::new(),
        randomness: Vec::new(),
        repeat: 0,
        swap_data: HashMap::new(),
    });
}

pub fn get_global_input() -> EVMInput {
    GLOBAL_INPUT.lock().unwrap().clone()
}

pub fn set_global_input(new_input: EVMInput) {
    *GLOBAL_INPUT.lock().unwrap() = new_input;
}

//GLOBAL_MUTATION==========================================================================================================================
lazy_static! {
    pub static ref GLOBAL_MUTATION: Mutex<i64> = Mutex::new(0);
    pub static ref MUTATOR_SELECTION: Mutex<HashMap<&'static str, u8>> = {
        let mut m = HashMap::new();
        m.insert("0_mutate_mode", 0);
        m.insert("1_mutate_method", 0);
        m.insert("2_env_args", 0);
        m.insert("3_mutate_field", 0);
        m.insert("4_mutate_method", 0);
        m.insert("5_byte_expansion", 0);
        m.insert("6_detail_mutation", 0);
        Mutex::new(m)
    };
}

pub fn set_global_mutation(value: i32) {
    let nums = read_nums_from_csv("action.csv").unwrap();
    let value_str= value.to_string();
    let value_nums: Vec<i64> = nums.into_iter().filter(|num| num.to_string().starts_with(&value_str)).collect();
    let len = value_nums.len();
    let ran_idx= rand::thread_rng().gen_range(0..len);
    let ran_value = value_nums[ran_idx];
    *GLOBAL_MUTATION.lock().unwrap() = ran_value;
}

pub fn set_mutator_selection() -> HashMap<&'static str, u8> {
    let global_mutation = *GLOBAL_MUTATION.lock().unwrap();
    let global_mutation_string = global_mutation.to_string();
    let mutations: Vec<_> = global_mutation_string.chars().map(|c| c.to_digit(10).unwrap() as u8).collect();
    // println!("global_mutation: {}", global_mutation);
    let keys = vec![
        "0_mutate_mode",
        "1_mutate_method",
        "2_env_args",
        "3_mutate_field",
        "4_mutate_method",
        "5_byte_expansion",
        "6_detail_mutation",
    ];
    let mut mutator_selection: HashMap<_, _> = keys.into_iter().zip(mutations.clone().into_iter()).collect();
    let detail_mutation_value = (mutations[6] as i64) * 10 + (mutations[7] as i64);
    mutator_selection.insert("6_detail_mutation", detail_mutation_value as u8);
    let mut global_mutator_selection = MUTATOR_SELECTION.lock().unwrap();
    *global_mutator_selection = mutator_selection.clone();
    mutator_selection.clone()
}


pub fn get_mutator_selection() -> HashMap<&'static str, u8> {
    MUTATOR_SELECTION.lock().unwrap().clone()
}
//action设计======================================================================================================================
lazy_static! {
    static ref ACTIONS: Mutex<Vec<i64>> = Mutex::new(Vec::new());
}

fn read_nums_from_csv(file_path: &str) -> Result<Vec<i64>, Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    let mut nums = Vec::new();

    for result in reader.records() {
        let record = result?;
        for field in record.iter() {
            if let Ok(num) = field.parse::<i64>() {
                nums.push(num);
            }
        }
    }

    Ok(nums)
}
pub fn encode_actions() -> Vec<i32> {
    let arr : [i32; 16] = [101, 102, 201, 202, 301, 302, 410, 421, 422, 510, 521, 522, 610, 620, 631, 632];
    let vec = arr.to_vec();
    vec
}

//定一个函数，获取当前的ACTION_COUNTS，并计算出每个actioN的频率赋值给ACTION_FREQUENCY
fn update_action_frequency(action:i32) {
    let mut action_counts = ACTION_COUNTS.lock().unwrap();
    let count = action_counts.entry(action).or_insert(0);
    *count += 1;
    let all_actions = vec![101, 102, 201, 202, 301, 302, 410, 421, 422, 510, 521, 522, 610, 620, 631, 632];
    let total_count: i32 = action_counts.values().map(|&v| v as i32).sum();
    let mut frequencies = Vec::new();
    for action in all_actions {
        let count = *action_counts.get(&action).unwrap_or(&0);
        let frequency = count as f32 / total_count as f32;
        frequencies.push(frequency);
    }

    let mut action_frequency = ACTION_FREQUENCY.lock().unwrap();
    *action_frequency = frequencies;
    //打印action_counts和action_frequency
    // println!("action_counts: {:?}", action_counts);
    // println!("action_frequency: {:?}", action_frequency);
}
fn update_opcode_frequency() {
    let opcodes = vec![
        "SHA3", "CALL", "CREATE", "SELFDESTRUCT", "JUMP", "JUMPI", "SLOAD", "SSTORE"
    ];
    let opcode_counts = OPCODE_COUNTS.lock().unwrap();
    let total_count: u32 = opcodes.iter()
        .map(|&opcode| *opcode_counts.get(opcode).unwrap_or(&0) as u32)
        .sum();
    let frequencies: Vec<f32> = opcodes.iter()
        .map(|&opcode| {
            let count = *opcode_counts.get(opcode).unwrap_or(&0);
            count as f32 / total_count as f32
        })
        .collect();

    let mut opcode_frequency = OPCODE_FREQUENCY.lock().unwrap();
    *opcode_frequency = frequencies;

    // println!("opcode_counts: {:?}", opcode_counts);
    // println!("opcode_frequency: {:?}", opcode_frequency);
}
//state的设计和方法================================================================================================================
pub struct State {
    //1）每轮中以下特征都是一样的
    // sstate_initialize: bool,
    // step: bool,
    // liquidation_percent: u8,
    // randomness: Vec<u8>,
    // repeat: u64,
    // limit_contract_code_size: Option<usize>,//合约代码的大小限制
    // memory_limit: u64,//内存的硬限制。在某些情况下，例如当 gas 限制可能非常高时，建议将此设置为一个合理的值，以防止内存分配出现 panic。
    // timestamp: U256,
    // difficulty: U256,
    // // prevrandao: Option<B256>,
    // basefee: U256,
    // gas_limit: U256,
    // gas_price: U256,
    // gas_priority_fee: Option<U256>,
    // value: U256,//交易发起者愿意转账给目标地址的以太币数量
    // chain_id: Option<u64>,
    // nonce: Option<u64>,//每个地址都有一个nonce，每次发起交易后，这个nonce就会增加。这个字段用于防止交易被重复执行。// data: BoxedABI,

    //2)需要 #[cfg(feature
    // disable_balance_check: bool,//是否跳过余额检查。如果为真，将交易成本添加到余额以确保执行不会失败。
    // disable_block_gas_limit: bool,//是否禁用区块 gas 限制验证。有些情况下，允许提供的 gas 限制高于区块的 gas 限制
    // disable_eip3607: bool,//是否禁用 EIP-3607。EIP-3607 拒绝来自部署了代码的发送者的交易
    // disable_gas_refund: bool,//是否禁用所有 gas 退款
    // disable_base_fee: bool,//是否禁用 EIP-1559 交易的基础费用检查。这对于测试零 gas 价格的方法调用很有用。
    // number: U256,//区块的编号，表示该区块在区块链中的位置
    // coinbase: B160,//矿工或者区块的创建者和签名者的地址
    // get_bytes: [u8;32],//ABI
    // contract ={revm_primitives::bits::B160}，_0 [u8,20]
    // data = {core::option::Option<ityfuzz::evm::abi::BoxedABl>}
    // variant0 ={enum2$<core::option::Option>::Variant0}:value,name (NONE)
    // variant1 ={enum2$<core::option::Option>::Variant1}:value name （some）:_0:
    //b:
    // pointer={*mut dyn$<ityfuzz::evm::abi::ABl>},指向动态类型ityfuzz::evm::abi::ABl
    // vtable={*mut [u64; 3]}指向虚函数表（vtable），这个表包含了指向实现了ityfuzz::evm::abi::ABl特性的类型的方法的指针。
    //function[u8,4]
    //tag
    // sstate_state: EVMState,
    //state,swap_data,reentrancy_metadata,integer_overflow,arbitrary_calls,typed_bug,self_destruct,bug_hit,flashloan_data,post_execution,balance
    // txn_value: Option<EVMU256>,
    // direct_data: Vec<u8>,
    // access_pattern: Arc<Mutex<AccessPattern>>,
    // swap_data:
    // spec_id: SpecId,
    // perf_analyse_created_bytecodes: AnalysisKind,
    // access_list: Vec<(B160, Vec<U256>)>,//交易的访问列表。这个字段是在EIP-2930升级后引入的，用于指定交易可以访问的地址和存储槽。


    function: [u8;4],//函数签名
    action_frequency: [f32;16],//action的频率
    opcode_frequency: [f32;8],
}

impl State {
    pub fn new() -> Self {
        Self {
            // sstate_initialize: false,
            // step: false,
            // liquidation_percent: 0,
            // repeat: 0,
            // randomness: vec![0],
            // disable_balance_check: false,
            // disable_block_gas_limit: false,
            // disable_eip3607: false,
            // disable_gas_refund: false,
            // disable_base_fee: false,
            // number: U256::from(0),
            // coinbase: B160::zero(),
            // prevrandao: None,
            // limit_contract_code_size: Some(0),
            // memory_limit: 0,
            // timestamp: U256::from(0),
            // difficulty: U256::from(0),
            // basefee: U256::from(0),
            // gas_limit: U256::from(0),
            // gas_price: U256::from(0),
            // gas_priority_fee: None,
            // value: U256::from(0),
            // chain_id: None,
            // nonce: None,
            // get_bytes: [0;32],
            function: [0;4],
            action_frequency:[0.0;16],
            opcode_frequency:[0.0;8]
        }
    }

    fn u256_to_f32(value: U256) -> f32 {
        match <alloy_primitives::Uint<256, 4> as TryInto<u64>>::try_into(value) {
            Ok(u64_value) => {
                let scaled_down_value = u64_value as f64 * 0.0000000000000000001; // 乘以一个小数
                scaled_down_value as f32
            },
            Err(_) => {
                let u64_max = u64::MAX as f64;
                u64_max as f32 // 返回f32可能的最大值
            }
        }
    }

    pub fn to_tensor(&self) -> Tensor {
        // let sstate_initialize_f32 = if self.sstate_initialize { 1.0 } else { 0.0 };
        // let step_f32 = if self.step { 1.0 } else { 0.0 };
        // let liquidation_percent_f32 = self.liquidation_percent as f32;
        // let repeat_f32 = self.repeat as f32;
        // let disable_balance_check_f32 = if self.disable_balance_check { 1.0 } else { 0.0 };
        // let disable_block_gas_limit_f32 = if self.disable_block_gas_limit { 1.0 } else { 0.0 };
        // let disable_eip3607_f32 = if self.disable_eip3607 { 1.0 } else { 0.0 };
        // let disable_gas_refund_f32 = if self.disable_gas_refund { 1.0 } else { 0.0 };
        // let disable_base_fee_f32 = if self.disable_base_fee { 1.0 } else { 0.0 };
        // let limit_contract_code_size_f32 = self.limit_contract_code_size.unwrap_or(0) as f32;
        // let memory_limit_f32 = self.memory_limit as f32;
        // let timestamp_f32 = State::u256_to_f32(self.timestamp);;
        // let difficulty_f32 =  State::u256_to_f32(self.difficulty);;//现在是没有用的状态???
        // let basefee_f32 =  State::u256_to_f32(self.basefee);;
        // let gas_limit_f32 =  State::u256_to_f32(self.gas_limit);;
        // let gas_price_f32 =  State::u256_to_f32(self.gas_price);;
        // let gas_priority_fee_f32 = match self.gas_priority_fee {
        //     Some(value) => State::u256_to_f32(value),
        //     None => 0.0,
        // };
        // let value_f32 = State::u256_to_f32(self.value);
        // let chain_id_f32 = match self.chain_id {
        //     Some(value) => value as f32,
        //     None => 0.0,
        // };
        // let nonce_f32 = match self.nonce {
        //     Some(value) => value as f32,
        //     None => 0.0,
        // };
        // let mut temp: f32 = 0.0;
        // for &value in &self.randomness {
        //     temp = value as f32;
        //     break
        // }
        // let get_bytes_f32: Vec<f32> = self.get_bytes.iter().map(|&b| b as f32).collect();
      let action_frequency_f32: Vec<f32> = self.action_frequency.iter().cloned().collect();
    let function_f32: Vec<f32> = self.function.iter().map(|&b| b as f32).collect();
    let opcode_frequency_f32: Vec<f32> = self.opcode_frequency.iter().cloned().collect();

    let mut input_data = Vec::new();
    input_data.extend(action_frequency_f32);
    input_data.extend(function_f32);
    input_data.extend(opcode_frequency_f32);

    let input_tensor = Tensor::from_slice(&input_data);
    // println!("tensor~~~~~~~~~~~~~~~~~~~~{:?}", input_data);

    input_tensor
    }


}


// FuzzEnv========================================================================================================
pub struct FuzzEnv {
    state: EVMInput,
}

impl FuzzEnv {
    pub fn new() -> FuzzEnv {
        FuzzEnv {
            state: get_global_input(),
        }
    }

    pub fn reset(&mut self) -> Tensor {
        let global_input = get_global_input();
        let global_abi = global_input.get_data_abi().unwrap_or_else(|| {
            BoxedABI::default()
        });
        let state = State {
            // sstate_initialize: global_input.sstate.initialized,
            // step: global_input.step,
            // liquidation_percent: global_input.liquidation_percent,
            // repeat: global_input.repeat as u64,
            // randomness: global_input.randomness.clone(),
            // limit_contract_code_size: global_input.env.cfg.limit_contract_code_size,
            // memory_limit: global_input.env.cfg.memory_limit,
            // timestamp: global_input.env.block.timestamp,
            // difficulty: global_input.env.block.difficulty,
            // basefee: global_input.env.block.basefee,
            // gas_limit: global_input.env.block.gas_limit,
            // gas_price: global_input.env.tx.gas_price,
            // gas_priority_fee: global_input.env.tx.gas_priority_fee,
            // value: global_input.env.tx.value,
            // chain_id: global_input.env.tx.chain_id,
            // nonce: global_input.env.tx.nonce,
            function:global_abi.function,
            action_frequency:[0.0;16],
            opcode_frequency:[0.0;8]
            // get_bytes: match <[u8; 32]>::try_from(global_abi.b.get_bytes()) {
            //     Ok(bytes) => bytes,
            //     Err(_) => [0u8; 32], // provide a default value or handle the error appropriately
            // },
        };
        self.state = global_input;
        state.to_tensor()
    }
    pub fn step_1(&mut self, action: i32) {
        set_global_mutation(action);
        set_mutator_selection();
    }

    pub fn step_2(&mut self) -> (Tensor, i64){
        let reward = get_value() as i64;
        let mut reward_values = REWARD_VALUES.lock().unwrap();
        reward_values.push(reward as i32);
        let global_input = get_global_input();
        let global_abi = global_input.get_data_abi().unwrap_or_else(|| {
            BoxedABI::default()
        });
        let state = State {
            // sstate_initialize: global_input.sstate.initialized,
            // step: global_input.step,
            // liquidation_percent: global_input.liquidation_percent,
            // repeat: global_input.repeat as u64,
            // randomness: global_input.randomness.clone(),
            // limit_contract_code_size: global_input.env.cfg.limit_contract_code_size,
            // memory_limit: global_input.env.cfg.memory_limit,
            // timestamp: global_input.env.block.timestamp,
            // difficulty: global_input.env.block.difficulty,
            // basefee: global_input.env.block.basefee,
            // gas_limit: global_input.env.block.gas_limit,
            // gas_price: global_input.env.tx.gas_price,
            // gas_priority_fee: global_input.env.tx.gas_priority_fee,
            // value: global_input.env.tx.value,
            // chain_id: global_input.env.tx.chain_id,
            // nonce: global_input.env.tx.nonce,
            function:global_abi.function,
            action_frequency: <[f32; 16]>::try_from(ACTION_FREQUENCY.lock().unwrap().clone()).unwrap(),
            opcode_frequency: <[f32; 8]>::try_from(OPCODE_FREQUENCY.lock().unwrap().clone()).unwrap()
            // get_bytes: match <[u8; 32]>::try_from(global_abi.b.get_bytes()) {
            //     Ok(bytes) => bytes,
            //     Err(_) => [0u8; 32], // provide a default value or handle the error appropriately
            // },
        };
        (state.to_tensor(),reward)
    }
}
//ReplayBuffer存储经验元组（state, action, reward, next_state）========================================================
pub struct ReplayBuffer {
    buffer: VecDeque<(Tensor, i64, i64, Tensor)>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> ReplayBuffer {
        ReplayBuffer {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, state: Tensor, action: i64, reward: i64, next_state: Tensor) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back((state, action, reward, next_state));
    }

    pub fn sample(&self, batch_size: usize) -> Option<Vec<(Tensor, i64, i64, Tensor)>> {
        if self.buffer.len() < batch_size {
            None
        } else {
            // Some(self.buffer.iter().map(|(s, a, r, ns)| (s.copy(), *a, *r, ns.copy())).take(batch_size).collect())
            let mut rng = rand::thread_rng();
            let sampled_indices: Vec<usize> = (0..self.buffer.len()).choose_multiple(&mut rng, batch_size);
            Some(sampled_indices.iter().map(|&i| {
                let (ref s, a, r, ref ns) = self.buffer[i];
                (s.copy(), a, r, ns.copy())
            }).collect())
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

//DQN Net===============================================================================================================
#[derive(Debug)]
pub struct NoisyLinear {
    weight_mu: Tensor,
    weight_sigma: Tensor,
    bias_mu: Tensor,
    bias_sigma: Tensor,
    weight_epsilon: RefCell<Tensor>,
    bias_epsilon: RefCell<Tensor>,
}

impl NoisyLinear {
    pub fn new(vs: &nn::Path, in_features: i64, out_features: i64) -> Self {
        let weight_mu = vs.randn("weight_mu", &[out_features, in_features], 0.0, 0.017);
        let weight_sigma = vs.ones("weight_sigma", &[out_features, in_features]) * 0.017;
        let bias_mu = vs.randn("bias_mu", &[out_features], 0.0, 0.017);
        let bias_sigma = vs.ones("bias_sigma", &[out_features]) * 0.017;

        // 使用 RefCell 包装 weight_epsilon 和 bias_epsilon
        let weight_epsilon = RefCell::new(Tensor::zeros(&[out_features, in_features], (tch::Kind::Float, vs.device())));
        let bias_epsilon = RefCell::new(Tensor::zeros(&[out_features], (tch::Kind::Float, vs.device())));

        NoisyLinear {
            weight_mu,
            weight_sigma,
            bias_mu,
            bias_sigma,
            weight_epsilon,
            bias_epsilon,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        *self.weight_epsilon.borrow_mut() = Tensor::randn(&[self.weight_mu.size()[0], self.weight_mu.size()[1]], (tch::Kind::Float, self.weight_mu.device()));
        *self.bias_epsilon.borrow_mut() = Tensor::randn(&[self.bias_mu.size()[0]], (tch::Kind::Float, self.bias_mu.device()));

        let weight = &self.weight_mu + &self.weight_sigma * &*self.weight_epsilon.borrow();
        let bias = &self.bias_mu + &self.bias_sigma * &*self.bias_epsilon.borrow();

        input.matmul(&weight.tr()) + bias
    }
}

impl nn::Module for NoisyLinear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut noisy_layer = self.clone();
        NoisyLinear::forward(&mut noisy_layer, input)  // 调用结构体中的 forward 函数
    }
}


#[derive(Debug)]
pub struct DqnNet {
    // fc1: nn::Linear,
    fc1: NoisyLinear,
    fc2: nn::Linear,
    // fc2: NoisyLinear,
    fc3: nn::Linear,
    fc4: nn::Linear,
    vs: Arc<Mutex<VarStore>>,  // 存储神经网络参数的结构
}


fn kaiming_uniform_init(tensor: &mut Tensor, fan_in: i64) {
    let bound = (2.0 / (fan_in as f64)).sqrt();
    let mut tensor_clone = tensor.shallow_clone();
    no_grad(|| {
        let new_tensor = tensor_clone.uniform_(-bound, bound);
        *tensor = new_tensor;
    });

}

// Xavier initialization
fn xavier_init(tensor: &mut Tensor, fan_in: i64, fan_out: i64) {
    let bound = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt();
    let mut tensor_clone = tensor.shallow_clone();
    no_grad(|| {
        let new_tensor = tensor_clone.uniform_(-bound, bound);
        *tensor = new_tensor;
    });
}

// Zero initialization
fn zero_init(tensor: &mut Tensor) {
    no_grad(|| {
        let new_tensor = tensor.fill_(0.0);
        *tensor = new_tensor;
    });
}
impl DqnNet {
    pub fn new(vs: Arc<Mutex<nn::VarStore>>, input_dim: i64, output_dim: i32) -> DqnNet {
        let vs_clone = Arc::clone(&vs);
        let mut vs = vs.lock().unwrap();
        // let mut fc1 = nn::linear(vs.root() / "fc1", input_dim, 256, Default::default());
        let fc1 = NoisyLinear::new(&(vs.root() / "fc1"), input_dim, 256);
        let mut fc2 = nn::linear(vs.root() / "fc2", 256, 128, Default::default());
        // let fc2 = NoisyLinear::new(&(vs.root() / "fc2"), 256, 128);
        let mut fc3 = nn::linear(vs.root() / "fc3", 128, 64, Default::default());
        let mut fc4 = nn::linear(vs.root() / "fc5", 64, output_dim as i64, Default::default());

        // Kaiming均匀初始化————默认使用这个初始化
        // kaiming_uniform_init(&mut fc1.ws, input_dim);
        // kaiming_uniform_init(&mut fc2.ws, 256);
        // kaiming_uniform_init(&mut fc3.ws, 128);
        // kaiming_uniform_init(&mut fc4.ws, 64);
        // kaiming_uniform_init(&mut fc5.ws, 32);


        // Xavier initialization---效果貌似变差了
        // xavier_init(&mut fc1.ws, input_dim, 256);
        // xavier_init(&mut fc2.ws, 256, 128);
        // xavier_init(&mut fc3.ws, 128, 64);
        // xavier_init(&mut fc4.ws, 64, 32);
        // xavier_init(&mut fc5.ws, 32, output_dim);

        // Or zero initialization---效果貌似变差了
        // zero_init(&mut fc1.ws);
        // zero_init(&mut fc2.ws);
        // zero_init(&mut fc3.ws);
        // zero_init(&mut fc4.ws);
        // zero_init(&mut fc5.ws);

        DqnNet { fc1, fc2, fc3, fc4, vs: vs_clone }
    }


    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let vs = self.vs.lock().unwrap();
        vs.save(path)?;
        Ok(())
    }

    pub fn load(vs: Arc<Mutex<nn::VarStore>>, path: &str, input_dim: i64, output_dim: i32) -> Result<DqnNet, Box<dyn Error>> {
        let vs_clone = Arc::clone(&vs);
        let mut vs = vs.lock().unwrap();
        vs.load(path)?;
        Ok(DqnNet::new(vs_clone, input_dim, output_dim))
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.fc3)
            .relu()
            .apply(&self.fc4)
    }
}
//DQNAgent=============================================================================================================
// pub struct DQNAgent {
//     pub(crate) state_dim: i64,
//     pub(crate) action_dim: i32,
//     pub(crate) model: DqnNet,
//     pub(crate) replay_buffer: ReplayBuffer,
//     pub(crate) optimizer: Optimizer,
//     pub(crate) actions: Vec<i32>,
//     pub(crate) discount_factor: f32,
// }
//
// impl DQNAgent {
//     pub fn new(vs: Arc<Mutex<VarStore>>, state_dim: i64, action_dim: i32, replay_buffer_capacity: usize) -> DQNAgent {
//         // 加载模型
//         // let vs_loaded = Arc::new(Mutex::new(nn::VarStore::new(tch::Device::Cpu)));
//         // println!("我在读");
//         // let pb = ProgressBar::new(100);
//         // pb.set_style(ProgressStyle::default_bar()
//         //     .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})").expect("REASON")
//         //     .progress_chars("#>-"));
//         // let loaded_dqn_net_result = DqnNet::load(vs_loaded.clone(), "./dqn_net.ot", state_dim, action_dim);
//         // let model=match loaded_dqn_net_result {
//         //     Ok(net) => {
//         //         pb.finish_with_message("Model loaded successfully");
//         //         net
//         //     },
//         //     Err(e) => {
//         //         eprintln!("Failed to load the model: {}", e);
//         //         pb.abandon_with_message("Failed to load the model");
//         //         DqnNet::new(vs.clone(), state_dim, action_dim)
//         //     }
//         // };
//         let model=DqnNet::new(vs.clone(), state_dim, action_dim);
//         let optimizer_vs = vs.clone();
//         let optimizer = nn::Adam::default().build(&mut optimizer_vs.lock().unwrap(), 1e-7).unwrap();
//         let replay_buffer = ReplayBuffer::new(replay_buffer_capacity);
//         let actions = encode_actions();
//         let discount_factor = 0.5;  // 设置折扣因子
//
//         DQNAgent { state_dim, action_dim, model, replay_buffer, optimizer, actions, discount_factor }
//     }
//     // pub fn train(&mut self, env: &mut FuzzEnv, episodes: usize, batch_size: usize) {
//     //     for _ in 0..episodes {
//     //         let mut state = env.reset();
//     //         let mut done = false;
//     //         while !done {
//     //             let action = self.get_action(&state);
//     //             //
//     //             let (next_state, reward, is_done) = env.step(action);
//     //             self.replay_buffer.push(state, action, reward, next_state.clone(&next_state));
//     //             state = next_state;
//     //             self.update_model(batch_size);
//     //             println!("update model===========");
//     //             done = is_done;
//     //         }
//     //     }
//     // }
//     // pub fn evaluate(&self, env: &mut FuzzEnv, episodes: usize) -> f64 {
//     //     let mut total_rewards = 0.0;
//     //     for _ in 0..episodes {
//     //         let mut state = env.reset();
//     //         let mut done = false;
//     //         while !done {
//     //             let action = self.get_action(&state);
//     //             let (next_state, reward, is_done) = env.step(action);
//     //             state = next_state;
//     //             total_rewards += reward as f64;
//     //             done = is_done;
//     //         }
//     //     }
//     //     //要不要修改该类型i64????
//     //     total_rewards / episodes as f64
//     // }
//     pub fn update_model(&mut self, batch_size: usize) -> Result<(), libafl_bolts::Error>{
//         if self.replay_buffer.len() < batch_size {
//             return Err(libafl::Error::Unknown("can't update".to_string(), Default::default()))
//         }
//         let samples = self.replay_buffer.sample(batch_size).unwrap();
//         let mut states = Vec::new();
//         let mut actions = Vec::new();
//         let mut rewards = Vec::new();
//         let mut next_states = Vec::new();
//
//         for (state, action, reward, next_state) in samples.into_iter() {
//             states.push(state);
//             actions.push(action);
//             rewards.push(reward);
//             next_states.push(next_state);
//         }
//
//         let state = Tensor::stack(&states, 0);
//         let action = Tensor::from_slice(&actions).unsqueeze(-1);
//         let reward = Tensor::from_slice(&rewards);
//         let next_state = Tensor::stack(&next_states, 0);
//
//         println!("State shape: {:?}", state.size());
//         println!("Action shape: {:?}", action.unsqueeze(-1).size());
//         println!("Action values: {:?}", action);
//         //实际value
//         let curr_q_value = self.model.forward(&state).gather(-1, &action, false).squeeze_dim(-1);
//         //下一个状态下最优的动作对应的Q值
//         let next_q_value = self.model.forward(&next_state).max_dim(-1, false).0.detach();
//         //期望的Q值等于即时奖励reward加上折扣因子（这里设为0.99）乘以下一个状态的最大Q值next_q_value。
//         // let expected_q_value = reward.to_kind(Kind::Float) + 0.9             * next_q_value;
//         // 期望的Q值等于即时奖励reward加上折扣因子乘以下一个状态的最大Q值next_q_value。
//         let expected_q_value = reward.to_kind(Kind::Float) + self.discount_factor * next_q_value;
//
//         let loss = curr_q_value.mse_loss(&expected_q_value, tch::Reduction::Mean);
//         println!("loss-------------: {:?}", loss);
//         let loss_value = loss.double_value(&[]) as f32;
//         let mut loss_values = LOSS_VALUES.lock().unwrap();
//
//         if loss_values.len() > 5 {
//             let last_values: Vec<f32> = loss_values.iter().rev().take(5).cloned().collect();
//             let max_diff: f32 = last_values.windows(2).map(|w| (w[0] - w[1]).abs()).fold(0.0, f32::max);
//             // if max_diff < 0.5 {
//             //     match plot_loss_values(&loss_values) {
//             //         Ok(_) => (),
//             //         Err(e) => return Err(libafl::Error::Unknown(format!("{}", e), ErrorBacktrace::new())),
//             //     }
//             //     std::process::exit(0);
//             // }
//             //动态调整折扣因子——Frame Skipping
//             if max_diff < 10.0 {
//                 // if max_diff < 0.5 { //训练到收敛才结束，注释后 只有找到bug才结束
//                 //     match plot_loss_values(&loss_values) {
//                 //         Ok(_) => (),
//                 //         Err(e) => return Err(libafl::Error::Unknown(format!("{}", e), ErrorBacktrace::new())),
//                 //     }
//                 //     std::process::exit(0);
//                 // }
//                 // else{
//                     self.discount_factor *= 1.01;  // 如果表现好，适当增大折扣因子
//                     self.discount_factor = self.discount_factor.min(1.0);  // 保证折扣因子不超过1
//                 // }
//             } else {
//                 self.discount_factor *= 0.99;  // 如果表现不佳，适当减小折扣因子
//                 self.discount_factor = self.discount_factor.max(0.1);  // 保证折扣因子不低于0.1
//             }
//         }
//         loss_values.push(loss_value);
//
//         self.optimizer.zero_grad();
//         loss.backward();
//         self.optimizer.step();
//         Ok(())
//     }
//
//     pub fn get_action(&mut self, state: &Tensor, epsilon: f64) -> (i32,i64) {
//         //epsilon-greedy 策略：以一定的概率随机选择一个动作
//         let mut rng = rand::thread_rng();
//         let action_index;
//         let action;
//         if rng.gen::<f64>() < epsilon {
//             action_index = rng.gen_range(0..self.actions.len());
//             action = self.actions[action_index];
//         } else {
//             // 使用模型对输入的状态进行前向传播，得到Q值
//             let q_value = self.model.forward(&state.unsqueeze(0));
//             // 使用argmax函数找到Q值中最大值的索引，这个索引就是最佳的动作
//             // -1表示在最后一个维度上找最大值的索引；false表示不保持维度，即降维
//             // let action = q_value.argmax(-1, false).int64_value(&[]);
//             action_index = q_value.argmax(-1, false).int64_value(&[]) as usize % self.actions.len();
//             action = self.actions[action_index];
//
//         }
//         // Update the global action counts
//         let mut action_counts = ACTION_COUNTS.lock().unwrap();
//         let count = action_counts.entry(action).or_insert(0);
//         *count += 1;
//
//         (action, action_index as i64)
//     }
// }
pub struct DQNAgent {
    pub(crate) state_dim: i64,
    pub(crate) action_dim: i32,
    pub(crate) model: DqnNet,
    pub(crate) replay_buffer: ReplayBuffer,
    pub(crate) optimizer: Optimizer,
    pub(crate) actions: Vec<i32>,
    pub(crate) discount_factor: f64,
}

impl DQNAgent {
    pub fn new(vs: Arc<Mutex<VarStore>>, state_dim: i64, action_dim: i32, replay_buffer_capacity: usize, alpha: f64) -> DQNAgent {
        let model = DqnNet::new(vs.clone(), state_dim, action_dim);
        let optimizer_vs = vs.clone();
        let optimizer = nn::Adam::default().build(&mut optimizer_vs.lock().unwrap(), 1e-7).unwrap();
        let replay_buffer = ReplayBuffer::new(replay_buffer_capacity);
        let actions = encode_actions();
        let discount_factor = 0.5;

        DQNAgent { state_dim, action_dim, model, replay_buffer, optimizer, actions, discount_factor }
    }

    pub fn update_model(&mut self, batch_size: usize) -> Result<(), libafl_bolts::Error> {
    if self.replay_buffer.len() < batch_size {
        return Err(libafl::Error::Unknown("can't update".to_string(), Default::default()));
    }
    let samples = self.replay_buffer.sample(batch_size).unwrap();
    let mut states = Vec::new();
    let mut actions = Vec::new();
    let mut rewards = Vec::new();
    let mut next_states = Vec::new();

    for (state, action, reward, next_state) in samples.into_iter() {
        states.push(state);
        actions.push(action);
        rewards.push(reward);
        next_states.push(next_state);
    }

    let state = Tensor::stack(&states, 0);
    let action = Tensor::from_slice(&actions).unsqueeze(-1);
    let reward = Tensor::from_slice(&rewards);
    let next_state = Tensor::stack(&next_states, 0);

    let curr_q_value = self.model.forward(&state).gather(-1, &action, false).squeeze_dim(-1);
    let next_q_value = self.model.forward(&next_state).max_dim(-1, false).0.detach();
    let expected_q_value = reward.to_kind(Kind::Float) + self.discount_factor * next_q_value;

    let loss = curr_q_value.mse_loss(&expected_q_value, tch::Reduction::Mean);
    let loss_value = loss.double_value(&[]) as f32;
    let mut loss_values = LOSS_VALUES.lock().unwrap();

    if loss_values.len() > 5 {
        let last_values: Vec<f32> = loss_values.iter().rev().take(5).cloned().collect();
        let max_diff: f32 = last_values.windows(2).map(|w| (w[0] - w[1]).abs()).fold(0.0, f32::max);
        if max_diff < 10.0 {
            self.discount_factor *= 1.01;
            self.discount_factor = self.discount_factor.min(1.0);
        } else {
            self.discount_factor *= 0.99;
            self.discount_factor = self.discount_factor.max(0.1);
        }
    }
    loss_values.push(loss_value);

    self.optimizer.zero_grad();
    loss.backward();
    self.optimizer.step();

    Ok(())
}

    pub fn get_action(&mut self, state: &Tensor, epsilon: f64) -> (i32, i64) {
        // epsilon-greedy 策略：以一定的概率随机选择一个动作
        let mut rng = rand::thread_rng();
        let action_index;
        let action;
        if rng.gen::<f64>() < epsilon {
            action_index = rng.gen_range(0..self.actions.len());
            action = self.actions[action_index];
        } else {
            // 使用模型对输入的状态进行前向传播，得到Q值
            let q_value = self.model.forward(&state.unsqueeze(0));
            // 使用argmax函数找到Q值中最大值的索引，这个索引就是最佳的动作
            // -1表示在最后一个维度上找最大值的索引；false表示不保持维度，即降维
            action_index = q_value.argmax(-1, false).int64_value(&[]) as usize % self.actions.len();
            action = self.actions[action_index];
        }
        // Update action counts
        update_action_frequency(action);
        update_opcode_frequency();

        (action, action_index as i64)
    }
}